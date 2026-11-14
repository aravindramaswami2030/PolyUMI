"""
Live log-mel spectrogram of the finger mic, as a scrolling image for Foxglove.

This is the inference-side twin of the ``/finger/logmel`` channel that
``pingest export`` writes into an MCAP: it exists so a human can see at a glance whether the
contact mic is picking anything up, next to the finger camera and the raw waveform. **Nothing
trains on it.** The spectrogram a policy sees is computed inside the training container from the
raw waveform, after waveform-domain augmentation -- see ``docs/maniwav-audio-policy.md``.

Parameters mirror ``ingest/config/contact_audio.yaml``'s ``logmel`` block, which in turn follows
ManiWAV's, so the picture resembles what the model is fed: 64 HTK mel bins over a 25 ms frame at
a 10 ms hop, natural log of the power spectrum, triangular unnormalised (Kaldi/AST) filterbank.
The mel maths is mirrored here rather than imported, for the same reason
:mod:`polyumi_ros2.audio_preproc` mirrors its contract: the ingest and ROS packages run under
different interpreters and cannot share an import. Unlike that module this one is only a
diagnostic, so it is not required to agree with ingest bit-for-bit.

**On which channel to draw.** ``channel`` defaults to 0, the piezo contact mic, because that is
what becomes ``mic_0``. Note that ``docs/contact-mic-audio.md`` records the two channels as *not*
independent on this HAT -- the overlay routes the Mic jack to both LINPUT1 and RINPUT1, so one
microphone can reach both -- and the left channel ships heavily attenuated. If the image looks
flat, draw channel 1 before concluding the sensor is dead.
"""

from __future__ import annotations

import numpy as np
import rclpy
from foxglove_msgs.msg import RawAudio
from rclpy.node import Node
from sensor_msgs.msg import Image

# ingest/config/contact_audio.yaml :: logmel
N_FFT = 400
HOP_LENGTH = 160
WIN_LENGTH = 400
N_MELS = 64
FMIN = 20.0
FMAX = 8000.0
LOG_OFFSET = 1.0e-6


def hz_to_mel(hz: np.ndarray | float) -> np.ndarray:
    """Convert Hz to the HTK mel scale."""
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray:
    """Convert HTK mel back to Hz."""
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(sample_rate: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    """
    Triangular HTK mel filterbank of shape ``(n_mels, n_fft // 2 + 1)``.

    Unnormalised -- each triangle peaks at 1.0 rather than being scaled to unit area, which is
    the Kaldi convention the AST feature extractor uses. The log of an area-normalised bank sits
    at a different offset, so a diagnostic drawn one way would not resemble a model fed the other.
    """
    nyquist = sample_rate / 2.0
    hi = min(fmax, nyquist)
    edges_hz = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(hi), n_mels + 2))
    bin_hz = np.linspace(0.0, nyquist, n_fft // 2 + 1)

    bank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float64)
    for m in range(n_mels):
        left, centre, right = edges_hz[m], edges_hz[m + 1], edges_hz[m + 2]
        rising = (bin_hz - left) / max(centre - left, 1e-12)
        falling = (right - bin_hz) / max(right - centre, 1e-12)
        bank[m] = np.clip(np.minimum(rising, falling), 0.0, None)
    return bank


def log_mel(audio: np.ndarray, sample_rate: int, bank: np.ndarray) -> np.ndarray:
    """
    Log-mel spectrogram of a mono waveform, shape ``(n_hops, n_mels)`` float32.

    Time-major, matching the layout the AST feature extractor produces. Returns an empty
    ``(0, n_mels)`` when the waveform is shorter than one analysis frame.
    """
    audio = np.asarray(audio, dtype=np.float64).reshape(-1)
    if audio.size < N_FFT:
        return np.zeros((0, N_MELS), dtype=np.float32)

    window = np.hanning(WIN_LENGTH + 1)[:WIN_LENGTH]  # periodic, matching sym=False
    if WIN_LENGTH < N_FFT:
        pad = N_FFT - WIN_LENGTH
        window = np.pad(window, (pad // 2, pad - pad // 2))

    n_hops = 1 + (audio.size - N_FFT) // HOP_LENGTH
    idx = np.arange(N_FFT)[None, :] + HOP_LENGTH * np.arange(n_hops)[:, None]
    frames = audio[idx] * window
    power = np.abs(np.fft.rfft(frames, n=N_FFT, axis=1)) ** 2
    return np.log(power @ bank.T + LOG_OFFSET).astype(np.float32)


def colourise(spec: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """
    Map a (n_mels, n_hops) spectrogram to an RGB image, low frequencies at the bottom.

    A blue-through-yellow ramp rather than a rainbow: monotonic in lightness, so loud reads as
    bright in both themes and in greyscale.
    """
    norm = np.clip((spec - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    r = np.clip(2.2 * norm - 0.45, 0.0, 1.0)
    g = np.clip(1.7 * norm - 0.25, 0.0, 1.0)
    b = np.clip(1.1 - 1.6 * np.abs(norm - 0.25), 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1) * 255.0
    return np.flipud(rgb).astype(np.uint8)  # flip so bin 0 (lowest Hz) is at the bottom


class LogMelNode(Node):
    """Keeps a rolling audio window and republishes it as a scrolling log-mel image."""

    def __init__(self) -> None:
        """Declare parameters, size the ring buffer, and wire the subscription and timer."""
        super().__init__('logmel_node')
        self.declare_parameter('audio_topic', '/pi/audio/raw')
        self.declare_parameter('image_topic', '/pi/finger/logmel')
        self.declare_parameter('channel', 0)
        self.declare_parameter('window_s', 4.0)
        self.declare_parameter('publish_hz', 10.0)
        # Fixed dB-ish range keeps the picture steady; autoscaling makes a quiet room look loud.
        self.declare_parameter('log_floor', -11.0)
        self.declare_parameter('log_ceil', 2.0)

        self.channel = int(self.get_parameter('channel').value)
        self.window_s = float(self.get_parameter('window_s').value)
        self.lo = float(self.get_parameter('log_floor').value)
        self.hi = float(self.get_parameter('log_ceil').value)

        self.sample_rate: int | None = None
        self.bank: np.ndarray | None = None
        self.buf = np.zeros(0, dtype=np.float32)
        self.stamp = None

        self.pub = self.create_publisher(Image, self.get_parameter('image_topic').value, 1)
        self.create_subscription(RawAudio, self.get_parameter('audio_topic').value, self._on_audio, 10)
        hz = float(self.get_parameter('publish_hz').value)
        self.create_timer(1.0 / hz, self._publish)
        self.get_logger().info(
            f'log-mel on channel {self.channel}, {self.window_s:.1f}s window -> '
            f'{self.get_parameter("image_topic").value}'
        )

    def _on_audio(self, msg: RawAudio) -> None:
        """Append this chunk's selected channel to the ring buffer."""
        channels = max(int(msg.number_of_channels), 1)
        if self.channel >= channels:
            return
        if self.sample_rate != msg.sample_rate:
            self.sample_rate = int(msg.sample_rate)
            self.bank = mel_filterbank(self.sample_rate, N_FFT, N_MELS, FMIN, FMAX)
            self.buf = np.zeros(0, dtype=np.float32)

        pcm = np.frombuffer(bytes(msg.data), dtype=np.int16)
        mono = pcm[self.channel :: channels].astype(np.float32) / 32768.0
        keep = int(self.window_s * self.sample_rate)
        self.buf = np.concatenate([self.buf, mono])[-keep:]
        self.stamp = msg.timestamp

    def _publish(self) -> None:
        """Compute the spectrogram over the current window and publish it as an Image."""
        if self.bank is None or self.sample_rate is None or self.buf.size < N_FFT:
            return
        spec = log_mel(self.buf, self.sample_rate, self.bank)
        if spec.shape[0] == 0:
            return

        rgb = colourise(spec.T, self.lo, self.hi)  # (n_mels, n_hops, 3)
        msg = Image()
        if self.stamp is not None:
            msg.header.stamp = self.stamp
        msg.header.frame_id = 'finger'
        msg.height, msg.width = rgb.shape[0], rgb.shape[1]
        msg.encoding = 'rgb8'
        msg.is_bigendian = 0
        msg.step = rgb.shape[1] * 3
        msg.data = rgb.tobytes()
        self.pub.publish(msg)


def main() -> None:
    """Spin the log-mel diagnostic node."""
    rclpy.init()
    node = LogMelNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
