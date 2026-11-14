"""
Republish one channel of the Pi's stereo audio as a mono stream.

The Pi always captures stereo and must keep doing so: the WM8960 advertises ``CHANNELS: 2``
only, and a mono request froze capture within seconds (see ``docs/contact-mic-audio.md``).
Channel 1 also has to keep flowing for data collection, because
``ingest/polyumi_ingest/preproc/time_sync.py`` cross-correlates it against the GoPro audio to
find the sync chirp. So neither the capture nor the bridge can be narrowed to the contact mic;
the split belongs here, downstream of both.

What this buys is a topic carrying only the piezo contact mic, so a viewer draws one waveform
instead of two and an operator is never left guessing which of the pair the policy listens to.
That question has a fixed answer -- ``mic_0`` is channel 0, see :mod:`polyumi_ros2.audio_preproc`
-- and this node makes it visible rather than implied.

Nothing downstream of the policy changes: this publishes a *new* topic and leaves the stereo one
alone, so ingest, the sync chirp and ``policy_client_node`` all carry on reading what they read
before.

**A caveat on what channel 1 actually is.** ``docs/contact-mic-audio.md`` records the two
channels as not independent on this HAT -- the overlay routes the Mic jack to both LINPUT1 and
RINPUT1, giving a measured L/R correlation of 0.87-0.98 -- so ``finger_air`` may not be a
separate air mic at all. That does not affect channel 0, which is what this node exists to
isolate, but it is why the default is 0 and why "the other channel" is not a fallback sensor.
"""

from __future__ import annotations

import numpy as np
import rclpy
from foxglove_msgs.msg import RawAudio
from rclpy.node import Node


class AudioChannelNode(Node):
    """Subscribes to an interleaved stereo RawAudio topic and republishes one channel as mono."""

    def __init__(self) -> None:
        """Declare parameters and wire the subscription to the republisher."""
        super().__init__('audio_channel_node')
        self.declare_parameter('input_topic', '/pi/audio/raw')
        self.declare_parameter('output_topic', '/pi/audio/piezo')
        self.declare_parameter('channel', 0)

        self.channel = int(self.get_parameter('channel').value)
        out = str(self.get_parameter('output_topic').value)
        self.pub = self.create_publisher(RawAudio, out, 10)
        self.create_subscription(RawAudio, str(self.get_parameter('input_topic').value), self._on_audio, 10)
        self._warned = False
        self.get_logger().info(
            f'republishing channel {self.channel} '
            f'({"piezo contact mic" if self.channel == 0 else "second channel"}) as mono -> {out}'
        )

    def _on_audio(self, msg: RawAudio) -> None:
        """Deinterleave the selected channel and republish it, preserving the source timestamp."""
        channels = max(int(msg.number_of_channels), 1)
        if self.channel >= channels:
            if not self._warned:
                self.get_logger().warn(
                    f'channel {self.channel} requested but stream carries {channels}; publishing nothing'
                )
                self._warned = True
            return

        pcm = np.frombuffer(bytes(msg.data), dtype=np.int16)
        mono = np.ascontiguousarray(pcm[self.channel :: channels])

        out = RawAudio()
        out.timestamp = msg.timestamp  # the Pi's capture instant, passed through untouched
        out.data = mono.tobytes()
        out.format = msg.format
        out.sample_rate = msg.sample_rate
        out.number_of_channels = 1
        self.pub.publish(out)


def main() -> None:
    """Spin the channel-splitting node."""
    rclpy.init()
    node = AudioChannelNode()
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
