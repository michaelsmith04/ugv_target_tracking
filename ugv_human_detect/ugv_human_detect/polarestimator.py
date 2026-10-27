import math
import rclpy
from rclpy.node import Node

from ugv_human_detect_msgs.msg import SourcedPoint, PolarEstimate


class _SourceFilterState:
    """Persistence/jump-reject filter state, tracked independently per sensor
    so LiDAR noise can't reset OAK-D's filter or vice versa."""

    def __init__(self):
        self.consecutive_count = 0
        self.last_published_r = None


class PolarEstimatorNode(Node):
    """
    ONE node replaces the three separate "Conversion to [...] estimate"
    boxes in the diagram.

    Single subscription to /human_estimates/points. Which of r / theta /
    pitch are actually meaningful depends on the source:
      - lidar:    theta + r            (2D only, no pitch)
      - oak:      theta + r + pitch    (full 3D)
      - pan_tilt: theta + pitch only   (no range - is_direction=True)

    That dispatch is a few lines of branching in one callback, instead of
    three separate node processes each re-implementing the range/bearing
    math and the persistence filter.
    """

    def __init__(self):
        super().__init__('polar_estimator_node')

        self.declare_parameter('input_topic', '/human_estimates/points')
        self.declare_parameter('output_topic', '/human_estimates/polar')
        self.declare_parameter('min_valid_range', 0.3)
        self.declare_parameter('max_valid_range', 10.0)
        self.declare_parameter('persistence_frames', 3)
        self.declare_parameter('jump_reject_m', 1.0)

        self.min_valid_range = self.get_parameter('min_valid_range').value
        self.max_valid_range = self.get_parameter('max_valid_range').value
        self.persistence_frames = self.get_parameter('persistence_frames').value
        self.jump_reject_m = self.get_parameter('jump_reject_m').value

        self._filter_state = {}  # source (str) -> _SourceFilterState

        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value

        self.sub = self.create_subscription(SourcedPoint, input_topic, self._on_point, 10)
        self.pub = self.create_publisher(PolarEstimate, output_topic, 10)

        self.get_logger().info(f'Consolidated polar estimator ready: {input_topic} -> {output_topic}')

    def _on_point(self, msg: SourcedPoint):
        x, y, z = msg.point.x, msg.point.y, msg.point.z

        out = PolarEstimate()
        out.header = msg.header
        out.source = msg.source
        out.theta = math.atan2(y, x)  # bearing is always meaningful

        if msg.is_direction:
            # Pan-tilt: no range measurement at all, so there is nothing
            # to range-filter. Pitch comes straight from the z component
            # of the (already base_link-transformed) unit direction.
            out.has_range = False
            out.r = 0.0
            out.has_pitch = True
            out.pitch = math.atan2(z, math.hypot(x, y))
            self.get_logger().info(
                f'[{msg.source}] bearing={math.degrees(out.theta):.1f}deg '
                f'pitch={math.degrees(out.pitch):.1f}deg (no range)'
            )
            self.pub.publish(out)
            return

        r_full = math.sqrt(x ** 2 + y ** 2 + z ** 2)
        r_ground = math.hypot(x, y)

        if not self._range_filter_ok(msg.source, r_full):
            return

        out.has_range = True
        out.r = r_full

        if msg.source == 'oak':
            # OAK-D has real depth in all 3 axes -> pitch is meaningful.
            out.has_pitch = True
            out.pitch = math.atan2(z, r_ground) if r_ground > 1e-6 else 0.0
        else:
            # LiDAR is 2D by construction (z forced to 0 in the transform
            # node) -> no pitch estimate.
            out.has_pitch = False
            out.pitch = 0.0

        self.get_logger().info(
            f'[{msg.source}] r={out.r:.2f}m bearing={math.degrees(out.theta):.1f}deg'
            + (f' pitch={math.degrees(out.pitch):.1f}deg' if out.has_pitch else '')
        )

        self.pub.publish(out)

    def _range_filter_ok(self, source: str, r: float) -> bool:
        state = self._filter_state.setdefault(source, _SourceFilterState())

        if not (self.min_valid_range <= r <= self.max_valid_range):
            state.consecutive_count = 0
            return False

        # Reject a single-frame spike: a reading that jumps too far from
        # the last accepted value for THIS source is more likely a
        # false-positive detection than the person teleporting.
        if (state.last_published_r is not None
                and abs(r - state.last_published_r) > self.jump_reject_m
                and state.consecutive_count < self.persistence_frames):
            state.consecutive_count = 0
            return False

        state.consecutive_count += 1
        if state.consecutive_count < self.persistence_frames:
            return False

        state.last_published_r = r
        return True


def main(args=None):
    rclpy.init(args=args)
    node = PolarEstimatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()