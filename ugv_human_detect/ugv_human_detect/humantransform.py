import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from geometry_msgs.msg import PoseArray, PointStamped, Vector3Stamped, Point
from vision_msgs.msg import Detection3DArray
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_point, do_transform_vector3

from ugv_human_detect_msgs.msg import SourcedPoint

# MobileNet-SSD (VOC label map, as used by the depthai spatial detection
# network) class index for "person".
PERSON_CLASS_ID = '15'

class HumanDetectionTransformNode(Node):
    """
    ONE node replaces the three separate "Transformation from X link to
    base link" boxes in the diagram.

    ROS2 subscriptions are still bound to a fixed message type, so each
    sensor still needs its own callback - that part is unavoidable. What's
    consolidated is everything else: one TF buffer/listener, one output
    topic, and one shared `_transform_and_publish()` that does the actual
    lookup + transform + republish, instead of three copies of that logic
    living in three separate node processes.

    LiDAR and OAK-D detections carry real range info, so they're
    transformed as points (translation + rotation). The pan-tilt rig only
    gives a bearing/elevation direction (no range), so it's transformed as
    a direction vector (rotation only) - transforming it as a point would
    silently and incorrectly inject the sensor's translation offset into
    an angle-only measurement.

    Output: /human_estimates/points (SourcedPoint), always in target_frame.
    """

    def __init__(self):
        super().__init__('human_detection_transform_node')

        self.declare_parameter('target_frame', 'base_link')
        self.declare_parameter('tf_timeout_sec', 0.2)
        self.declare_parameter('lidar_topic', '/dr_spaam_detections')
        self.declare_parameter('oak_topic', '/oak/nn/spatial_detections')
        # NOTE: placeholder until you confirm the real pan-tilt output.
        # Assumed here to be a unit bearing/elevation direction vector in
        # pt_link, e.g. published by whatever node already reads the servo
        # angles + YOLO bbox offset. If it's something else, only this
        # topic name/type and _on_pan_tilt() need to change.
        self.declare_parameter('pan_tilt_topic', '/pt/human_direction')
        self.declare_parameter('output_topic', '/human_estimates/points')
        self.declare_parameter('min_target_range', 0.35)

        self.target_frame = self.get_parameter('target_frame').value
        self.tf_timeout = self.get_parameter('tf_timeout_sec').value
        self.min_target_range = self.get_parameter('min_target_range').value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        output_topic = self.get_parameter('output_topic').value
        self.pub = self.create_publisher(SourcedPoint, output_topic, 10)

        self.create_subscription(
            PoseArray, self.get_parameter('lidar_topic').value, self._on_lidar, 10)
        self.create_subscription(
            Detection3DArray, self.get_parameter('oak_topic').value, self._on_oak, 10)
        self.create_subscription(
            Vector3Stamped, self.get_parameter('pan_tilt_topic').value, self._on_pan_tilt, 10)

        self.get_logger().info(
            f'Consolidated transform node ready -> {output_topic} '
            f'(target frame: {self.target_frame})'
        )

    # ---------------- per-source normalization callbacks ----------------

    def _on_lidar(self, msg: PoseArray):
        if not msg.poses:
            return

        # Same "reject readings right at the sensor origin" filter as
        # before - almost certainly chassis/mount, not a person.
        min_range_sq = self.min_target_range ** 2
        candidates = [
            p for p in msg.poses
            if p.position.x ** 2 + p.position.y ** 2 > min_range_sq
        ]
        if not candidates:
            return

        target = min(candidates, key=lambda p: p.position.x ** 2 + p.position.y ** 2)
        point = Point(x=target.position.x, y=target.position.y, z=0.0)
        self._transform_and_publish(msg.header, point, source='lidar', is_direction=False)

    def _on_oak(self, msg: Detection3DArray):
        # In vision_msgs/Detection3D, pose lives on each *result*
        # (ObjectHypothesisWithPose), not on the Detection3D itself -
        # results[i].pose.pose.position / results[i].hypothesis.class_id.
        best_result = None
        best_score = -1.0
        for d in msg.detections:
            for r in d.results:
                #if r.hypothesis.class_id == PERSON_CLASS_ID and r.hypothesis.score > best_score:
                if r.hypothesis.class_id == PERSON_CLASS_ID:
                    #best_score = r.hypothesis.score
                    #best_result = r
                    p = r.pose.pose.position
                    self.get_logger().info(
                        f'candidate score={r.hypothesis.score:.2f} x={p.x:.2f} y={p.y:.2f} z={p.z:.2f}'
                    )

        if best_result is None:
            return

        pos = best_result.pose.pose.position
        point = Point(x=pos.x, y=pos.y, z=pos.z)
        self._transform_and_publish(msg.header, point, source='oak', is_direction=False)

    def _on_pan_tilt(self, msg: Vector3Stamped):
        point = Point(x=msg.vector.x, y=msg.vector.y, z=msg.vector.z)
        self._transform_and_publish(msg.header, point, source='pan_tilt', is_direction=True)

    # ---------------- shared TF lookup + transform + publish ----------------

    def _transform_and_publish(self, header, point: Point, source: str, is_direction: bool):
        # lidar/oak sensor mounts are rigid (no relative motion vs the
        # chassis), so looking up the *latest* transform instead of one
        # at the message's exact stamp is both harmless and sidesteps
        # "extrapolation into the future/past" errors caused by clock
        # skew between the main processor and sensors running on
        # separate compute (e.g. the OAK). pan_tilt is NOT rigid - its
        # bearing depends on the live servo angle, so pt_link -> base_link
        # genuinely changes over time and needs the real stamp; using
        # "latest" there would introduce lag whenever the servo moves.
        lookup_time = Time() if source in ('lidar', 'oak') else Time.from_msg(header.stamp)
        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                header.frame_id,
                lookup_time,
                timeout=Duration(seconds=self.tf_timeout),
            )
        except Exception as e:
            self.get_logger().warn(
                f'TF lookup {header.frame_id} -> {self.target_frame} failed ({source}): {e}'
            )
            return

        if is_direction:
            v_in = Vector3Stamped()
            v_in.header = header
            v_in.vector.x, v_in.vector.y, v_in.vector.z = point.x, point.y, point.z
            v_out = do_transform_vector3(v_in, transform)
            out_point = Point(x=v_out.vector.x, y=v_out.vector.y, z=v_out.vector.z)
        else:
            p_in = PointStamped()
            p_in.header = header
            p_in.point = point
            p_out = do_transform_point(p_in, transform)
            out_point = p_out.point

        out = SourcedPoint()
        out.header.frame_id = self.target_frame
        out.header.stamp = header.stamp
        out.source = source
        out.point = out_point
        out.is_direction = is_direction
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = HumanDetectionTransformNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()