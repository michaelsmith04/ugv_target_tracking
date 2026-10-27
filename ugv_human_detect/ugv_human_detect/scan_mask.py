import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

class ScanMaskNode(Node):
    def __init__(self):
        super().__init__('scan_mask_node')
        # Raw index bounds covering the pan-tilt bracket's blind zone
        # AND its close-range reflection spike, with margin on both sides.
        # Confirmed from a live scan dump: solid NaN 63-177, 6cm spike 178-180.
        self.declare_parameter('mask_index_min', 55)
        self.declare_parameter('mask_index_max', 190)
        self.i_min = self.get_parameter('mask_index_min').value
        self.i_max = self.get_parameter('mask_index_max').value

        self.pub = self.create_publisher(LaserScan, '/scan_masked', 10)
        self.sub = self.create_subscription(LaserScan, '/scan', self._cb, 10)

    def _cb(self, msg: LaserScan):
        ranges = list(msg.ranges)
        n = len(ranges)
        front_center_idx = -125  # corrected per tf: base_link forward = -90 deg in lidar frame
        half_width_idx = n // 4
        ramp_len = 15  # indices to taper over, tune based on angle_increment

        for i in range(n):
            d = min((i - front_center_idx) % n, (front_center_idx - i) % n)
            if d > half_width_idx + ramp_len:
                ranges[i] = float('nan')
            elif d > half_width_idx:
                # linear ramp: ranges[i] grows toward 29.99 as it approaches the true cutoff
                t = (d - half_width_idx) / ramp_len
                r = ranges[i]
                if r is not None and r == r and not (r != r):  # skip if already nan
                    ranges[i] = r + t * (29.99 - r) if r == r else float('nan')

        out = LaserScan()
        out.header = msg.header
        out.angle_min = msg.angle_min
        out.angle_max = msg.angle_max
        out.angle_increment = msg.angle_increment
        out.time_increment = msg.time_increment
        out.scan_time = msg.scan_time
        out.range_min = msg.range_min
        out.range_max = msg.range_max
        out.ranges = ranges
        out.intensities = msg.intensities
        self.pub.publish(out)


def main():
    rclpy.init()
    node = ScanMaskNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()