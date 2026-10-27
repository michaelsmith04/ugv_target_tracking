#!/usr/bin/env python3
"""Re-publish a PoseWithCovarianceStamped with an honest covariance.
 
Why: in the recorded bag the pan-tilt estimates disagreed with the OAK by
~0.3-0.45 m (1 sigma) while reporting ~0.15 m, and their scatter in `odom`
grew with the rover's yaw rate (0.36 m standing still -> 0.77 m at 1-2 rad/s).
The UKF therefore over-trusted them and turned the noise into velocity.
 
New covariance (over the inflated axes A, a sub-block of the 6x6 matrix):
    C'[A,A] = var_scale * C[A,A]  +  diag((range * |yaw_rate| * rot_dt)^2)
  * var_scale : covers the static under-estimate (about 8-10 for the pan-tilt,
                ~1 for the OAK once its depth polynomial is fixed).
                The whole block is scaled, off-diagonals included, so the
                correlation between the inflated axes is preserved and the
                result is always positive semi-definite.
  * rot_dt    : effective timing error (s) between the data used and the
                odometry yaw used. At 2.5 m range and 3 rad/s, 100 ms is
                0.75 m of lateral error.
                Measured: ~0.1 s for the pan-tilt, ~0.03 s for the OAK
                (bags 17 and 20, after correcting the OAK depth bias).
`yaw_rate` is looked up at the measurement's STAMP (these estimates are ~1.3-1.9 s
old when they arrive), from /odom twist.
 
Optionally drop measurements taken while the rover turned faster than
`drop_above_yaw_rate` (<= 0 disables).
 
Run one instance per sensor (give each its own node name), then point the UKF
poseN topics at the outputs:
 
  python3 pose_cov_inflate.py --ros-args -r __node:=pt_cov_inflate \
      -p input_topic:=/pantilt/pose_estimates \
      -p output_topic:=/pantilt/pose_inflated
 
  # OAK: optical frame (x lateral, z depth). Its fitted covariance is already
  # roughly right for the static case, so var_scale ~ 1 and a small rot_dt.
  python3 pose_cov_inflate.py --ros-args -r __node:=oak_cov_inflate \
      -p input_topic:=/target/oak/pose_gated \
      -p output_topic:=/target/oak/pose_inflated \
      -p var_scale:=1.0 -p rot_dt:=0.03 \
      -p cov_indices:="[0,14]" -p range_fields:="['x','z']"
 
`cov_indices` are DIAGONAL indices of the row-major 6x6 covariance: 0=x, 7=y,
14=z (multiples of 7).  `range_fields` are the position fields that form the
range to the target in the message's frame.
"""
import bisect
import collections
import math
 
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
 
 
def inflate_variance(var, var_scale, rng, yaw_rate, rot_dt):
    """Single-axis form (kept for the offline tests against the recorded bag)."""
    return var_scale * var + (rng * abs(yaw_rate) * rot_dt) ** 2
 
 
def axes_from_diag_indices(cov_indices):
    """Diagonal index (7*a) -> axis number a of the 6x6 covariance."""
    axes = []
    for idx in cov_indices:
        if idx % 7 != 0 or not 0 <= idx <= 35:
            raise ValueError(
                f'cov_indices must be diagonal indices of a 6x6 matrix '
                f'(0, 7, 14, 21, 28, 35); got {idx}')
        axes.append(idx // 7)
    return axes
 
 
def inflate_covariance(cov, cov_indices, var_scale, rng, yaw_rate, rot_dt):
    """Return a new flat 36-list with the block over `cov_indices` inflated.
 
    C' = var_scale * C  (on the block)  +  diag(extra),  extra = (rng*|w|*rot_dt)^2
    The rest of the matrix is untouched.
    """
    out = list(cov)
    axes = axes_from_diag_indices(cov_indices)
    extra = (rng * abs(yaw_rate) * rot_dt) ** 2
    for a in axes:
        for b in axes:
            out[6 * a + b] = var_scale * cov[6 * a + b]
        out[7 * a] += extra
    return out
 
 
class YawRateBuffer:
    """Time-indexed yaw-rate history with linear interpolation."""
 
    def __init__(self, horizon=6.0):
        self.horizon = horizon
        self.t = collections.deque()
        self.w = collections.deque()
 
    def add(self, t, w):
        self.t.append(t)
        self.w.append(w)
        while self.t and t - self.t[0] > self.horizon:
            self.t.popleft()
            self.w.popleft()
 
    def at(self, t):
        if not self.t:
            return 0.0
        ts = list(self.t)
        i = bisect.bisect_left(ts, t)
        if i <= 0:
            return self.w[0]
        if i >= len(ts):
            return self.w[-1]
        t0, t1 = ts[i - 1], ts[i]
        a = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
        return (1 - a) * self.w[i - 1] + a * self.w[i]
 
 
class PoseCovInflateOak(Node):
    def __init__(self):
        super().__init__('pose_cov_inflate_oak')
        self.declare_parameter('input_topic', '/target/oak/pose_gated')
        self.declare_parameter('output_topic', '/target/oak/pose_inflated')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('var_scale', 1.0)
        self.declare_parameter('rot_dt', 0.03)
        self.declare_parameter('drop_above_yaw_rate', 0.0)  # rad/s, <=0 disables
        self.declare_parameter('cov_indices', [0, 14])        # diagonal entries to inflate
        self.declare_parameter('range_fields', ['x', 'z'])   # position fields forming the range
 
        self.cov_indices = [int(i) for i in self.p('cov_indices')]
        axes_from_diag_indices(self.cov_indices)  # fail fast on a bad parameter
 
        self.buf = YawRateBuffer()
        self.pub = self.create_publisher(
            PoseWithCovarianceStamped, self.get_parameter('output_topic').value, 10)
        self.create_subscription(Odometry, self.get_parameter('odom_topic').value, self.on_odom, 50)
        self.create_subscription(
            PoseWithCovarianceStamped, self.get_parameter('input_topic').value, self.on_pose, 10)
        self.dropped = 0
        self.get_logger().info(
            f"{self.p('input_topic')} -> {self.p('output_topic')} | "
            f"var_scale={self.p('var_scale')} rot_dt={self.p('rot_dt')} "
            f"cov_indices={self.cov_indices} range_fields={list(self.p('range_fields'))} "
            f"drop_above_yaw_rate={self.p('drop_above_yaw_rate')}")
 
    def p(self, n):
        return self.get_parameter(n).value
 
    @staticmethod
    def stamp_s(h):
        return h.stamp.sec + h.stamp.nanosec * 1e-9
 
    def on_odom(self, msg):
        self.buf.add(self.stamp_s(msg.header), msg.twist.twist.angular.z)
 
    def on_pose(self, msg):
        w = self.buf.at(self.stamp_s(msg.header))
        lim = self.p('drop_above_yaw_rate')
        if lim > 0 and abs(w) > lim:
            self.dropped += 1
            self.get_logger().info(
                f'dropped measurement at yaw rate {w:.2f} rad/s ({self.dropped} total)',
                throttle_duration_sec=2.0)
            return
        pos = msg.pose.pose.position
        rng = math.sqrt(sum(getattr(pos, f) ** 2 for f in self.p('range_fields')))
        out = PoseWithCovarianceStamped()
        out.header = msg.header
        out.pose.pose = msg.pose.pose
        out.pose.covariance = inflate_covariance(
            msg.pose.covariance, self.cov_indices,
            self.p('var_scale'), rng, w, self.p('rot_dt'))
        self.pub.publish(out)
 
 
def main():
    rclpy.init()
    node = PoseCovInflateOak()
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
 