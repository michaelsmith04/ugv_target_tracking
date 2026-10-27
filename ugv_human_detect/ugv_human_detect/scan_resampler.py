#!/usr/bin/env python3
"""
scan_resampler: /scan (variable length) -> /scan_masked (fixed length).

Every output scan has exactly `n_beams` ranges on a fixed angular grid
(angle_min, angle_max, angle_increment identical on every message), which is
what slam_toolbox and DR-SPAAM want.

Parameters
----------
input_topic        (str)    default '/scan'
output_topic       (str)    default '/scan_masked'
n_beams            (int)    default 504   output size; set to what DR-SPAAM needs
out_angle_min      (float)  default -pi
out_angle_max      (float)  default  pi   (span >= ~2*pi is treated as a full circle:
                                           increment = span / n_beams, no duplicate
                                           first/last beam)
interpolation      (str)    'nearest' (default) or 'linear'
                            nearest: never invents points, preserves sharp edges
                            (good for DR-SPAAM). linear: smooths, but only between
                            two valid neighbours closer than `jump_threshold`.
jump_threshold     (float)  default 0.3 m, linear mode only
max_gap_factor     (float)  default 2.5   output beams further than this many input
                            increments from any input beam become NaN (matters only
                            when the output FOV is larger than the input coverage)
"""
import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

TWO_PI = 2.0 * math.pi


class ScanResampler(Node):
    def __init__(self):
        super().__init__('scan_resampler')

        self.declare_parameter('input_topic', '/scan')
        self.declare_parameter('output_topic', '/scan_resampled')
        self.declare_parameter('n_beams', 504)
        self.declare_parameter('out_angle_min', -math.pi)
        self.declare_parameter('out_angle_max', math.pi)
        self.declare_parameter('interpolation', 'nearest')
        self.declare_parameter('jump_threshold', 0.3)
        self.declare_parameter('max_gap_factor', 2.5)

        g = lambda k: self.get_parameter(k).value
        self.n_out = int(g('n_beams'))
        self.out_min = float(g('out_angle_min'))
        out_max = float(g('out_angle_max'))
        self.interp = str(g('interpolation')).lower()
        self.jump = float(g('jump_threshold'))
        self.gap_factor = float(g('max_gap_factor'))

        if self.n_out < 2:
            raise ValueError('n_beams must be >= 2')
        if self.interp not in ('nearest', 'linear'):
            raise ValueError("interpolation must be 'nearest' or 'linear'")

        span = out_max - self.out_min
        full_circle = span >= TWO_PI - 1e-3
        if full_circle:
            span = TWO_PI
            self.out_inc = span / self.n_out
        else:
            self.out_inc = span / (self.n_out - 1)
        self.out_max = self.out_min + (self.n_out - 1) * self.out_inc

        # Target angles relative to out_min, in [0, 2*pi)
        self.tgt_rel = np.arange(self.n_out, dtype=np.float64) * self.out_inc

        self.pub = self.create_publisher(
            LaserScan, g('output_topic'),
            QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                       history=HistoryPolicy.KEEP_LAST, depth=10))
        self.sub = self.create_subscription(
            LaserScan, g('input_topic'), self._cb, qos_profile_sensor_data)

        self.get_logger().info(
            f'Resampling {g("input_topic")} -> {g("output_topic")}: {self.n_out} beams, '
            f'[{self.out_min:.4f}, {self.out_max:.4f}] rad, inc={self.out_inc:.6f} rad, '
            f'mode={self.interp}')

    # ------------------------------------------------------------------
    def _cb(self, msg: LaserScan):
        n = len(msg.ranges)
        if n < 2 or msg.angle_increment == 0.0:
            return

        r = np.asarray(msg.ranges, dtype=np.float32)
        have_i = len(msg.intensities) == n
        it = np.asarray(msg.intensities, dtype=np.float32) if have_i else None

        # Input angles wrapped into [0, 2*pi) relative to the output origin, sorted
        ang = msg.angle_min + np.arange(n, dtype=np.float64) * msg.angle_increment
        rel = np.mod(ang - self.out_min, TWO_PI)
        order = np.argsort(rel, kind='stable')
        rel = rel[order]
        r = r[order]
        if have_i:
            it = it[order]

        # Circular padding so interpolation wraps across the +-pi seam
        ext_rel = np.concatenate(([rel[-1] - TWO_PI], rel, [rel[0] + TWO_PI]))
        ext_r = np.concatenate(([r[-1]], r, [r[0]]))

        tgt = self.tgt_rel
        hi = np.clip(np.searchsorted(ext_rel, tgt, side='left'), 1, len(ext_rel) - 1)
        lo = hi - 1
        dl = tgt - ext_rel[lo]
        dh = ext_rel[hi] - tgt

        nearest = np.where(dl <= dh, lo, hi)
        out = ext_r[nearest].copy()

        max_gap = self.gap_factor * abs(msg.angle_increment)
        out[np.minimum(dl, dh) > max_gap] = np.nan

        if self.interp == 'linear':
            rl, rh = ext_r[lo], ext_r[hi]
            span = dl + dh
            ok = (np.isfinite(rl) & np.isfinite(rh)
                  & (np.abs(rh - rl) < self.jump)
                  & (span <= max_gap) & (span > 0.0))
            w = np.zeros_like(dl)
            w[ok] = dl[ok] / span[ok]
            out = np.where(ok, rl + w * (rh - rl), out)

        out_msg = LaserScan()
        out_msg.header = msg.header
        out_msg.angle_min = float(self.out_min)
        out_msg.angle_max = float(self.out_max)
        out_msg.angle_increment = float(self.out_inc)
        out_msg.scan_time = msg.scan_time
        if msg.scan_time > 0.0:
            out_msg.time_increment = msg.scan_time / self.n_out
        else:
            out_msg.time_increment = msg.time_increment * n / self.n_out
        out_msg.range_min = msg.range_min
        out_msg.range_max = msg.range_max
        out_msg.ranges = out.astype(np.float32).tolist()

        if have_i:
            ext_it = np.concatenate(([it[-1]], it, [it[0]]))
            ii = ext_it[nearest].copy()
            ii[np.minimum(dl, dh) > max_gap] = 0.0
            out_msg.intensities = ii.astype(np.float32).tolist()

        self.pub.publish(out_msg)


def main():
    rclpy.init()
    node = ScanResampler()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()