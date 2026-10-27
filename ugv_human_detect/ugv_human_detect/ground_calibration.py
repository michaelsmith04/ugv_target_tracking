"""
Fit tilt_offset and ankle_height for the ground-plane estimator from stationary ground-truth samples.

For each sample stand still, facing the rover, and let the camera settle. Read from the node's log line:
  tilt_raw  (deg)  -> 'tilt_raw=..'
  v         (px)   -> 'gnd: ground ... v=..'   (ankle row)
and measure D = horizontal distance from the PAN AXIS to your ankles (any bearing, tape measure).
Use at least 3 different distances (e.g. 1, 1.5, 2, 3 m); one distance cannot separate the two parameters.
"""
import numpy as np

FY, CY = 234.53159, 231.54267       # rectified P matrix
CAM_BASE_H, CAM_ARM = 0.22, 0.065   # node parameters cam_base_height, cam_arm

# (D [m], ankle row v [px], tilt_raw [deg])  <-- replace with your measurements
SAMPLES = [
    (1.0, 387.0, 30.5),
    (1.5, 342.0, 25.7),
    (2.0, 322.0, 21.0),
    (2.5, 306.0, 19.0),
]


def residuals(c, a):
    """Predicted minus measured depression (deg) for tilt_offset c [rad] and ankle height a [m]."""
    out = []
    for D, v, tilt in SAMPLES:
        rho = np.radians(tilt) + c
        hc = CAM_BASE_H + CAM_ARM * np.sin(rho)
        d_cam = D - CAM_ARM * np.cos(rho)                 # pan axis -> camera, horizontal
        pred = np.arctan2(hc - a, d_cam)
        meas = np.arctan((v - CY) / FY) - rho
        out.append(np.degrees(pred - meas))
    return np.array(out)


def fit():
    best = None
    c0, a0, dc, da = -0.1, 0.07, 0.15, 0.08
    for _ in range(4):   # coarse-to-fine grid search, numpy only
        for c in np.linspace(c0 - dc, c0 + dc, 41):
            for a in np.linspace(max(a0 - da, 0.0), a0 + da, 41):
                e = float(np.sum(residuals(c, a) ** 2))
                if best is None or e < best[0]:
                    best = (e, c, a)
        _, c0, a0 = best
        dc, da = dc / 5, da / 5
    return best


if __name__ == '__main__':
    e, c, a = fit()
    r = residuals(c, a)
    print(f'tilt_offset = {c:.4f} rad ({np.degrees(c):.2f} deg)')
    print(f'ankle_height = {a:.3f} m')
    print(f'residual per sample (deg): {np.round(r, 2)}   rms {np.sqrt(np.mean(r ** 2)):.2f} deg')
    print('Run with: --ros-args -p tilt_offset:=%.4f -p ankle_height:=%.3f' % (c, a))