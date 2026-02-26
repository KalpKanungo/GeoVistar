import numpy as np

def create_intrinsics(width, height, fov_deg=60):
    f = width / (2 * np.tan(np.deg2rad(fov_deg) / 2))
    cx = width / 2
    cy = height / 2
    return f, f, cx, cy

def backproject(depth, f_x, f_y, c_x, c_y):
    h, w = depth.shape
    i, j = np.meshgrid(np.arange(w), np.arange(h))

    Z = depth.astype(np.float32).copy()

    Z[~np.isfinite(Z)] = 0
    Z[Z < 0.1] = 0
    Z[Z > 8.0] = 0

    X = (i - c_x) * Z / f_x
    Y = (j - c_y) * Z / f_y

    X[~np.isfinite(X)] = 0
    Y[~np.isfinite(Y)] = 0

    X = np.clip(X, -10, 10)
    Y = np.clip(Y, -10, 10)

    points = np.stack((X, Y, Z), axis=-1)

    return points