import numpy as np

def fit_plane(points):
    h, w, _ = points.shape

    bottom = int(h * 0.6)
    pts = points[bottom:, :, :].reshape(-1, 3)

    valid = np.isfinite(pts).all(axis=1)
    pts = pts[valid]

    pts = pts[pts[:, 2] > 0.2]

    if pts.shape[0] > 20000:
        indices = np.random.choice(pts.shape[0], 20000, replace=False)
        pts = pts[indices]

    centroid = np.mean(pts, axis=0)
    pts_centered = pts - centroid

    U, S, Vt = np.linalg.svd(pts_centered)

    normals = Vt
    best_normal = None

    for normal in normals[::-1]:
        a, b, c = normal
        if abs(b) > 0.6:
            best_normal = normal
            break

    if best_normal is None:
        best_normal = Vt[-1]

    normal = best_normal

    a, b, c = normal
    d = -np.dot(normal, centroid)

    norm = np.sqrt(a*a + b*b + c*c)
    a, b, c, d = a/norm, b/norm, c/norm, d/norm

    return a, b, c, d

def plane_distance(points, a, b, c, d):
    h, w, _ = points.shape
    pts = points.reshape(-1, 3)
    dist = np.abs(a*pts[:,0] + b*pts[:,1] + c*pts[:,2] + d)
    dist = dist.reshape(h, w)
    return dist