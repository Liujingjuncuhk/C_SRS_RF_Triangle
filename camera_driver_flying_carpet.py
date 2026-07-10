import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
from dataclasses import dataclass


@dataclass
class PointCloud:
    """Raw fused point cloud from all cameras, in the global frame."""
    points: np.ndarray   # (N, 3) float64 — XYZ in metres
    colors: np.ndarray   # (N, 3) float64 — RGB in [0, 1]


class SpeckleVisualizer:
    """Non-blocking Open3D window that displays the raw speckle point cloud.

    Usage
    -----
    with SpeckleVisualizer() as vis:
        while True:
            cloud = cams.get_speckle_cloud()
            vis.update(cloud)
    """

    def __init__(self, title: str = "Speckle Cloud — global frame",
                 point_color: tuple[float, float, float] = (0.2, 0.7, 1.0)):
        self._point_color = np.array(point_color, dtype=np.float64)
        self._vis = o3d.visualization.Visualizer()
        self._vis.create_window(title, width=1024, height=720)
        self._pcd   = o3d.geometry.PointCloud()
        self._axes  = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        self._vis.add_geometry(self._axes)
        self._vis.add_geometry(self._pcd)
        self._first = True

    def update(self, cloud) -> None:
        """Refresh the window with a new SpeckleCloud (or None to just pump events)."""
        if cloud is not None and len(cloud.points):
            colors = np.tile(self._point_color, (len(cloud.points), 1))
            self._pcd.points = o3d.utility.Vector3dVector(cloud.points)
            self._pcd.colors = o3d.utility.Vector3dVector(colors)
            if self._first:
                self._vis.reset_view_point(True)
                self._first = False
            self._vis.update_geometry(self._pcd)

        self._vis.poll_events()
        self._vis.update_renderer()

    def close(self) -> None:
        self._vis.destroy_window()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


CAMERA_SERIALS = [
    "152222071771",
    "152222071914",
    "830112071462",
    "213622072727",
]

# ---------------------------------------------------------------------------
# AprilTag 16h5 detector (shared, constructed once)
# ---------------------------------------------------------------------------
_ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
_ARUCO_PARAMS = cv2.aruco.DetectorParameters()
_DETECTOR = cv2.aruco.ArucoDetector(_ARUCO_DICT, _ARUCO_PARAMS)


def get_board_pos_cam3():
    board_pos = np.zeros((8*11, 3))
    STEP = 0.025
    # board is 8X11
    start_pos = [0.45-0.02-0.025,0.03,0.083+0.0175+0.025]
    for i in range(11):
        for j in range(8):
            board_pos[i * 8 + j] = [start_pos[0] - i * STEP, start_pos[1], start_pos[2]+j * STEP]
    return board_pos

def get_board_pos_cam2():
    board_pos = np.zeros((8*11, 3))
    STEP = 0.025
    # board is 8X11
    start_pos = [0.11+0.02+0.025,0.73,0.083+0.0175+0.025]
    for i in range(11):
        for j in range(8):
            board_pos[i * 8 + j] = [start_pos[0] + i * STEP, start_pos[1], start_pos[2]+j * STEP]
    return board_pos


class TagDetection:
    """Result for a single detected AprilTag 16h5."""

    __slots__ = ("tag_id", "corners", "center")

    def __init__(self, tag_id: int, corners: np.ndarray):
        """
        tag_id  : int
        corners : float32 (4, 2)  in pixel coordinates, top-left going clockwise
        center  : float32 (2,)    pixel centre of the tag
        """
        self.tag_id = tag_id
        self.corners = corners          # shape (4, 2)
        self.center = corners.mean(axis=0)  # shape (2,)

    def __repr__(self) -> str:
        cx, cy = self.center
        return f"TagDetection(id={self.tag_id}, center=({cx:.1f}, {cy:.1f}))"


def detect_apriltags(color_bgr: np.ndarray) -> list[TagDetection]:
    """Detect AprilTag 16h5 markers in *color_bgr* and return a list of
    :class:`TagDetection` objects (one per detected tag).

    Parameters
    ----------
    color_bgr : uint8 (H, W, 3)
    """
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    corners_list, ids, _ = _DETECTOR.detectMarkers(gray)
    if ids is None:
        return []
    return [
        TagDetection(int(ids[i][0]), corners_list[i].reshape(4, 2))
        for i in range(len(ids))
    ]


def draw_apriltags(
    color_bgr: np.ndarray,
    detections: list[TagDetection],
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw tag outlines, corner markers, and ID labels onto *color_bgr*.

    Visual elements per tag:
    - Green polygon border around the tag
    - Red dot on corner 0 (top-left) to show orientation
    - Filled black background behind the ID label for readability
    - ID label centred on the tag

    Returns a new annotated image (original is not modified).
    """
    out = color_bgr.copy()
    for det in detections:
        corners = det.corners.astype(np.int32)

        # --- border ---
        pts = corners.reshape((-1, 1, 2))
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=thickness)

        # --- corner dots: red on corner 0, white on the rest ---
        for k, (px, py) in enumerate(corners):
            dot_color = (0, 0, 255) if k == 0 else (255, 255, 255)
            cv2.circle(out, (px, py), 5, dot_color, -1)

        # --- ID label with filled background ---
        label = f"id:{det.tag_id}"
        font, font_scale, font_thick = cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thick)
        cx, cy = int(det.center[0]), int(det.center[1])
        tx, ty = cx - tw // 2, cy + th // 2
        # filled background rectangle
        cv2.rectangle(out, (tx - 3, ty - th - 3), (tx + tw + 3, ty + baseline + 3),
                      (0, 0, 0), cv2.FILLED)
        cv2.putText(out, label, (tx, ty), font, font_scale, color, font_thick, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# Tag pose estimation helpers
# ---------------------------------------------------------------------------

# 3-D corners of a tag in its local frame (Z=0 plane, origin at tag centre).
# OpenCV aruco corner order: top-left, top-right, bottom-right, bottom-left.
def _tag_object_points(tag_size: float) -> np.ndarray:
    h = tag_size / 2.0
    return np.array([
        [-h,  h, 0.0],
        [ h,  h, 0.0],
        [ h, -h, 0.0],
        [-h, -h, 0.0],
    ], dtype=np.float64)


class TagPose:
    """6-DOF pose of a single detected tag in the camera frame.

    Attributes
    ----------
    tag_id  : int
    R       : float64 (3, 3)  rotation matrix  — tag → camera
    t       : float64 (3,)    translation (metres) — tag origin in camera frame
    T       : float64 (4, 4)  homogeneous transform [R|t; 0 1]
    """

    __slots__ = ("tag_id", "R", "t", "T")

    def __init__(self, tag_id: int, R: np.ndarray, t: np.ndarray):
        self.tag_id = tag_id
        self.R = R
        self.t = t.ravel()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3,  3] = self.t
        self.T = T

    def __repr__(self) -> str:
        tx, ty, tz = self.t
        return f"TagPose(id={self.tag_id}, t=[{tx:.3f}, {ty:.3f}, {tz:.3f}] m)"


def _intrinsics_to_cv(intr: rs.intrinsics) -> tuple[np.ndarray, np.ndarray]:
    """Convert a pyrealsense2 intrinsics object to OpenCV camera matrix and
    distortion coefficients."""
    K = np.array([
        [intr.fx,       0, intr.ppx],
        [      0, intr.fy, intr.ppy],
        [      0,       0,        1],
    ], dtype=np.float64)
    dist = np.array(intr.coeffs, dtype=np.float64)
    return K, dist


def estimate_tag_poses(
    detections: list[TagDetection],
    intrinsics: rs.intrinsics,
    tag_size: float,
) -> list[TagPose]:
    """Estimate the 6-DOF pose of each detected tag using PnP.

    Parameters
    ----------
    detections : list[TagDetection]
        Output of :func:`detect_apriltags`.
    intrinsics : rs.intrinsics
        Colour-stream intrinsics from :meth:`D435Camera.get_intrinsics`.
    tag_size : float
        Physical side length of the printed tag in **metres**.

    Returns
    -------
    list[TagPose]  — one entry per detection, same order as *detections*.
    """
    K, dist = _intrinsics_to_cv(intrinsics)
    obj_pts = _tag_object_points(tag_size)
    poses = []
    for det in detections:
        img_pts = det.corners.astype(np.float64)
        ok, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        if not ok:
            continue
        R, _ = cv2.Rodrigues(rvec)
        poses.append(TagPose(det.tag_id, R, tvec))
    return poses


def compute_relative_transform(
    poses_a: list[TagPose],
    poses_b: list[TagPose],
) -> tuple[np.ndarray, list[int]]:
    """Estimate the rigid transform from camera B to camera A using tags that
    are visible in both frames simultaneously.

    For each shared tag, T_A_B = T_a_tag @ inv(T_b_tag).  The results from all
    shared tags are averaged (rotation via Rodrigues mean, translation mean).

    Parameters
    ----------
    poses_a, poses_b : list[TagPose]
        Tag poses in camera A and camera B respectively (same time step).

    Returns
    -------
    T_A_B : float64 (4, 4)
        Homogeneous transform such that p_A = T_A_B @ p_B.
    shared_ids : list[int]
        Tag IDs that contributed to the estimate.

    Raises
    ------
    ValueError
        If no tag is visible in both cameras simultaneously.
    """
    map_b = {p.tag_id: p for p in poses_b}
    rvecs, tvecs, shared_ids = [], [], []

    for pa in poses_a:
        pb = map_b.get(pa.tag_id)
        if pb is None:
            continue
        # T_A_B = T_A_tag @ T_tag_B = T_A_tag @ inv(T_B_tag)
        T_AB = pa.T @ np.linalg.inv(pb.T)
        rvec, _ = cv2.Rodrigues(T_AB[:3, :3])
        rvecs.append(rvec.ravel())
        tvecs.append(T_AB[:3, 3])
        shared_ids.append(pa.tag_id)

    if not shared_ids:
        raise ValueError("No shared tag visible in both cameras.")

    mean_rvec = np.mean(rvecs, axis=0)
    mean_t    = np.mean(tvecs, axis=0)
    R_mean, _ = cv2.Rodrigues(mean_rvec)
    T_AB = np.eye(4)
    T_AB[:3, :3] = R_mean
    T_AB[:3,  3] = mean_t
    return T_AB, shared_ids


# ---------------------------------------------------------------------------
# Point cloud helpers
# ---------------------------------------------------------------------------

def _make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = t.ravel()
    return T


def _invert_T(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3,  3] = -R.T @ t
    return Ti


class D435Camera:
    """Wraps a single Intel RealSense D435 camera stream."""

    def __init__(
        self,
        serial: str,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        enable_depth: bool = True,
        enable_color: bool = True,
        align_to_color: bool = True,
    ):
        self.serial = serial
        self.width = width
        self.height = height
        self.fps = fps
        self.align_to_color = align_to_color

        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)

        if enable_color:
            config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        if enable_depth:
            config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

        self._profile = self._pipeline.start(config)

        # Depth scale: converts raw uint16 to metres
        depth_sensor = self._profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        self._align = rs.align(rs.stream.color) if align_to_color else None

        # Device-calibrated extrinsics: depth sensor → color sensor (fixed hardware offset)
        _d_stream = self._profile.get_stream(rs.stream.depth)
        _c_stream = self._profile.get_stream(rs.stream.color)
        _extr = _d_stream.get_extrinsics_to(_c_stream)
        self._T_depth_to_color = _make_T(
            np.array(_extr.rotation).reshape(3, 3),
            np.array(_extr.translation),   # metres
        )

    # ------------------------------------------------------------------
    def read(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return (color_bgr, depth_m) as numpy arrays.

        color_bgr : uint8  (H, W, 3)  BGR, or None if not enabled
        depth_m   : float32 (H, W)    depth in metres, or None if not enabled

        Returns (None, None) if no frame arrives within the timeout (e.g. due
        to USB bandwidth contention when multiple cameras share a controller).
        """
        try:
            frames = self._pipeline.wait_for_frames(timeout_ms=5000)
        except RuntimeError:
            return None, None

        if self._align is not None:
            frames = self._align.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        color_image = (
            np.asanyarray(color_frame.get_data()) if color_frame else None
        )
        depth_image = (
            np.asanyarray(depth_frame.get_data()).astype(np.float32) * self.depth_scale
            if depth_frame
            else None
        )
        return color_image, depth_image

    def estimate_tag_poses(
        self, detections: list[TagDetection], tag_size: float
    ) -> list[TagPose]:
        """Estimate 6-DOF pose for each detected tag using this camera's intrinsics.

        Parameters
        ----------
        detections : list[TagDetection]
            Output of :meth:`detect_tags` or :func:`detect_apriltags`.
        tag_size : float
            Physical side length of the printed tag in **metres**
            (e.g. ``0.15`` for a 15 cm tag).

        Returns
        -------
        list[TagPose]
        """
        return estimate_tag_poses(detections, self.get_intrinsics(), tag_size)

    def detect_tags(self, color_bgr: np.ndarray) -> list[TagDetection]:
        """Detect AprilTag 16h5 markers in a colour image.

        Convenience wrapper around the module-level :func:`detect_apriltags`.

        Parameters
        ----------
        color_bgr : uint8 (H, W, 3)  — e.g. the first element returned by :meth:`read`
        """
        return detect_apriltags(color_bgr)

    def read_with_tags(
        self, annotate: bool = True
    ) -> tuple[np.ndarray | None, np.ndarray | None, list[TagDetection]]:
        """Capture a frame and run AprilTag 16h5 detection on the colour image.

        Parameters
        ----------
        annotate : bool
            When *True* the returned colour image has tag outlines and IDs drawn
            on it.  Set to *False* to get the raw frame alongside detections.

        Returns
        -------
        color_bgr   : uint8 (H, W, 3) or None
        depth_m     : float32 (H, W)  or None
        detections  : list[TagDetection]
        """
        color, depth = self.read()
        if color is None:
            return color, depth, []
        detections = detect_apriltags(color)
        if annotate:
            color = draw_apriltags(color, detections)
        return color, depth, detections

    def read_raw(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return (color_bgr, depth_raw_m) with depth in the **depth camera frame**
        (no alignment applied).  Use this for point cloud generation."""
        try:
            frames = self._pipeline.wait_for_frames(timeout_ms=5000)
        except RuntimeError:
            return None, None
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        color_image = np.asanyarray(color_frame.get_data()) if color_frame else None
        depth_image = (
            np.asanyarray(depth_frame.get_data()).astype(np.float32) * self.depth_scale
            if depth_frame else None
        )
        return color_image, depth_image

    def get_raw_frames(self) -> tuple[rs.frame | None, rs.depth_frame | None]:
        """Return (color_frame, depth_frame) as raw RealSense frame objects (no alignment).
        Use this when a rs.depth_frame is required, e.g. for rs.pointcloud.calculate()."""
        try:
            frames = self._pipeline.wait_for_frames(timeout_ms=5000)
        except RuntimeError:
            return None, None
        return frames.get_color_frame() or None, frames.get_depth_frame() or None

    def get_intrinsics(self) -> rs.intrinsics:
        """Return colour-stream camera intrinsics."""
        stream = self._profile.get_stream(rs.stream.color)
        return stream.as_video_stream_profile().get_intrinsics()

    def get_depth_intrinsics(self) -> rs.intrinsics:
        """Return depth-stream intrinsics (native depth camera, not aligned)."""
        stream = self._profile.get_stream(rs.stream.depth)
        return stream.as_video_stream_profile().get_intrinsics()

    def get_depth_to_color_T(self) -> np.ndarray:
        """Return the 4×4 device-calibrated transform from depth frame to color frame."""
        return self._T_depth_to_color

    def is_ir_emitter_enabled(self) -> bool:
        """Return whether the depth camera's infrared projector is enabled."""
        depth_sensor = self._profile.get_device().first_depth_sensor()
        if not depth_sensor.supports(rs.option.emitter_enabled):
            raise RuntimeError(
                f"Camera {self.serial} does not expose the IR emitter option."
            )
        return bool(depth_sensor.get_option(rs.option.emitter_enabled))

    def set_ir_emitter(self, enabled: bool) -> None:
        """Enable or disable the IR projector used to add a depth texture.

        With the emitter disabled, the D435 still computes passive stereo depth
        using its two infrared cameras.
        """
        depth_sensor = self._profile.get_device().first_depth_sensor()
        if not depth_sensor.supports(rs.option.emitter_enabled):
            raise RuntimeError(
                f"Camera {self.serial} does not expose the IR emitter option."
            )
        depth_sensor.set_option(rs.option.emitter_enabled, 1.0 if enabled else 0.0)

    def disable_ir_emitter(self) -> None:
        """Disable the infrared projector and use passive stereo depth."""
        self.set_ir_emitter(False)

    def enable_ir_emitter(self) -> None:
        """Enable the infrared projector for active stereo depth."""
        self.set_ir_emitter(True)

    def stop(self):
        self._pipeline.stop()

    # context-manager support
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.stop()


class D435CameraArray:
    """Manages an array of Intel RealSense D435 cameras.

    Parameters
    ----------
    serials : list[str] | None
        Serial numbers of cameras to open.  Defaults to all four cameras
        defined in CAMERA_SERIALS.
    """

    def __init__(
        self,
        serials: list[str] | None = None,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        align_to_color: bool = True,
    ):
        serials = serials if serials is not None else CAMERA_SERIALS
        if len(serials) >= 4 and fps > 15:
            print(f"[WARNING] {len(serials)} cameras at {fps} fps may saturate USB bandwidth. "
                  f"Consider fps=15 if you see frame timeout errors.")
        self.cameras: list[D435Camera] = [
            D435Camera(
                serial=s,
                width=width,
                height=height,
                fps=fps,
                align_to_color=align_to_color,
            )
            for s in serials
        ]
        self.R_global2rgb_cam0 = np.array([
            [-0.00208392, -0.89714812, -0.44172493],
            [-0.99969033,  0.01282260, -0.02132661],
            [ 0.02479719,  0.44154370, -0.89689702],
        ])
        self.T_global2rgb_cam0 = np.array([[0.52368597, 0.29831344, 0.79789714]])
        self.R_depth2global_cam0 = np.array([
            [ 0.00588417, -0.99973285,  0.02235244],
            [-0.90267338,  0.00430804,  0.43030484],
            [-0.43028617, -0.02270894, -0.90240687],
        ])
        self.T_depth2global_cam0 = np.array([[0.27944368, 0.10042357, 0.94627525]])

        self.R_global2rgb_cam1 = np.array([
            [ 0.02115157,  0.91157609, -0.41058696],
            [ 0.99962341, -0.02646414, -0.00725906],
            [-0.01748302, -0.41027879, -0.91179255],
        ])
        self.T_global2rgb_cam1 = np.array([[-0.19021806, -0.25731438, 1.14084507]])
        self.R_depth2global_cam1 = np.array([
            [-0.00660622,  0.99991596, -0.01115738],
            [ 0.91425253,  0.00151955, -0.40514193],
            [-0.40509091, -0.01287712, -0.91418572],
        ])
        self.T_depth2global_cam1 = np.array([[0.28119081, 0.64783685, 0.95389252]])

        self.finetuneR_0 = np.array([
            [ 9.99997470e-01, -2.08579283e-03,  8.42870858e-04],
            [ 2.08914381e-03,  9.99989839e-01, -3.99454713e-03],
            [-8.34530497e-04,  3.99629790e-03,  9.99991667e-01],
        ])
        self.finetunet_0 = np.array([0.00147063, 0.00091940, 0.00359344])
        self.finetuneR_1 = np.array([
            [ 0.99999736, -0.00132472, -0.00187654],
            [ 0.00133641,  0.99997964,  0.00623904],
            [ 0.00186824, -0.00624153,  0.99997878],
        ])
        self.finetune_t_1 = np.array([0.00096585, -0.00250919, -0.00212032])

        self.R_global2rgb_cam2 = np.array([
            [ 0.99997177,  0.00714798,  0.00231599],
            [ 0.00035797,  0.26256042, -0.96491549],
            [-0.00750529,  0.96488908,  0.26255045],
        ])
        self.T_global2rgb_cam2 = np.array([[-0.24607567, 0.12146247, 0.22456740]])
        self.T_depth2global_cam2 = np.array([
            [ 0.99991515,  0.00482132, -0.01209886,  0.26270179],
            [ 0.01041825,  0.26138127,  0.96517939, -0.24595902],
            [ 0.00781585, -0.96522363,  0.26130886,  0.05868327],
            [ 0.00000000,  0.00000000,  0.00000000,  1.00000000],
        ])

        self.R_global2rgb_cam3 = np.array([
            [-0.99998752, -0.00498912, -0.00024900],
            [ 0.00151509, -0.25542276, -0.96682828],
            [ 0.00476003, -0.96681659,  0.25542714],
        ])
        self.T_global2rgb_cam3 = np.array([[0.31618833, 0.32287188, 0.95419180]])
        self.T_depth2global_cam3 = np.array([
            [-0.99996291,  0.00049941,  0.00859962,  0.29595497],
            [-0.00844364, -0.25444867, -0.96704944,  1.00637115],
            [ 0.00170521, -0.96708616,  0.25444344,  0.06851284],
            [ 0.00000000,  0.00000000,  0.00000000,  1.00000000],
        ])
        # Dynamically calibrated extrinsics: maps 0-based cam index → 4×4 T_cam_to_global
        self._calibrated_extrinsics: dict[int, np.ndarray] = {}
        # Depth-camera-to-global: computed via R_depth2global (cam0/1) or derived (cam2/3)
        self._calibrated_depth_extrinsics: dict[int, np.ndarray] = {}
        self.Transform_depth2global = [self.get_depth_to_global(0), self.get_depth_to_global(1), self.T_depth2global_cam2, self.T_depth2global_cam3]
        self.R_global2rgb_list = [self.R_global2rgb_cam0, self.R_global2rgb_cam1, self.R_global2rgb_cam2, self.R_global2rgb_cam3]
        self.T_global2rgb_list = [self.T_global2rgb_cam0, self.T_global2rgb_cam1, self.T_global2rgb_cam2, self.T_global2rgb_cam3]
        self._speckle: dict | None = None   # lazily initialised by get_speckle_cloud()

    # ------------------------------------------------------------------
    def read_all(
        self,
    ) -> list[tuple[np.ndarray | None, np.ndarray | None]]:
        """Read one frame from every camera.

        Returns a list of (color_bgr, depth_m) tuples in the same order as
        ``self.cameras``.
        """
        return [cam.read() for cam in self.cameras]

    def read_all_with_tags(
        self, annotate: bool = True
    ) -> list[tuple[np.ndarray | None, np.ndarray | None, list[TagDetection]]]:
        """Read one frame from every camera and run AprilTag 16h5 detection.

        Returns a list of ``(color_bgr, depth_m, detections)`` tuples in the
        same order as ``self.cameras``.
        """
        return [cam.read_with_tags(annotate=annotate) for cam in self.cameras]

    def disable_ir_emitters(self) -> None:
        """Disable the infrared projector on every camera in the array."""
        for idx, cam in enumerate(self.cameras):
            cam.disable_ir_emitter()
            print(f"  cam{idx} ({cam.serial}): IR emitter disabled")

    def enable_ir_emitters(self) -> None:
        """Enable the infrared projector on every camera in the array."""
        for idx, cam in enumerate(self.cameras):
            cam.enable_ir_emitter()
            print(f"  cam{idx} ({cam.serial}): IR emitter enabled")

    # ------------------------------------------------------------------
    # Extrinsic helpers
    # ------------------------------------------------------------------

    def get_cam_to_global(self, cam_idx: int) -> np.ndarray:
        """Return the 4×4 transform that maps points from camera *cam_idx*'s
        colour frame into the global frame.

        Checks dynamically calibrated extrinsics first, then falls back to
        the hard-coded values.  Cameras 0 and 1 have fine-tune corrections;
        camera 3 was calibrated from a chessboard (no fine-tune yet).
        """
        if cam_idx in self._calibrated_extrinsics:
            return self._calibrated_extrinsics[cam_idx]
        if cam_idx == 0:
            T_global2cam = _make_T(self.R_global2rgb_cam0, self.T_global2rgb_cam0)
            T_finetune   = _make_T(self.finetuneR_0, self.finetunet_0)
        elif cam_idx == 1:
            T_global2cam = _make_T(self.R_global2rgb_cam1, self.T_global2rgb_cam1)
            T_finetune   = _make_T(self.finetuneR_1, self.finetune_t_1)
        elif cam_idx == 2:
            T_global2cam = _make_T(self.R_global2rgb_cam2, self.T_global2rgb_cam2)
            return _invert_T(T_global2cam)   # no fine-tune for cam2 yet
        elif cam_idx == 3:
            T_global2cam = _make_T(self.R_global2rgb_cam3, self.T_global2rgb_cam3)
            return _invert_T(T_global2cam)   # no fine-tune for cam3 yet
        else:
            raise ValueError(f"No extrinsics for camera index {cam_idx}. Run calibrate_camera_extrinsics() first.")
        return T_finetune @ _invert_T(T_global2cam)

    def get_depth_to_global(self, cam_idx: int) -> np.ndarray:
        """Return the 4×4 transform from the **depth camera frame** of camera
        *cam_idx* into the global frame.

        - cam0 / cam1 : uses the stored ``R_depth2global`` + fine-tune (directly
          calibrated in depth camera frame).
        - cam2 / cam3 : derives depth-to-global from color-to-global composed
          with the device's hardware depth→color extrinsic.
        - Any dynamically calibrated camera: same derivation as cam2/3.
        """
        if cam_idx in self._calibrated_depth_extrinsics:
            return self._calibrated_depth_extrinsics[cam_idx]
        if cam_idx == 0:
            T = _make_T(self.R_depth2global_cam0, self.T_depth2global_cam0)
            return _make_T(self.finetuneR_0, self.finetunet_0) @ T
        if cam_idx == 1:
            T = _make_T(self.R_depth2global_cam1, self.T_depth2global_cam1)
            return _make_T(self.finetuneR_1, self.finetune_t_1) @ T
        # All other cameras: color-to-global ∘ depth-to-color (device)
        T_color2global  = self.get_cam_to_global(cam_idx)
        T_depth2color   = self.cameras[cam_idx].get_depth_to_color_T()
        # T_depth2color = _invert_T(T_depth2color)
        return T_color2global @ T_depth2color

    def calibrate_camera_extrinsics(
        self,
        ref_cam_idx: int,
        target_cam_idx: int,
        board_size: tuple[int, int] = (8, 11),
        square_size: float = 0.025,
        n_frames: int = 30,
    ) -> np.ndarray:
        """Calibrate the extrinsics of *target_cam_idx* relative to the global
        frame using a chessboard visible in both cameras simultaneously.

        *ref_cam_idx* must already have known extrinsics (via the hard-coded
        values or a previous call to this method).

        Parameters
        ----------
        ref_cam_idx    : 0-based index of the reference camera (known extrinsics)
        target_cam_idx : 0-based index of the camera to calibrate
        board_size     : (cols, rows) of inner corners — (7,10) for an 8×11 board
        square_size    : physical side of one square in metres
        n_frames       : number of valid stereo frame pairs to collect

        Controls
        --------
        's'   — capture the current frame pair (only when both cameras see the board)
        'q'   — stop collecting early and compute with frames gathered so far
        ESC   — abort without saving

        Returns
        -------
        T_target_to_global : float64 (4, 4)
            Also stored in ``self._calibrated_extrinsics[target_cam_idx]``.
        """
        cam_ref = self.cameras[ref_cam_idx]
        cam_tgt = self.cameras[target_cam_idx]
        K_ref,  dist_ref  = _intrinsics_to_cv(cam_ref.get_intrinsics())
        K_tgt,  dist_tgt  = _intrinsics_to_cv(cam_tgt.get_intrinsics())
        img_size = (cam_ref.width, cam_ref.height)

        cols, rows = board_size
        obj_single = np.zeros((cols * rows, 3), np.float32)
        obj_single[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        obj_single *= square_size

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        obj_pts, img_pts_ref, img_pts_tgt = [], [], []

        print(f"Calibrating cam{target_cam_idx} using cam{ref_cam_idx} as reference.")
        print(f"Board: {cols}×{rows} inner corners, {square_size*100:.1f} cm squares.")
        print(f"Target: {n_frames} frame pairs.  's'=capture  'q'=finish  ESC=abort\n")

        while len(obj_pts) < n_frames:
            color_ref, _ = cam_ref.read()
            color_tgt, _ = cam_tgt.read()
            if color_ref is None or color_tgt is None:
                continue

            gray_ref = cv2.cvtColor(color_ref, cv2.COLOR_BGR2GRAY)
            gray_tgt = cv2.cvtColor(color_tgt, cv2.COLOR_BGR2GRAY)
            found_ref, c_ref = cv2.findChessboardCorners(gray_ref, board_size)
            found_tgt, c_tgt = cv2.findChessboardCorners(gray_tgt, board_size)

            vis_ref = color_ref.copy()
            vis_tgt = color_tgt.copy()
            cv2.drawChessboardCorners(vis_ref, board_size, c_ref, found_ref)
            cv2.drawChessboardCorners(vis_tgt, board_size, c_tgt, found_tgt)

            both_found = found_ref and found_tgt
            status_col = (0, 255, 0) if both_found else (0, 100, 255)
            status = (f"Frames: {len(obj_pts)}/{n_frames}  "
                      f"ref:{'OK' if found_ref else '--'}  "
                      f"tgt:{'OK' if found_tgt else '--'}"
                      + ("  Press 's' to capture" if both_found else ""))
            vis = np.hstack([vis_ref, vis_tgt])
            cv2.putText(vis, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, status_col, 2, cv2.LINE_AA)
            cv2.imshow(f"Calibration  cam{ref_cam_idx} | cam{target_cam_idx}", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                cv2.destroyAllWindows()
                raise RuntimeError("Calibration aborted.")
            elif key == ord('q'):
                break
            elif key == ord('s') and both_found:
                c_ref2 = cv2.cornerSubPix(gray_ref, c_ref, (11, 11), (-1, -1), criteria)
                c_tgt2 = cv2.cornerSubPix(gray_tgt, c_tgt, (11, 11), (-1, -1), criteria)
                obj_pts.append(obj_single)
                img_pts_ref.append(c_ref2)
                img_pts_tgt.append(c_tgt2)
                print(f"  Captured frame {len(obj_pts)}/{n_frames}")

        cv2.destroyAllWindows()

        if len(obj_pts) < 5:
            raise RuntimeError(f"Need at least 5 frames, only got {len(obj_pts)}.")

        print(f"\nRunning stereoCalibrate on {len(obj_pts)} frames ...")
        rms, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
            obj_pts, img_pts_ref, img_pts_tgt,
            K_ref, dist_ref, K_tgt, dist_tgt,
            img_size,
            flags=cv2.CALIB_FIX_INTRINSIC,
        )
        print(f"  RMS reprojection error: {rms:.4f} px")

        # stereoCalibrate: p_tgt_color = R @ p_ref_color + T
        # ref_color and tgt_color are both COLOR camera frames
        T_ref_color_to_tgt_color = _make_T(R, T)
        T_tgt_color_to_ref_color = _invert_T(T_ref_color_to_tgt_color)

        # ref color-to-global: derived from ref's depth-to-global + device depth→color
        T_ref_depth2global = self.get_depth_to_global(ref_cam_idx)
        T_ref_depth2color  = cam_ref.get_depth_to_color_T()
        T_ref_color2global = T_ref_depth2global @ _invert_T(T_ref_depth2color)

        # target color-to-global
        T_tgt_color2global = T_ref_color2global @ T_tgt_color_to_ref_color

        # target depth-to-global (what point cloud generation needs)
        T_tgt_depth2color  = cam_tgt.get_depth_to_color_T()
        T_tgt_depth2global = T_tgt_color2global @ T_tgt_depth2color

        self._calibrated_extrinsics[target_cam_idx]       = T_tgt_color2global
        self._calibrated_depth_extrinsics[target_cam_idx] = T_tgt_depth2global
        self.Transform_depth2global[target_cam_idx]        = T_tgt_depth2global
        T_global2tgt = np.linalg.inv(T_tgt_color2global)
        self.R_global2rgb_list[target_cam_idx] = T_global2tgt[:3, :3]
        self.T_global2rgb_list[target_cam_idx] = T_global2tgt[:3, [3]].T

        def _fmt_3x3(M: np.ndarray) -> str:
            rows = "\n".join(
                "    [" + ", ".join(f"{v: .8f}" for v in row) + "],"
                for row in M
            )
            return f"np.array([\n{rows}\n])"

        def _fmt_4x4(M: np.ndarray) -> str:
            rows = "\n".join(
                "    [" + ", ".join(f"{v: .8f}" for v in row) + "],"
                for row in M
            )
            return f"np.array([\n{rows}\n])"

        n = target_cam_idx
        T_global2rgb = np.linalg.inv(T_tgt_color2global)
        t_rgb = T_global2rgb[:3, 3]
        t_rgb_str = ", ".join(f"{v:.8f}" for v in t_rgb)

        print("\n" + "─" * 66)
        print("  Paste into D435CameraArray.__init__:")
        print("─" * 66)
        print(f"self.R_global2rgb_cam{n} = {_fmt_3x3(T_global2rgb[:3, :3])}")
        print(f"self.T_global2rgb_cam{n} = np.array([[{t_rgb_str}]])")
        print(f"self.T_depth2global_cam{n} = {_fmt_4x4(T_tgt_depth2global)}")
        print("─" * 66)

        return T_tgt_depth2global

    def get_reading_global(self, idx: int):
        pc = rs.pointcloud()
        _, depth_frame = self.cameras[idx].get_raw_frames()
        if depth_frame is None:
            return np.empty((0, 3)), np.empty((0, 3))
        points = pc.calculate(depth_frame)
        point_coord = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
        T_pts = self.Transform_depth2global[idx]
        pts_global = (T_pts @ np.hstack([point_coord, np.ones((len(point_coord), 1))]).T).T[:, :3]
        R_col = self.R_global2rgb_list[idx]
        T_col = self.T_global2rgb_list[idx]
        trans_col = _make_T(R_col, T_col)
        col_global = (trans_col @ np.hstack([pts_global, np.ones((len(pts_global), 1))]).T).T[:, :3]
        return col_global, pts_global

    def get_merged_pointcloud(
        self,
        cam_indices: list[int] | None = None,
        voxel_size: float = 0.005,
        max_depth: float = 3.0,
    ) -> o3d.geometry.PointCloud:
        """Capture one frame from each calibrated camera, transform every point
        cloud into the global frame and merge them.

        Parameters
        ----------
        cam_indices : cameras to include (default: [0, 1, 3])
        voxel_size  : down-sample leaf size in metres (0 = no down-sampling)
        max_depth   : discard points beyond this distance (metres)
        """
        if cam_indices is None:
            cam_indices = [0, 1, 3]

        all_pts, all_col = [], []
        for idx in cam_indices:
            cols, pts = self.get_reading_global(idx)
            if len(pts) == 0:
                continue
            all_pts.append(pts)
            all_col.append(cols)

        pcd = o3d.geometry.PointCloud()
        if not all_pts:
            return pcd
        pcd.points = o3d.utility.Vector3dVector(np.vstack(all_pts))
        pcd.colors = o3d.utility.Vector3dVector(np.vstack(all_col))
        if voxel_size > 0:
            pcd = pcd.voxel_down_sample(voxel_size)
        return pcd

    def get_point_cloud(
        self,
        cam_indices: list[int] | None = None,
        voxel_size: float = 0.005,
        x_range: tuple[float, float] | None = None,
        y_range: tuple[float, float] | None = None,
        z_range: tuple[float, float] | None = None,
    ) -> PointCloud:
        """Capture one frame from each camera and return a fused global-frame
        point cloud as plain numpy arrays.

        Parameters
        ----------
        cam_indices : cameras to include (default: all four)
        voxel_size  : voxel down-sample leaf size in metres (0 = no down-sampling)
        x_range, y_range, z_range : optional (min, max) axis-aligned clip in metres

        Returns
        -------
        PointCloud with .points (N,3) and .colors (N,3) in the global frame.
        Both arrays are empty (shape (0,3)) if no valid depth is available.

        Example
        -------
        with D435CameraArray() as cams:
            cloud = cams.get_point_cloud(x_range=(0.1, 0.5), z_range=(0.0, 1.0))
            print(cloud.points.shape)   # (N, 3)
        """
        if cam_indices is None:
            cam_indices = list(range(len(self.cameras)))

        all_pts, all_col = [], []
        for idx in cam_indices:
            col, pts = self.get_reading_global(idx)
            if len(pts) == 0:
                continue
            all_pts.append(pts)
            all_col.append(col)

        if not all_pts:
            return PointCloud(
                points=np.empty((0, 3), dtype=np.float64),
                colors=np.empty((0, 3), dtype=np.float64),
            )

        points = np.vstack(all_pts).astype(np.float64)
        colors = np.vstack(all_col).astype(np.float64)

        # Axis-aligned bounding box filter
        mask = np.ones(len(points), dtype=bool)
        if x_range is not None:
            mask &= (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1])
        if y_range is not None:
            mask &= (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1])
        if z_range is not None:
            mask &= (points[:, 2] >= z_range[0]) & (points[:, 2] <= z_range[1])
        points, colors = points[mask], colors[mask]

        # Optional voxel down-sampling (delegates to Open3D, result back to numpy)
        if voxel_size > 0 and len(points):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
            pcd = pcd.voxel_down_sample(voxel_size)
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors)

        return PointCloud(points=points, colors=colors)

    # ------------------------------------------------------------------
    # Speckle-pattern tracker (stateful, lazy init)
    # ------------------------------------------------------------------

    def _init_speckle_tracker(self, pattern_path: str | None = None) -> bool:
        """Load pattern, build SIFT descriptors, create per-camera trackers.
        Called automatically on the first get_speckle_cloud() call.
        Returns True on success, False if the pattern image cannot be loaded.
        track_speckle is imported lazily here to avoid a circular import
        (track_speckle imports camera_driver at module level).
        """
        import importlib, sys as _sys
        # Allow the scripts sub-directory to be found
        import os as _os
        _scripts = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "scripts")
        if _scripts not in _sys.path:
            _sys.path.insert(0, _scripts)
        _ts = importlib.import_module("track_speckle")

        path = pattern_path or _ts.PATTERN_PATH
        pat_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if pat_gray is None:
            print(f"[SpeckleTracker] Cannot load pattern: {path}")
            return False

        pat_kp, pat_des = _ts.build_pattern_descriptors(pat_gray)
        Ks, dists = [], []
        for cam in self.cameras:
            K, d = _intrinsics_to_cv(cam.get_intrinsics())
            Ks.append(K)
            dists.append(d)

        self._speckle = dict(
            ts        = _ts,
            pat_kp    = pat_kp,
            pat_des   = pat_des,
            trackers  = [_ts.CamTracker(i) for i in range(len(self.cameras))],
            ref_dict  = {},
            curr_dict = {},
            frame_n   = 0,
            Ks        = Ks,
            dists     = dists,
        )
        print(f"[SpeckleTracker] Ready — {_ts.N_GRID_X}×{_ts.N_GRID_Y} grid, "
              f"{len(pat_kp)} SIFT keypoints")
        return True

    def get_speckle_cloud(
        self,
        pattern_path: str | None = None,
        warmup_frames: int = 30,
    ):
        """Run one step of the speckle tracker and return the fused labelled cloud.

        Lazily initialises the tracker on the first call (loads pattern, runs
        SIFT).  Returns None during the warmup window or before any grid points
        have been back-projected successfully.

        Parameters
        ----------
        pattern_path  : path to speckle_pattern.png.  Defaults to the path
                        defined in track_speckle.PATTERN_PATH.
        warmup_frames : frames to flush before first init so that camera
                        auto-exposure and depth stabilise (default 30 ≈ 2 s
                        at 15 fps).

        Returns
        -------
        SpeckleCloud or None

        Example
        -------
        with D435CameraArray(fps=15) as cams:
            while True:
                cloud = cams.get_speckle_cloud()
                if cloud is not None:
                    print(cloud.points.shape, cloud.displacements_m.max() * 1000, "mm")
        """
        from collections import defaultdict

        if self._speckle is None:
            if not self._init_speckle_tracker(pattern_path):
                return None

        sp = self._speckle
        _ts = sp["ts"]
        sp["frame_n"] += 1

        frames = self.read_all()

        if sp["frame_n"] <= warmup_frames:
            return None

        obs: defaultdict = defaultdict(list)

        for i, (color, depth) in enumerate(frames):
            if color is None or depth is None:
                continue
            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            trk  = sp["trackers"][i]
            T2g  = self.get_cam_to_global(i)

            if trk.stale:
                trk.init(gray, sp["pat_kp"], sp["pat_des"], depth_m=depth)
            else:
                trk.step(gray)

            if trk.pts is not None and len(trk.pts):
                pts3d, phys_v = _ts.backproject(
                    trk.pts, trk.phys_xy, depth,
                    sp["Ks"][i], sp["dists"][i], T2g,
                )
                if len(pts3d) and phys_v is not None:
                    for pt3, (x, y) in zip(pts3d, phys_v):
                        obs[(int(round(float(x))),
                             int(round(float(y))))].append(pt3)

        sp["curr_dict"].clear()
        for key, pts_list in obs.items():
            sp["curr_dict"][key] = _ts.fuse_observations(pts_list)

        if not sp["ref_dict"] and sp["curr_dict"]:
            sp["ref_dict"].update(sp["curr_dict"])
            print(f"[SpeckleTracker] Reference set: {len(sp['ref_dict'])} pts")

        return _ts.get_point_cloud(sp["curr_dict"], sp["ref_dict"])

    def reset_speckle_reference(self) -> None:
        """Capture the current speckle positions as the new zero-displacement reference."""
        if self._speckle and self._speckle["curr_dict"]:
            sp = self._speckle
            sp["ref_dict"].clear()
            sp["ref_dict"].update(sp["curr_dict"])
            print(f"[SpeckleTracker] New reference: {len(sp['ref_dict'])} pts")

    def reset_speckle_tracker(self) -> None:
        """Force SIFT re-detection on all cameras (equivalent to pressing 'r')."""
        if self._speckle:
            for trk in self._speckle["trackers"]:
                trk.pts = None
            print("[SpeckleTracker] Forced re-detection")

    def capture_colored_pointclouds(
        self,
        voxel_size: float = 0.005,
        max_depth: float = 1.6,
    ) -> list[o3d.geometry.PointCloud]:
        """Capture one frame from every calibrated camera, transform each point
        cloud into the global frame, and paint each camera a distinct solid colour.

        Camera colours (RGB):
            cam 0 — red    [1, 0, 0]
            cam 1 — green  [0, 1, 0]
            cam 2 — blue   [0, 0, 1]
            cam 3 — yellow [1, 1, 0]

        Returns
        -------
        list[o3d.geometry.PointCloud]
            One entry per camera that successfully produced points, in index order.
        """
        CAM_COLORS = {
            0: [1.0, 0.0, 0.0],   # red
            1: [0.0, 1.0, 0.0],   # green
            2: [0.0, 0.0, 1.0],   # blue
            3: [1.0, 1.0, 0.0],   # yellow
        }

        pcds = []
        for idx in range(len(self.cameras)):
            pts_camera, pts_global = self.get_reading_global(idx)

            if len(pts_global) == 0:
                continue

            # A zero Z value means the depth pixel had no valid measurement.
            valid = np.isfinite(pts_camera).all(axis=1)
            valid &= pts_camera[:, 2] > 0.0
            if max_depth is not None:
                valid &= pts_camera[:, 2] <= max_depth
            pts_global = pts_global[valid]
            if len(pts_global) == 0:
                print(f"  cam{idx}: no valid depth points")
                continue

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts_global)
            pcd.paint_uniform_color(CAM_COLORS.get(idx, [1.0, 1.0, 1.0]))
            if voxel_size > 0:
                pcd = pcd.voxel_down_sample(voxel_size)
            pcds.append(pcd)
            print(f"  cam{idx}: {len(pcd.points):,} pts  colour={CAM_COLORS.get(idx)}")

        return pcds

    @staticmethod
    def filter_pointcloud(
        pcd: o3d.geometry.PointCloud,
        x_range: tuple[float, float] | None = None,
        y_range: tuple[float, float] | None = None,
        z_range: tuple[float, float] | None = None,
    ) -> o3d.geometry.PointCloud:
        """Remove points outside the given axis-aligned bounding ranges.

        Parameters
        ----------
        pcd     : input point cloud (not modified in place)
        x_range : (min, max) in metres, or None to skip this axis
        y_range : (min, max) in metres, or None to skip this axis
        z_range : (min, max) in metres, or None to skip this axis

        Returns
        -------
        o3d.geometry.PointCloud  — filtered copy
        """
        pts = np.asarray(pcd.points)
        mask = np.ones(len(pts), dtype=bool)
        if x_range is not None:
            mask &= (pts[:, 0] >= x_range[0]) & (pts[:, 0] <= x_range[1])
        if y_range is not None:
            mask &= (pts[:, 1] >= y_range[0]) & (pts[:, 1] <= y_range[1])
        if z_range is not None:
            mask &= (pts[:, 2] >= z_range[0]) & (pts[:, 2] <= z_range[1])
        return pcd.select_by_index(np.where(mask)[0])

    def annotate_global_points(
        self,
        points_global: np.ndarray,
        radius: int = 8,
        color: tuple[int, int, int] = (0, 0, 255),
        labels: list[str] | None = None,
    ) -> list[np.ndarray | None]:
        """Capture one RGB frame from each camera and annotate 3-D points
        (expressed in the global frame) onto each image.

        Points that project outside the image boundary or lie behind the
        camera are silently ignored for that camera.

        Parameters
        ----------
        points_global : float (N, 3)
            3-D points in the global coordinate frame.
        radius : int
            Circle radius for each projected point annotation, in pixels.
        color : (B, G, R)
            OpenCV BGR colour used for circles and labels.
        labels : list[str] | None
            Optional per-point text labels.  Length must equal N when given.

        Returns
        -------
        list[np.ndarray | None]
            One annotated BGR image per camera, in camera index order.
            Entry is None if the camera failed to deliver a frame.
        """
        points_global = np.asarray(points_global, dtype=np.float64)
        if points_global.ndim == 1:
            points_global = points_global[np.newaxis, :]   # (1, 3)
        N = len(points_global)

        # Homogeneous: (4, N)
        pts_h = np.hstack([points_global, np.ones((N, 1))]).T

        annotated = []
        for cam_idx, cam in enumerate(self.cameras):
            color_bgr, _ = cam.read()
            if color_bgr is None:
                annotated.append(None)
                continue

            out = color_bgr.copy()
            H, W = out.shape[:2]

            # Transform global → colour camera frame
            T_global2cam = _invert_T(self.get_cam_to_global(cam_idx))   # (4,4)
            pts_cam = (T_global2cam @ pts_h).T[:, :3]                   # (N, 3)

            K, dist = _intrinsics_to_cv(cam.get_intrinsics())

            for i, (xc, yc, zc) in enumerate(pts_cam):
                if zc <= 0:
                    continue   # behind camera

                px, _ = cv2.projectPoints(
                    np.array([[[xc, yc, zc]]], dtype=np.float64),
                    np.zeros(3), np.zeros(3), K, dist,
                )
                u, v = int(round(px[0, 0, 0])), int(round(px[0, 0, 1]))

                if not (0 <= u < W and 0 <= v < H):
                    continue   # outside image boundary

                cv2.circle(out, (u, v), radius, color, -1)
                if labels is not None:
                    cv2.putText(
                        out, labels[i], (u + radius + 2, v),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
                    )

            annotated.append(out)

        return annotated

    # ------------------------------------------------------------------
    # Chessboard corner annotation
    # ------------------------------------------------------------------

    def annotate_chessboard_corners(
        self,
        cam_idx: int,
        board_size: tuple[int, int],
        color_bgr: np.ndarray | None = None,
        corner_positions_global: np.ndarray | None = None,
        font_scale: float = 0.4,
    ) -> np.ndarray | None:
        """Detect chessboard corners in a frame from *cam_idx* and annotate
        each one with its flat index (and its global-frame XYZ if provided).

        The corner ordering follows OpenCV's convention: row-major, left to
        right, top to bottom as seen in the image.  Use this to verify that
        your ``corner_positions_global`` array is in the correct order before
        calling :meth:`calibrate_from_global_corners`.

        Parameters
        ----------
        cam_idx : int
            Camera to capture from (0-based).
        board_size : (cols, rows)
            Number of inner corners along (width, height) of the board.
        color_bgr : (H, W, 3) uint8 | None
            Pre-captured frame to use.  Captures a fresh frame when None.
        corner_positions_global : float (N, 3) | None
            If given, each corner label also shows its global XYZ.
            N must equal ``cols * rows``.
        font_scale : float
            Scale of the index label text.

        Returns
        -------
        np.ndarray | None
            Annotated BGR image, or None if no frame / corners not found.
        """
        if color_bgr is None:
            color_bgr, _ = self.cameras[cam_idx].read()
        if color_bgr is None:
            return None

        gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, board_size)
        if not found:
            print(f"[annotate_chessboard_corners] cam{cam_idx}: board not detected.")
            return color_bgr.copy()

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        out = color_bgr.copy()
        cv2.drawChessboardCorners(out, board_size, corners, found)

        for idx, corner in enumerate(corners):
            u, v = int(round(corner[0, 0])), int(round(corner[0, 1]))
            if corner_positions_global is not None:
                x, y, z = corner_positions_global[idx]
                label = f"{idx}({x:.2f},{y:.2f},{z:.2f})"
            else:
                label = str(idx)

            # Dark background for readability
            (tw, th), bl = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )
            cv2.rectangle(
                out, (u + 3, v - th - 3), (u + 3 + tw, v + bl), (0, 0, 0), cv2.FILLED
            )
            cv2.putText(
                out, label, (u + 3, v),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), 1, cv2.LINE_AA,
            )

        return out

    # ------------------------------------------------------------------
    # Single-camera calibration using known global corner positions
    # ------------------------------------------------------------------

    def calibrate_from_global_corners(
        self,
        cam_idx: int,
        corner_positions_global: np.ndarray,
        board_size: tuple[int, int],
        n_frames: int = 10,
    ) -> np.ndarray:
        """Calibrate camera *cam_idx* using a chessboard whose corner positions
        are known in the global frame.

        Unlike :meth:`calibrate_camera_extrinsics`, this method requires only
        one camera and no reference camera — you supply the absolute 3-D
        coordinates of every inner corner directly.

        The corner order must match OpenCV's ``findChessboardCorners`` output:
        row-major, left to right, top to bottom as seen in the image.  Run
        :meth:`annotate_chessboard_corners` first to verify the ordering.

        Parameters
        ----------
        cam_idx : int
            0-based index of the camera to calibrate.
        corner_positions_global : float (N, 3)
            Known 3-D positions (in metres) of each inner chessboard corner
            in the global coordinate frame.  N must equal cols × rows.
        board_size : (cols, rows)
            Number of inner corners along (width, height) of the board.
        n_frames : int
            Number of valid detections to accumulate before computing the
            final pose (averaged over all frames for robustness).

        Controls
        --------
        's'   — capture the current frame (only when board is detected)
        'q'   — stop early and compute with frames gathered so far
        ESC   — abort without saving

        Returns
        -------
        T_cam_to_global : float64 (4, 4)
            Also stored in ``self._calibrated_extrinsics[cam_idx]`` and
            reflected in ``self.R_global2rgb_list`` / ``self.T_global2rgb_list``.

        Raises
        ------
        RuntimeError
            If fewer than 3 frames were collected or calibration is aborted.
        """
        cols, rows = board_size
        expected_n = cols * rows
        obj_pts_global = np.asarray(corner_positions_global, dtype=np.float64)
        if obj_pts_global.shape != (expected_n, 3):
            raise ValueError(
                f"corner_positions_global must be ({expected_n}, 3) for a "
                f"{cols}×{rows} board, got {obj_pts_global.shape}."
            )

        cam = self.cameras[cam_idx]
        K, dist = _intrinsics_to_cv(cam.get_intrinsics())
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        print(f"Calibrating cam{cam_idx} from known global corner positions.")
        print(f"Board: {cols}×{rows} inner corners.  Target: {n_frames} frames.")
        print("'s'=capture  'q'=finish early  ESC=abort\n")

        rvecs_acc, tvecs_acc = [], []

        while len(rvecs_acc) < n_frames:
            color_bgr, _ = cam.read()
            if color_bgr is None:
                continue

            gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, board_size)

            vis = color_bgr.copy()
            cv2.drawChessboardCorners(vis, board_size, corners, found)
            status_col = (0, 255, 0) if found else (0, 100, 255)
            status = (
                f"Frames: {len(rvecs_acc)}/{n_frames}  "
                f"board:{'OK' if found else '--'}"
                + ("  Press 's' to capture" if found else "")
            )
            cv2.putText(vis, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, status_col, 2, cv2.LINE_AA)
            cv2.imshow(f"Calibration  cam{cam_idx}", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:   # ESC
                cv2.destroyAllWindows()
                raise RuntimeError("Calibration aborted.")
            elif key == ord('q'):
                break
            elif key == ord('s') and found:
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                ok, rvec, tvec = cv2.solvePnP(
                    obj_pts_global, corners2, K, dist, flags=cv2.SOLVEPNP_ITERATIVE
                )
                if not ok:
                    print("  solvePnP failed for this frame, skipping.")
                    continue
                rvecs_acc.append(rvec.ravel())
                tvecs_acc.append(tvec.ravel())
                print(f"  Captured frame {len(rvecs_acc)}/{n_frames}")

        cv2.destroyAllWindows()

        if len(rvecs_acc) < 3:
            raise RuntimeError(
                f"Need at least 3 frames, only got {len(rvecs_acc)}."
            )

        # Average rotation (Rodrigues mean) and translation
        mean_rvec = np.mean(rvecs_acc, axis=0)
        mean_tvec = np.mean(tvecs_acc, axis=0)
        R_global2cam, _ = cv2.Rodrigues(mean_rvec)

        # solvePnP gives: p_cam = R @ p_global + t  →  T_global2cam
        T_global2cam = _make_T(R_global2cam, mean_tvec)
        T_cam2global = _invert_T(T_global2cam)

        # Derive depth-to-global via device depth→color hardware extrinsic
        T_depth2color  = cam.get_depth_to_color_T()
        T_depth2global = T_cam2global @ T_depth2color

        self._calibrated_extrinsics[cam_idx]       = T_cam2global
        self._calibrated_depth_extrinsics[cam_idx] = T_depth2global
        self.Transform_depth2global[cam_idx]        = T_depth2global
        self.R_global2rgb_list[cam_idx] = T_global2cam[:3, :3]
        self.T_global2rgb_list[cam_idx] = T_global2cam[:3, [3]].T

        # Pretty-print copy-paste block (same format as calibrate_camera_extrinsics)
        def _fmt_3x3(M: np.ndarray) -> str:
            rows_s = "\n".join(
                "    [" + ", ".join(f"{v: .8f}" for v in row) + "],"
                for row in M
            )
            return f"np.array([\n{rows_s}\n])"

        def _fmt_4x4(M: np.ndarray) -> str:
            rows_s = "\n".join(
                "    [" + ", ".join(f"{v: .8f}" for v in row) + "],"
                for row in M
            )
            return f"np.array([\n{rows_s}\n])"

        t_rgb = T_global2cam[:3, 3]
        t_rgb_str = ", ".join(f"{v:.8f}" for v in t_rgb)
        n = cam_idx
        print("\n" + "─" * 66)
        print("  Paste into D435CameraArray.__init__:")
        print("─" * 66)
        print(f"self.R_global2rgb_cam{n} = {_fmt_3x3(T_global2cam[:3, :3])}")
        print(f"self.T_global2rgb_cam{n} = np.array([[{t_rgb_str}]])")
        print(f"self.T_depth2global_cam{n} = {_fmt_4x4(T_depth2global)}")
        print("─" * 66)

        return T_cam2global

    def stop(self):
        for cam in self.cameras:
            cam.stop()

    def __len__(self) -> int:
        return len(self.cameras)

    def __getitem__(self, idx: int) -> D435Camera:
        return self.cameras[idx]

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.stop()


# ----------------------------------------------------------------------
# Quick smoke-test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    # ── mode selection ──────────────────────────────────────────────────
    # python camera_driver.py            → annotate test points (default)
    # python camera_driver.py view       → live colour + depth for every camera
    # python camera_driver.py speckle    → live speckle-tracked cloud (3-D + displacement)
    # python camera_driver.py calibrate  → calibrate cam extrinsics
    # python camera_driver.py pointcloud → all four point clouds in the global frame
    MODE = sys.argv[1] if len(sys.argv) > 1 else "test"

    # calibration settings (only used in 'calibrate' mode)
    REF_CAM     = 1   # cam0 — known extrinsics
    TARGET_CAM  = 3   # cam2 — to be calibrated
    BOARD_SIZE  = (8, 11)
    SQUARE_SIZE = 0.025   # metres
    N_FRAMES    = 30

    with D435CameraArray() as array:
        array.enable_ir_emitters()

        if MODE == "view":
            # ── live view: colour + depth for every camera ───────────────
            print(f"Viewing {len(array)} cameras. Press 'q' to quit.")
            while True:
                frames = array.read_all()
                for i, (color, depth) in enumerate(frames):
                    if color is not None:
                        cv2.imshow(f"Color {i}", color)
                    if depth is not None:
                        d_vis = (depth / (depth.max() + 1e-6) * 255).astype(np.uint8)
                        cv2.imshow(f"Depth {i}", cv2.applyColorMap(d_vis, cv2.COLORMAP_JET))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            cv2.destroyAllWindows()

        elif MODE == "calibrate":
            # ── Step 1: calibrate target camera ──────────────────────────
            array.calibrate_camera_extrinsics(
                ref_cam_idx    = REF_CAM,
                target_cam_idx = TARGET_CAM,
                board_size     = BOARD_SIZE,
                square_size    = SQUARE_SIZE,
                n_frames       = N_FRAMES,
            )

        elif MODE in ("pointcloud", "debug"):
            # ── one-shot coloured point cloud, one colour per camera ──────
            X_RANGE = (0,  0.6)   # metres
            Y_RANGE = (0,  1.0)
            Z_RANGE = ( 0.0,  2.0)

            print(f"Capturing global-frame point clouds from {len(array)} cameras ...")
            pcds = array.capture_colored_pointclouds()
            pcds = [D435CameraArray.filter_pointcloud(p, X_RANGE, Y_RANGE, Z_RANGE)
                    for p in pcds]

            geometries = [o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)]
            geometries.extend(pcds)

            # legend
            labels = {0: "red", 1: "green", 2: "blue", 3: "yellow"}
            print("\nColour legend:")
            for idx, name in labels.items():
                print(f"  cam{idx} → {name}")

            o3d.visualization.draw_geometries(
                geometries,
                window_name=("Global point clouds | cam0=RED | cam1=GREEN | "
                             "cam2=BLUE | cam3=YELLOW"),
                width=1280, height=720,
            )
        elif MODE == "test":
            # print(array.get_depth_to_global(3))
            pts = np.array([
                [0.03, 0.03, 0.03],
                [0.53, 0.03, 0.03],
                [0.03, 0.73, 0.03],
                [0.53, 0.73, 0.03],
                [0.03, 0.03, 0.47],
                [0.53, 0.03, 0.47],
                [0.03, 0.73, 0.47],
                [0.53, 0.73, 0.47],
            ])
            images = array.annotate_global_points(pts, radius=10, color=(0, 255, 0))
            for i, img in enumerate(images):
                if img is not None:
                    cv2.imshow(f"cam{i}", img)
            cv2.waitKey(0)

        elif MODE == 'cali_global':
            # Perform global calibration
            img = array.annotate_chessboard_corners(2, [8,11])
            cv2.imshow("corners", img); cv2.waitKey(0)
            # print(get_board_pos_cam3())
            # board_pos = get_board_pos_cam2()
            # print(board_pos)
            # array.calibrate_from_global_corners(2, board_pos,[8,11])

        elif MODE == "speckle":
            # ── live speckle-tracked point cloud ──────────────────────────
            print("Speckle tracker running.")
            print("  c  — reset displacement reference")
            print("  r  — force re-detection")
            print("  q  — quit\n")
            with SpeckleVisualizer("Speckle Cloud — global frame") as vis:
                while True:
                    cloud = array.get_speckle_cloud()
                    # print("type of point cloud:", type(cloud))
                    print("size of points: ", len(cloud.points) if cloud is not None else 0)
                    vis.update(cloud)

                    if cloud is not None:
                        print(f"\r  pts={len(cloud.points):4d}", end="", flush=True)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('c'):
                        array.reset_speckle_reference()
                    elif key == ord('r'):
                        array.reset_speckle_tracker()

            print()

        else:
            print(f"Unknown mode '{MODE}'. Use: view | pointcloud | speckle | calibrate")
