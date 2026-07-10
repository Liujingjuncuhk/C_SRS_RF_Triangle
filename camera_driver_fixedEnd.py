import pyrealsense2 as rs
import numpy as np
import os
import time
os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts/truetype/dejavu"
import cv2
os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts/truetype/dejavu"
import open3d as o3d
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont


APRILTAG_FAMILIES = {
    "16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "36h11": cv2.aruco.DICT_APRILTAG_36h11,
}
DEFAULT_APRILTAG_FAMILIES = ("36h11", "16h5", "25h9", "36h10")
_ARUCO_PARAMS = cv2.aruco.DetectorParameters()

CAMERA_SERIAL = '346522071233'
RGB_frame = (1280, 720)
depth_frame = (640, 480)

RGB_FRAME = RGB_frame
DEPTH_FRAME = depth_frame


@dataclass
class TagDetection:
    tag_id: int
    corners: np.ndarray
    family: str = ""

    @property
    def center(self) -> np.ndarray:
        return self.corners.mean(axis=0)


def _make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t).reshape(3)
    return T


def _invert_T(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def _intrinsics_to_cv(intr: rs.intrinsics) -> tuple[np.ndarray, np.ndarray]:
    K = np.array([
        [intr.fx, 0.0, intr.ppx],
        [0.0, intr.fy, intr.ppy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    return K, np.asarray(intr.coeffs, dtype=np.float64)


def _tag_corners_from_center(center: np.ndarray, tag_size: float) -> np.ndarray:
    h = tag_size / 2.0
    center = np.asarray(center, dtype=np.float64).reshape(3)
    return center + np.array([
        [-h, h, 0.0],
        [h, h, 0.0],
        [h, -h, 0.0],
        [-h, -h, 0.0],
    ], dtype=np.float64)


def _get_label_font(size: int = 20) -> ImageFont.ImageFont:
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    if os.path.isfile(font_path):
        return ImageFont.truetype(font_path, size=size)
    return ImageFont.load_default()


def _draw_label_pil(
    rgb: np.ndarray,
    text: str,
    top_left: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    font = _get_label_font()
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    pad = 4
    image_w, image_h = image.size
    tx, ty = np.asarray(top_left, dtype=int)
    tx = int(np.clip(tx, 0, max(0, image_w - text_w - 2 * pad)))
    ty = int(np.clip(ty, 0, max(0, image_h - text_h - 2 * pad)))
    draw.rectangle(
        (tx - pad, ty - pad, tx + text_w + pad, ty + text_h + pad),
        fill=(0, 0, 0),
    )
    draw.text((tx, ty), text, font=font, fill=color)
    return np.array(image)


def _label_top_left(corners: np.ndarray, label_height: int = 24) -> np.ndarray:
    min_xy = corners.min(axis=0)
    max_xy = corners.max(axis=0)
    x = int(min_xy[0])
    y_above = int(min_xy[1] - label_height - 10)
    if y_above >= 0:
        return np.array([x, y_above])
    return np.array([x, int(max_xy[1] + 10)])


def _draw_tag_local_axes(rgb: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Draw tag-local +x/+y axes from detected AprilTag corners on an RGB image."""
    corners = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    center = corners.mean(axis=0)
    x_end = 0.5 * (corners[1] + corners[2])
    y_end = 0.5 * (corners[0] + corners[1])

    c = tuple(np.round(center).astype(int))
    x = tuple(np.round(x_end).astype(int))
    y = tuple(np.round(y_end).astype(int))
    cv2.arrowedLine(rgb, c, x, (255, 0, 0), 3, tipLength=0.25)
    cv2.arrowedLine(rgb, c, y, (0, 255, 0), 3, tipLength=0.25)
    cv2.circle(rgb, c, 4, (255, 255, 255), -1)
    return rgb


def _get_detector(tag_family: str) -> cv2.aruco.ArucoDetector:
    if tag_family not in APRILTAG_FAMILIES:
        valid_families = ", ".join(APRILTAG_FAMILIES)
        raise ValueError(f"Unknown AprilTag family '{tag_family}'. Use one of: {valid_families}.")
    dictionary = cv2.aruco.getPredefinedDictionary(APRILTAG_FAMILIES[tag_family])
    return cv2.aruco.ArucoDetector(dictionary, _ARUCO_PARAMS)


def _detect_apriltags(
    rgb: np.ndarray,
    tag_families: tuple[str, ...] | list[str] = DEFAULT_APRILTAG_FAMILIES,
) -> list[TagDetection]:
    if rgb is None:
        return []
    if rgb.ndim == 2:
        gray = rgb
    elif rgb.ndim == 3 and rgb.shape[2] == 3:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    else:
        raise ValueError("rgb must have shape (H, W), (H, W, 3), or be a valid RGB image.")

    detections = []
    seen = set()
    for family in tag_families:
        detector = _get_detector(family)
        corners_list, ids, _ = detector.detectMarkers(gray)
        if ids is None:
            continue
        ids = np.asarray(ids).reshape(-1)
        for i in range(len(ids)):
            tag_id = int(ids[i])
            key = (family, tag_id)
            if key in seen:
                continue
            seen.add(key)
            detections.append(
                TagDetection(tag_id, corners_list[i].reshape(4, 2), family)
            )
    return detections


def take_region(pts, region):
    """Return points inside [xmin, xmax, ymin, ymax, zmin, zmax]."""
    region = np.asarray(region, dtype=np.float64).reshape(-1)
    if region.shape[0] != 6:
        raise ValueError("region must be [xmin, xmax, ymin, ymax, zmin, zmax].")
    xmin, xmax, ymin, ymax, zmin, zmax = region

    if isinstance(pts, o3d.geometry.PointCloud):
        points = np.asarray(pts.points)
        mask = _region_mask(points, xmin, xmax, ymin, ymax, zmin, zmax)

        filtered = o3d.geometry.PointCloud()
        filtered.points = o3d.utility.Vector3dVector(points[mask])

        colors = np.asarray(pts.colors)
        if len(colors) == len(points):
            filtered.colors = o3d.utility.Vector3dVector(colors[mask])

        normals = np.asarray(pts.normals)
        if len(normals) == len(points):
            filtered.normals = o3d.utility.Vector3dVector(normals[mask])
        return filtered

    points = np.asarray(pts)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("pts must be an (N, 3) array, a single (3,) point, or an Open3D point cloud.")
    mask = _region_mask(points, xmin, xmax, ymin, ymax, zmin, zmax)
    return points[mask]


def _region_mask(points, xmin, xmax, ymin, ymax, zmin, zmax):
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3).")
    mask = np.isfinite(points).all(axis=1)
    mask &= points[:, 0] >= xmin
    mask &= points[:, 0] <= xmax
    mask &= points[:, 1] >= ymin
    mask &= points[:, 1] <= ymax
    mask &= points[:, 2] >= zmin
    mask &= points[:, 2] <= zmax
    return mask


class FixedEndCamera:
    """Single Intel RealSense camera used by the fixed-end system.

    The color stream is configured as RGB8, so :meth:`read` returns an RGB
    image, not OpenCV's usual BGR layout. Depth is returned in metres.
    """

    def __init__(
        self,
        serial: str = CAMERA_SERIAL,
        rgb_frame: tuple[int, int] = RGB_FRAME,
        depth_frame: tuple[int, int] = DEPTH_FRAME,
        fps: int = 30,
        align_depth_to_rgb: bool = True,
        tag_size: float = 0.022,
    ):
        self.serial = serial
        self.rgb_frame = rgb_frame
        self.depth_frame = depth_frame
        self.fps = fps
        self.align_depth_to_rgb = align_depth_to_rgb
        self.tag_size = tag_size
        self.T_world_to_rgb = np.array([[-0.99987068,  0.01594147,  0.00212133,  0.2278519 ],
       [ 0.01590486,  0.99974093, -0.01628195, -0.08544792],
       [-0.00238033, -0.0162461 , -0.99986519,  0.37095216],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
        self.T_rgb_to_world = np.array([[-0.99987068,  0.01590486, -0.00238033,  0.23006446],
       [ 0.01594147,  0.99974093, -0.0162461 ,  0.08782001],
       [ 0.00212133, -0.01628195, -0.99986519,  0.36902755],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
        self.T_world_to_depth = np.array([[-0.99978521,  0.02069741,  0.00107336,  0.21317994],
       [ 0.02067925,  0.99967642, -0.01481442, -0.08729099],
       [-0.00137964, -0.01478904, -0.9998897 ,  0.37024969],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
        self.T_depth_to_world = np.array([[-0.99978521,  0.02067925, -0.00137964,  0.21545007],
       [ 0.02069741,  0.99967642, -0.01478904,  0.08832611],
       [ 0.00107336, -0.01481442, -0.9998897 ,  0.36868687],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])

        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(
            rs.stream.color,
            rgb_frame[0],
            rgb_frame[1],
            rs.format.rgb8,
            fps,
        )
        config.enable_stream(
            rs.stream.depth,
            depth_frame[0],
            depth_frame[1],
            rs.format.z16,
            fps,
        )

        self._profile = self._pipeline.start(config)
        self.depth_scale = (
            self._profile.get_device()
            .first_depth_sensor()
            .get_depth_scale()
        )
        self._align = rs.align(rs.stream.color) if align_depth_to_rgb else None

        depth_stream = self._profile.get_stream(rs.stream.depth)
        rgb_stream = self._profile.get_stream(rs.stream.color)
        extr = depth_stream.get_extrinsics_to(rgb_stream)
        self.T_depth_to_rgb = _make_T(
            np.asarray(extr.rotation, dtype=np.float64).reshape(3, 3),
            np.asarray(extr.translation, dtype=np.float64),
        )
        self.T_rgb_to_depth = _invert_T(self.T_depth_to_rgb)

    def read(self, timeout_ms: int = 5000) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Read one RGB frame and one depth frame.

        Returns
        -------
        rgb : np.ndarray | None
            ``uint8`` array with shape ``(H, W, 3)`` in RGB order.
        depth : np.ndarray | None
            ``float32`` array with depth in metres. If alignment is enabled,
            this depth image is aligned to the RGB frame.
        """
        try:
            frames = self._pipeline.wait_for_frames(timeout_ms=timeout_ms)
        except RuntimeError:
            return None, None

        if self._align is not None:
            frames = self._align.process(frames)

        rgb_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        rgb = np.asanyarray(rgb_frame.get_data()) if rgb_frame else None
        depth = (
            np.asanyarray(depth_frame.get_data()).astype(np.float32) * self.depth_scale
            if depth_frame
            else None
        )
        return rgb, depth

    def read_rgb_depth(
        self,
        timeout_ms: int = 5000,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Alias for :meth:`read` with an explicit name."""
        return self.read(timeout_ms=timeout_ms)

    def visualize_depth(
        self,
        max_depth: float = 1.6,
        post_process: bool = False,
        timeout_ms: int = 5000,
        window_name: str = "Depth reading",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Show the current depth reading as a color-mapped image.

        Returns
        -------
        depth : np.ndarray
            Metric depth image in metres.
        depth_vis : np.ndarray
            BGR color visualization suitable for OpenCV display.
        """
        if post_process:
            _, depth_frame = self.get_raw_frames(timeout_ms=timeout_ms)
            if depth_frame is None:
                raise RuntimeError("No depth frame received.")
            depth_frame = self._post_process_depth_frame(depth_frame)
            depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * self.depth_scale
        else:
            _, depth = self.read_rgb_depth(timeout_ms=timeout_ms)
        if depth is None:
            raise RuntimeError("No depth frame received.")

        if max_depth <= 0:
            raise ValueError("max_depth must be positive.")

        valid = np.isfinite(depth) & (depth > 0.0)
        depth_scaled = np.zeros(depth.shape, dtype=np.uint8)
        depth_scaled[valid] = np.clip(depth[valid] / max_depth * 255.0, 0, 255).astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_scaled, cv2.COLORMAP_JET)
        depth_vis[~valid] = np.array([0, 0, 0], dtype=np.uint8)

        cv2.imshow(window_name, depth_vis)
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)
        return depth, depth_vis

    def _post_process_depth_frame(self, depth_frame: rs.depth_frame) -> rs.frame:
        """Apply RealSense-style depth cleanup filters for visualization."""
        depth_to_disparity = rs.disparity_transform(True)
        disparity_to_depth = rs.disparity_transform(False)
        spatial = rs.spatial_filter()
        temporal = rs.temporal_filter()
        hole_filling = rs.hole_filling_filter()
        try:
            spatial.set_option(rs.option.filter_magnitude, 2)
            spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
            spatial.set_option(rs.option.filter_smooth_delta, 20)
            hole_filling.set_option(rs.option.holes_fill, 2)
        except RuntimeError:
            pass

        depth_frame = depth_to_disparity.process(depth_frame)
        depth_frame = spatial.process(depth_frame)
        depth_frame = temporal.process(depth_frame)
        depth_frame = disparity_to_depth.process(depth_frame)
        depth_frame = hole_filling.process(depth_frame)
        return depth_frame

    def set_depth_options(
        self,
        laser_power: float | None = None,
        emitter_enabled: bool | None = None,
        visual_preset: int | None = None,
    ) -> None:
        """Set common D435 depth options, if the sensor supports them."""
        depth_sensor = self._profile.get_device().first_depth_sensor()

        def _set_option(option, value) -> None:
            if not depth_sensor.supports(option):
                return
            try:
                option_range = depth_sensor.get_option_range(option)
                value = float(np.clip(float(value), option_range.min, option_range.max))
                depth_sensor.set_option(option, value)
            except RuntimeError:
                pass

        if visual_preset is not None:
            _set_option(rs.option.visual_preset, visual_preset)
        if laser_power is not None:
            _set_option(rs.option.laser_power, laser_power)
        if emitter_enabled is not None:
            _set_option(rs.option.emitter_enabled, 1.0 if emitter_enabled else 0.0)

    def configure_depth_for_surface(
        self,
        laser_power: float = 360.0,
        emitter_enabled: bool = True,
        visual_preset: int = 4,
    ) -> None:
        """Bias the RealSense depth sensor toward denser close-range surfaces.

        Stream FPS and resolution must be chosen before the pipeline starts, so
        use ``FixedEndCamera(fps=15)`` or ``FixedEndCamera(fps=6)`` when you
        want a lower frame rate.
        """
        self.set_depth_options(
            visual_preset=visual_preset,
            laser_power=laser_power,
            emitter_enabled=emitter_enabled,
        )

    def get_raw_frames(
        self,
        timeout_ms: int = 5000,
    ) -> tuple[rs.frame | None, rs.depth_frame | None]:
        """Return raw RealSense color/depth frames without depth-to-RGB alignment."""
        try:
            frames = self._pipeline.wait_for_frames(timeout_ms=timeout_ms)
        except RuntimeError:
            return None, None
        return frames.get_color_frame() or None, frames.get_depth_frame() or None

    def detect_tag(
        self,
        rgb: np.ndarray,
        annotate: bool = True,
        tag_families: tuple[str, ...] | list[str] = DEFAULT_APRILTAG_FAMILIES,
        draw_local_axes: bool = True,
    ) -> tuple[np.ndarray, list[TagDetection]]:
        """Detect AprilTags in an RGB image and optionally annotate tag IDs."""
        detections = _detect_apriltags(rgb, tag_families=tag_families)

        if not annotate:
            return rgb, detections

        annotated = rgb.copy()
        for det in detections:
            corners = det.corners.astype(np.int32)
            cv2.polylines(
                annotated,
                [corners.reshape((-1, 1, 2))],
                isClosed=True,
                color=(0, 255, 0),
                thickness=2,
            )
            corner0 = tuple(corners[0])
            cv2.circle(annotated, corner0, 5, (255, 0, 0), -1)
            if draw_local_axes:
                annotated = _draw_tag_local_axes(annotated, det.corners)

            label = f"id:{det.tag_id}"
            annotated = _draw_label_pil(annotated, label, _label_top_left(det.corners))

        return annotated, detections

    def calibrate_extrinsic(
        self,
        tag_ids: list[int],
        tag_poses,
        tag_size: float | None = None,
        timeout_ms: int = 5000,
    ) -> dict[str, np.ndarray]:
        """Calibrate RGB and depth extrinsics from fixed AprilTag centers.

        ``tag_poses`` can be either a dict ``{tag_id: center_xyz}`` or an
        ``(N, 3)`` array whose rows match ``tag_ids``. Tags are assumed to have
        identity rotation in the world frame, so their corners are generated in
        the world XY plane around each provided center.
        """
        rgb, _ = self.read(timeout_ms=timeout_ms)
        if rgb is None:
            raise RuntimeError("No RGB frame received for extrinsic calibration.")

        _, detections = self.detect_tag(rgb, annotate=False)
        detections_by_id = {det.tag_id: det for det in detections}
        tag_centers = self._normalize_tag_centers(tag_ids, tag_poses)
        selected_ids = [tag_id for tag_id in tag_ids if tag_id in detections_by_id]
        if len(selected_ids) < 1:
            raise RuntimeError("None of the requested AprilTags were detected.")

        size = self.tag_size if tag_size is None else tag_size
        object_points = []
        image_points = []
        for tag_id in selected_ids:
            object_points.append(_tag_corners_from_center(tag_centers[tag_id], size))
            image_points.append(detections_by_id[tag_id].corners.astype(np.float64))

        object_points = np.vstack(object_points).astype(np.float64)
        image_points = np.vstack(image_points).astype(np.float64)
        K, dist = _intrinsics_to_cv(self.get_rgb_intrinsics())
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            K,
            dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            raise RuntimeError("solvePnP failed during extrinsic calibration.")

        R_world_to_rgb, _ = cv2.Rodrigues(rvec)
        return self.update_transformation({
            "T_world_to_rgb": _make_T(R_world_to_rgb, tvec),
        })

    def update_transformation(self, transformations: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Update stored extrinsics from a calibrate_extrinsic-style result."""
        if not isinstance(transformations, dict):
            raise TypeError("transformations must be a dict of 4x4 matrices.")

        def _matrix_or_none(key: str) -> np.ndarray | None:
            if key not in transformations:
                return None
            T = np.asarray(transformations[key], dtype=np.float64)
            if T.shape != (4, 4):
                raise ValueError(f"{key} must have shape (4, 4).")
            if not np.allclose(T[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-8):
                raise ValueError(f"{key} must be a homogeneous transform with bottom row [0, 0, 0, 1].")
            return T

        T_world_to_rgb = _matrix_or_none("T_world_to_rgb")
        T_rgb_to_world = _matrix_or_none("T_rgb_to_world")
        T_world_to_depth = _matrix_or_none("T_world_to_depth")
        T_depth_to_world = _matrix_or_none("T_depth_to_world")

        if T_world_to_rgb is None and T_rgb_to_world is not None:
            T_world_to_rgb = _invert_T(T_rgb_to_world)
        if T_world_to_rgb is None and T_world_to_depth is not None:
            T_world_to_rgb = self.T_depth_to_rgb @ T_world_to_depth
        if T_world_to_rgb is None and T_depth_to_world is not None:
            T_world_to_rgb = self.T_depth_to_rgb @ _invert_T(T_depth_to_world)
        if T_world_to_rgb is None:
            raise ValueError("transformations must include an RGB or depth world transform.")

        if T_rgb_to_world is None:
            T_rgb_to_world = _invert_T(T_world_to_rgb)
        if T_world_to_depth is None:
            T_world_to_depth = self.T_rgb_to_depth @ T_world_to_rgb
        if T_depth_to_world is None:
            T_depth_to_world = _invert_T(T_world_to_depth)

        self.T_world_to_rgb = T_world_to_rgb
        self.T_rgb_to_world = T_rgb_to_world
        self.T_world_to_depth = T_world_to_depth
        self.T_depth_to_world = T_depth_to_world

        return {
            "T_world_to_rgb": self.T_world_to_rgb,
            "T_rgb_to_world": self.T_rgb_to_world,
            "T_world_to_depth": self.T_world_to_depth,
            "T_depth_to_world": self.T_depth_to_world,
        }

    def _normalize_tag_centers(self, tag_ids: list[int], tag_poses) -> dict[int, np.ndarray]:
        if isinstance(tag_poses, dict):
            return {
                int(tag_id): np.asarray(tag_poses[tag_id], dtype=np.float64).reshape(3)
                for tag_id in tag_ids
            }

        tag_poses = np.asarray(tag_poses, dtype=np.float64)
        if tag_poses.shape != (len(tag_ids), 3):
            raise ValueError("tag_poses must be a dict or an array with shape (len(tag_ids), 3).")
        return {
            int(tag_id): tag_poses[i]
            for i, tag_id in enumerate(tag_ids)
        }

    def project_points_to_rgb(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project world-frame 3D points into the RGB image.

        Returns
        -------
        pixels : np.ndarray
            ``(N, 2)`` pixel coordinates.
        valid_mask : np.ndarray
            Boolean mask for points that are in front of the RGB camera and
            inside the current RGB image bounds.
        """
        if self.T_world_to_rgb is None:
            raise RuntimeError("T_world_to_rgb is not set. Run calibrate_extrinsic first.")

        points = np.asarray(points, dtype=np.float64)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        if points.shape[1] == 4:
            w = points[:, [3]]
            if np.any(np.abs(w) < 1e-12):
                raise ValueError("Homogeneous point has near-zero w.")
            points = points[:, :3] / w
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape (N, 3), (3,), or homogeneous shape (N, 4).")

        R_world_to_rgb = self.T_world_to_rgb[:3, :3]
        t_world_to_rgb = self.T_world_to_rgb[:3, 3]
        rvec, _ = cv2.Rodrigues(R_world_to_rgb)
        K, dist = _intrinsics_to_cv(self.get_rgb_intrinsics())
        pixels, _ = cv2.projectPoints(points, rvec, t_world_to_rgb, K, dist)
        pixels = pixels.reshape(-1, 2)

        points_rgb = (R_world_to_rgb @ points.T).T + t_world_to_rgb
        width, height = self.rgb_frame
        valid_mask = points_rgb[:, 2] > 0
        valid_mask &= pixels[:, 0] >= 0
        valid_mask &= pixels[:, 0] < width
        valid_mask &= pixels[:, 1] >= 0
        valid_mask &= pixels[:, 1] < height
        return pixels, valid_mask

    def draw_points(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Draw world-frame 3D points on the current RGB frame and show it."""
        rgb, _ = self.read_rgb_depth()
        if rgb is None:
            raise RuntimeError("No RGB frame received.")

        pixels, valid_mask = self.project_points_to_rgb(points)
        annotated = rgb.copy()
        for u, v in pixels[valid_mask]:
            cv2.circle(annotated, (int(round(u)), int(round(v))), 6, (255, 0, 0), -1)
            cv2.circle(annotated, (int(round(u)), int(round(v))), 9, (255, 255, 255), 2)

        cv2.imshow("RGB projected points", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyWindow("RGB projected points")
        return annotated, pixels, valid_mask

    def get_depth_pointcloud(
        self,
        max_depth: float | None = 1.6,
        voxel_size: float = 0.005,
        region=None,
        post_process: bool = False,
        timeout_ms: int = 5000,
    ) -> o3d.geometry.PointCloud:
        """Capture native depth readings and return them as a world-frame point cloud."""
        if self.T_depth_to_world is None:
            raise RuntimeError("T_depth_to_world is not set. Run calibrate_extrinsic first.")

        _, depth_frame = self.get_raw_frames(timeout_ms=timeout_ms)
        if depth_frame is None:
            raise RuntimeError("No depth frame received.")
        if post_process:
            depth_frame = self._post_process_depth_frame(depth_frame)

        pointcloud = rs.pointcloud()
        points = pointcloud.calculate(depth_frame)
        pts_depth = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
        pts_depth = pts_depth.astype(np.float64)

        valid = np.isfinite(pts_depth).all(axis=1)
        valid &= pts_depth[:, 2] > 0.0
        if max_depth is not None:
            valid &= pts_depth[:, 2] <= max_depth
        pts_depth = pts_depth[valid]

        if len(pts_depth) == 0:
            return o3d.geometry.PointCloud()

        pts_world = (
            self.T_depth_to_world
            @ np.hstack([pts_depth, np.ones((len(pts_depth), 1))]).T
        ).T[:, :3]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_world)
        pcd.paint_uniform_color([0.65, 0.78, 0.95])
        if region is not None:
            pcd = take_region(pcd, region)
        if voxel_size > 0:
            pcd = pcd.voxel_down_sample(voxel_size)
        return pcd

    def get_raw_pointcloud(
        self,
        timeout_ms: int = 5000,
    ) -> o3d.geometry.PointCloud:
        """Capture native depth readings and return them as a point cloud in the depth camera frame."""
        _, depth_frame = self.get_raw_frames(timeout_ms=timeout_ms)
        if depth_frame is None:
            raise RuntimeError("No depth frame received.")

        pointcloud = rs.pointcloud()
        points = pointcloud.calculate(depth_frame)
        pts_depth = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
        pts_depth = pts_depth.astype(np.float64)

        valid = np.isfinite(pts_depth).all(axis=1)
        valid &= pts_depth[:, 2] > 0.0
        pts_depth = pts_depth[valid]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_depth)
        pcd.paint_uniform_color([0.65, 0.78, 0.95])
        return pcd

    def draw_points_depth(
        self,
        points: np.ndarray,
        max_depth: float | None = 1.6,
        voxel_size: float = 0.005,
        region=None,
        post_process: bool = False,
        point_radius: float = 0.008,
        timeout_ms: int = 5000,
    ) -> tuple[o3d.geometry.PointCloud, list[o3d.geometry.Geometry]]:
        """Show depth readings as a point cloud with world-frame points overlaid."""
        pcd = self.get_depth_pointcloud(
            max_depth=max_depth,
            voxel_size=voxel_size,
            region=region,
            post_process=post_process,
            timeout_ms=timeout_ms,
        )

        points = np.asarray(points, dtype=np.float64)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        if points.shape[1] == 4:
            w = points[:, [3]]
            if np.any(np.abs(w) < 1e-12):
                raise ValueError("Homogeneous point has near-zero w.")
            points = points[:, :3] / w
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape (N, 3), (3,), or homogeneous shape (N, 4).")

        geometries: list[o3d.geometry.Geometry] = [pcd]
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        geometries.append(axes)
        for point in points:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=point_radius)
            sphere.translate(point)
            sphere.paint_uniform_color([1.0, 0.0, 0.0])
            geometries.append(sphere)

        o3d.visualization.draw_geometries(
            geometries,
            window_name="Depth point cloud with projected points",
        )
        return pcd, geometries

    def get_rgb_intrinsics(self) -> rs.intrinsics:
        stream = self._profile.get_stream(rs.stream.color)
        return stream.as_video_stream_profile().get_intrinsics()

    def get_depth_intrinsics(self) -> rs.intrinsics:
        stream = self._profile.get_stream(rs.stream.depth)
        return stream.as_video_stream_profile().get_intrinsics()

    def stop(self) -> None:
        self._pipeline.stop()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.stop()


def read_rgb_depth(
    camera: FixedEndCamera | None = None,
    timeout_ms: int = 5000,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Read one RGB/depth pair using an existing camera or a temporary one."""
    if camera is not None:
        return camera.read_rgb_depth(timeout_ms=timeout_ms)

    with FixedEndCamera() as fixed_end_camera:
        return fixed_end_camera.read_rgb_depth(timeout_ms=timeout_ms)
    

if __name__ == "__main__":
    camera = FixedEndCamera()
    time.sleep(3.0)
    rgb, depth = camera.read_rgb_depth()
    # pcd = camera.get_raw_pointcloud()
    # visualize the point cloud
    # o3d.visualization.draw_geometries([pcd], window_name="Depth point cloud")
    # add coordinate frame to the point cloud
    # axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    # o3d.visualization.draw_geometries([pcd, axes], window_name="Depth point cloud with coordinate frame")
    # exit(0)
    # annotated, detections = camera.detect_tag(rgb, annotate=True)
    # cv2.imshow("Annotated RGB", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    # cv2.waitKey(0)


    # transformations = camera.calibrate_extrinsic(tag_ids=[7, 4, 11, 3], tag_poses={7: [0.038, -0.02, 0.0], 4: [0.073, -0.02, 0.0], 11: [0.038, 0.178, 0.0], 3: [0.073, 0.178, 0.0]})  
    filtered_region = [-0.02, 0.3, 0, 0.16, -0.02, 0.02]

    # print("Calibrated transformations:")
    # print(transformations)
    points = np.array([[0.27,0.08,0.025],[0.02, 0.0, 0.0], [0.02, 0.08, 0.0], [0.02, 0.16, 0.0]])
    pcd, geometries= camera.draw_points_depth(points, region=filtered_region)


    
    
