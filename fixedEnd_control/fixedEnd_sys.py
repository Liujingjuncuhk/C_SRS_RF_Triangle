import os
import sys
import inspect
import time
import numpy as np
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
if parentdir not in sys.path:
    sys.path.insert(0, parentdir)

try:
    from .fixedEnd_motor_controller import FeetechUDPDriver
except ImportError:
    from fixedEnd_motor_controller import FeetechUDPDriver

class FixedEndSystem:
    def __init__(self, description_file: str = "./models/flat_tri_surface/C_SRS_description_bary.pkl"
):
        """
        Initialize the FixedEndSystem with a description file.

        Args:
            description_file (str): Path to the description file.
        """
        self.description_file = description_file
        from C_SRS_fixedEnd import C_SRS_fixedEnd
        from camera_driver_fixedEnd import FixedEndCamera

        self.c_srs = C_SRS_fixedEnd(description_file)
        self.motor_controller = FeetechUDPDriver()
        self.camera = FixedEndCamera()
        self.nCable = 6
        self.calibrated_motor_pos = [2048, 1934, 1932, 2048, 2048, 2048]
        self.calibrated_cable_length = (np.array([437, 445, 437, 292, 272, 287])*1e-3).tolist()  # Default calibrated lengths in meters
        self.default_speed = 0.01
        self.stepPerm = 4096/(0.05*np.pi)
        self.mPerStep = 1/self.stepPerm
        self.moving_dir = [1, 1, 1, -1, -1, -1]  # Direction for each motor

    def _as_cable_vector(self, values, name: str) -> np.ndarray:
        values = np.asarray(values, dtype=float).reshape(-1)
        if values.shape[0] != self.nCable:
            raise ValueError(f"Expected {self.nCable} {name} values, got {values.shape[0]}.")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} must contain only finite values.")
        return values

    def _lengths_to_motor_positions(self, target_lengths) -> list[int]:
        target_lengths = self._as_cable_vector(target_lengths, "target length")
        calibrated_lengths = np.asarray(self.calibrated_cable_length, dtype=float)
        calibrated_positions = np.asarray(self.calibrated_motor_pos, dtype=float)
        moving_dir = np.asarray(self.moving_dir, dtype=float)
        steps_needed = (target_lengths - calibrated_lengths) / self.mPerStep
        target_positions = calibrated_positions + np.rint(steps_needed * moving_dir)
        return target_positions.astype(int).tolist()

    def _speeds_to_motor_steps(self, speed) -> list[int]:
        if speed is None:
            speed = [self.default_speed for _ in range(self.nCable)]
        elif isinstance(speed, (int, float)):
            speed = [speed for _ in range(self.nCable)]
        speed = self._as_cable_vector(speed, "speed")
        if np.any(speed < 0.0):
            raise ValueError("speed values must be non-negative.")
        motor_speed = np.rint(speed / self.mPerStep).astype(int)
        motor_speed[(speed > 0.0) & (motor_speed < 1)] = 1
        return motor_speed.tolist()

    def move_to_length(self, target_lengths: list, speed: list = None):
        """
        Move the motors to achieve the target cable lengths.

        Args:
            target_lengths (list): List of target cable lengths.
            speed (int): Speed of the motors, in m/s.
        """
        target_positions = self._lengths_to_motor_positions(target_lengths)
        motor_speed = self._speeds_to_motor_steps(speed)
        self.motor_controller.move_to_position(target_positions, motor_speed)

    def move_to_length_timed(self, target_lengths: list, consumed_time: float):
        """
        Move the motors to achieve the target cable lengths with timing.

        Args:
            target_lengths (list): List of target cable lengths.
            consumed_time (float): Time to consume for the movement, in seconds.
        """
        if consumed_time <= 0:
            raise ValueError("consumed_time must be positive.")
        target_lengths = self._as_cable_vector(target_lengths, "target length")
        current_lengths = np.asarray(self.get_cur_length(), dtype=float)
        speed = np.abs(target_lengths - current_lengths) / consumed_time
        self.move_to_length(target_lengths.tolist(), speed.tolist())
        time.sleep(consumed_time)

    def move_length_rel_timed(self, length_diffs: list, consumed_time: float):
        """
        Move the motors by relative cable length differences with timing.

        Args:
            length_diffs (list): List of relative cable length differences.
            consumed_time (float): Time to consume for the movement, in seconds.
        """
        if len(length_diffs) != self.nCable:
            raise ValueError(f"Expected {self.nCable} length differences, got {len(length_diffs)}.")
        if consumed_time <= 0:
            raise ValueError("consumed_time must be positive.")
        
        # Calculate target lengths based on current lengths
        length_diffs = self._as_cable_vector(length_diffs, "length difference")
        current_lengths = np.asarray(self.get_cur_length(), dtype=float)
        target_lengths = current_lengths + length_diffs
        speed = np.abs(length_diffs) / consumed_time
        
        # Move to target lengths with timing
        self.move_to_length(target_lengths.tolist(), speed.tolist())

    def move_length_rel(self, length_diffs: list, speed: list = None):
        """
        Move the motors by relative cable length differences.

        Args:
            length_diffs (list): List of relative cable length differences.
            speed (int or list): Speed of the motors, in m/s.
        """
        if len(length_diffs) != self.nCable:
            raise ValueError(f"Expected {self.nCable} length differences, got {len(length_diffs)}.")
        
        # Calculate target lengths based on current lengths
        length_diffs = self._as_cable_vector(length_diffs, "length difference")
        current_lengths = np.asarray(self.get_cur_length(), dtype=float)
        target_lengths = current_lengths + length_diffs
        
        # Move to target lengths
        self.move_to_length(target_lengths.tolist(), speed)

    def get_cur_length(self):
        """
        Get the current cable lengths based on the motor positions.

        Returns:
            list: Current cable lengths.
        """
        cur_motor_pos = self.motor_controller.read_positions()
        cur_motor_pos = self._as_cable_vector(cur_motor_pos, "motor position")
        calibrated_positions = np.asarray(self.calibrated_motor_pos, dtype=float)
        calibrated_lengths = np.asarray(self.calibrated_cable_length, dtype=float)
        moving_dir = np.asarray(self.moving_dir, dtype=float)
        steps_diff = (cur_motor_pos - calibrated_positions) * moving_dir
        cur_lengths = calibrated_lengths + steps_diff * self.mPerStep
        return cur_lengths.tolist()
    
    def execute_traj(self, time_stamp: list, traj_length: list):
        """
        Execute a trajectory of cable lengths over time.

        Args:
            time_stamp (list): Monotonic time stamps in seconds. The first
                timestamp must be 0.
            traj_length (list): Cable lengths for each timestamp, shape
                (num_waypoints, nCable). The first row is assumed to be the
                current cable length and is not commanded.
        """
        time_stamp = np.asarray(time_stamp, dtype=float).reshape(-1)
        traj_length = np.asarray(traj_length, dtype=float)

        if time_stamp.ndim != 1:
            raise ValueError("time_stamp must be a 1D list or array.")
        if len(time_stamp) == 0:
            raise ValueError("time_stamp must contain at least one timestamp.")
        if not np.isclose(time_stamp[0], 0.0):
            raise ValueError("time_stamp must start at 0.")
        if np.any(np.diff(time_stamp) <= 0):
            raise ValueError("time_stamp must be strictly increasing.")
        if traj_length.ndim != 2 or traj_length.shape[1] != self.nCable:
            raise ValueError(
                f"traj_length must have shape (num_waypoints, {self.nCable})."
            )
        if traj_length.shape[0] != len(time_stamp):
            raise ValueError(
                "traj_length and time_stamp must have the same number of waypoints."
            )

        if len(time_stamp) == 1:
            return

        start_time = time.monotonic()
        delta = 1e-3
        for i in range(1, len(time_stamp)):
            prev_lengths = traj_length[i - 1]
            target_lengths = traj_length[i]
            dt = time_stamp[i] - time_stamp[i - 1]
            segment_speed = np.abs(target_lengths - prev_lengths) / dt
            self.move_to_length(target_lengths.tolist(), segment_speed.tolist())

            next_time = start_time + time_stamp[i]
            remaining = next_time - time.monotonic()
            if remaining > delta:
                time.sleep((remaining - delta))
            while time.monotonic() < next_time:
                pass


    def close(self):
        if hasattr(self, "camera") and self.camera is not None:
            self.camera.stop()
        if hasattr(self, "motor_controller") and self.motor_controller is not None:
            self.motor_controller.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    
if __name__ == "__main__":
    filtered_region = [0.02, 0.3, 0, 0.16, -0.02, 0.02]
    fixedEnd_sys = FixedEndSystem()
    cl = fixedEnd_sys.get_cur_length()
    print("initial cl is: ", cl)
    # Q_list, tension= fixedEnd_sys.c_srs.FKD_static_length(fixedEnd_sys.c_srs.vertices, cl)
    # pcd = fixedEnd_sys.camera.get_depth_pointcloud(region = filtered_region)
    # pts_points = np.asarray(pcd.points)
    # vert_initial = fixedEnd_sys.c_srs.q_to_vertices(Q_list[-1])
    # plotter = fixedEnd_sys.c_srs.visualize_fb_surface_w_gt(vert_initial, pts_points)
    # plotter.show()