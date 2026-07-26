import csv
from pathlib import Path
import numpy as np
import pyvista as pv
import os
import sys
import inspect
import time
import matplotlib.pyplot as plt
import numpy as np
import pickle
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from C_SRS_fixedEnd import C_SRS_fixedEnd, IK_MLP
import pickle


# translation_2add = np.array([0.03608945, 0.2295359,  0.0021551])
translation_2add = np.array([0.03305734, 0.23004783, 0.00180389])
# translation_2add = np.array([0.03166899, 0.23057712, 0.00117929])

def get_dense_target(target_list, nSample):
    # interpolate target_list to nSample
    target_list = np.asarray(target_list, dtype=float)
    if target_list.ndim != 2 or target_list.shape[1] != 3:
        raise ValueError("target_list must have shape (number_of_targets, 3)")
    if target_list.shape[0] < 2:
        raise ValueError("target_list must contain at least two targets")

    dense_target = np.zeros((nSample, 3), dtype=float)
    for i in range(3):
        dense_target[:, i] = np.interp(
            np.linspace(0, target_list.shape[0] - 1, nSample),
            np.arange(target_list.shape[0]),
            target_list[:, i]
        )
    return dense_target

def get_tracking_ball_pos(filename, cut_time):
    """Read the final Tx, Ty and Tz columns from a tracking CSV file.

    Args:
        filename: Path to the tracking CSV file.
        cut_time: Number of seconds to keep, starting at the first timestamp.

    Returns:
        A float NumPy array with shape (number_of_samples, 3).
    """
    if cut_time < 0:
        raise ValueError("cut_time must be non-negative")

    positions = []

    with open(filename, "r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.reader(csv_file)

        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"Tracking data file is empty: {filename}")

        if [column.strip() for column in header[-3:]] != ["Tx", "Ty", "Tz"]:
            raise ValueError(
                f"Expected the last three CSV columns to be Tx, Ty, Tz: {filename}"
            )

        header = [column.strip() for column in header]
        try:
            time_column = header.index("Time [sec]")
        except ValueError as error:
            raise ValueError(
                f"CSV does not contain a Time [sec] column: {filename}"
            ) from error

        start_time = None
        for row_number, row in enumerate(reader, start=2):
            if not row or all(not value.strip() for value in row):
                continue
            if len(row) <= time_column or len(row) < 3:
                raise ValueError(
                    f"Row {row_number} does not contain the required columns"
                )

            try:
                timestamp = float(row[time_column])
            except ValueError as error:
                raise ValueError(
                    f"Invalid timestamp on row {row_number}: {row[time_column]}"
                ) from error

            if start_time is None:
                start_time = timestamp

            elapsed_time = timestamp - start_time
            if elapsed_time > cut_time:
                break

            try:
                positions.append([float(value) for value in row[-3:]])
            except ValueError as error:
                raise ValueError(
                    f"Invalid Tx, Ty or Tz value on row {row_number}: {row[-3:]}"
                ) from error

    return np.asarray(positions, dtype=float).reshape(-1, 3)


def plot_tracking_ball_trajectory(positions, save_path=None, show=True):
    """Draw a 3D trajectory from an (N, 3) position array."""
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (number_of_samples, 3)")
    if positions.shape[0] == 0:
        raise ValueError("positions cannot be empty")

    figure = plt.figure(figsize=(9, 7))
    axes = figure.add_subplot(111, projection="3d")

    axes.plot(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        color="tab:blue",
        linewidth=1.5,
        label="Trajectory",
    )
    axes.scatter(*positions[0], color="tab:green", s=55, label="Start")
    axes.scatter(*positions[-1], color="tab:red", s=55, label="End")

    axes.set_xlabel("Tx")
    axes.set_ylabel("Ty")
    axes.set_zlabel("Tz")
    axes.set_title("Tracking Ball 3D Trajectory")
    axes.legend()

    # Use the data ranges to avoid visually stretching one coordinate axis.
    axis_ranges = np.ptp(positions, axis=0)
    axis_ranges[axis_ranges == 0.0] = 1.0
    axes.set_box_aspect(axis_ranges)
    figure.tight_layout()

    if save_path is not None:
        figure.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()

    return figure, axes

def interpolate_normvecs(normalvec_list, n_sample):
    """Interpolate normal vectors to match the number of samples."""
    normalvec_array = np.asarray(normalvec_list, dtype=float)
    if normalvec_array.ndim != 2 or normalvec_array.shape[1] != 3:
        raise ValueError("normalvec_list must have shape (number_of_vectors, 3)")
    if normalvec_array.shape[0] == 0:
        raise ValueError("normalvec_list cannot be empty")

    interpolated_normvecs = np.zeros((n_sample, 3), dtype=float)
    for i in range(3):
        interpolated_normvecs[:, i] = np.interp(
            np.linspace(0, normalvec_array.shape[0] - 1, n_sample),
            np.arange(normalvec_array.shape[0]),
            normalvec_array[:, i]
        )
    return interpolated_normvecs

def get_dense_normvec(c_srs: C_SRS_fixedEnd, vert_list, tracking_ball_positions):
    normalvec_list = []
    for i in range(len(vert_list)):
        vert_this = vert_list[i]
        normalvec_list.append(c_srs.get_ee_normvec(vert_this))

    n_sample = len(tracking_ball_positions)
    interpolated_normvecs = interpolate_normvecs(normalvec_list, n_sample)
    return interpolated_normvecs

def get_ee_trackingball(c_srs: C_SRS_fixedEnd, tracking_ball_positions, normvecs):
    assert len(tracking_ball_positions) == len(normvecs), "Mismatch in number of samples"
    ee_positions = []
    for i in range(len(tracking_ball_positions)):
        ee_positions.append(tracking_ball_positions[i] - normvecs[i]*c_srs.tracker_r)
    return np.array(ee_positions)


def plot_xyz(positions, save_path=None, show=True):
    """Plot Tx, Ty and Tz separately against the sample index."""
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (number_of_samples, 3)")
    if positions.shape[0] == 0:
        raise ValueError("positions cannot be empty")

    sample_indices = np.arange(positions.shape[0])
    coordinate_names = ("Tx", "Ty", "Tz")
    colors = ("tab:red", "tab:green", "tab:blue")

    figure, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(10, 8),
        sharex=True,
    )

    for coordinate_index, (axis, name, color) in enumerate(
        zip(axes, coordinate_names, colors)
    ):
        axis.plot(
            sample_indices,
            positions[:, coordinate_index],
            color=color,
            linewidth=1.3,
        )
        axis.set_ylabel(name)
        axis.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Sample index")
    figure.suptitle("Tracking Ball Position Components")
    figure.tight_layout()

    if save_path is not None:
        figure.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()

    return figure, axes

def plot_xyz_target(target_position, recorded_position, total_t, save_path=None, show=True):
    """Plot target and recorded X, Y and Z positions against time."""
    target_position = np.asarray(target_position, dtype=float)
    recorded_position = np.asarray(recorded_position, dtype=float)

    if target_position.ndim != 2 or target_position.shape[1] != 3:
        raise ValueError("target_position must have shape (number_of_samples, 3)")
    if recorded_position.ndim != 2 or recorded_position.shape[1] != 3:
        raise ValueError("recorded_position must have shape (number_of_samples, 3)")
    if target_position.shape[0] == 0:
        raise ValueError("target_position and recorded_position cannot be empty")
    if target_position.shape[0] != recorded_position.shape[0]:
        raise ValueError("target_position and recorded_position must have equal length")
    if not np.isfinite(total_t) or total_t < 0:
        raise ValueError("total_t must be a non-negative finite number")

    time_values = np.linspace(0.0, total_t, target_position.shape[0])
    coordinate_names = ("X", "Y", "Z")

    figure, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(10, 8),
        sharex=True,
    )

    for coordinate_index, (axis, coordinate_name) in enumerate(
        zip(axes, coordinate_names)
    ):
        axis.plot(
            time_values,
            target_position[:, coordinate_index],
            color="tab:blue",
            linewidth=1.5,
            label="Target",
        )
        axis.plot(
            time_values,
            recorded_position[:, coordinate_index],
            color="tab:red",
            linewidth=1.3,
            label="Recorded EE",
        )
        axis.set_ylabel(f"{coordinate_name} position (m)")
        axis.grid(True, alpha=0.3)
        axis.legend()

    axes[-1].set_xlabel("Time (s)")
    figure.suptitle("Target and Recorded End-Effector Positions")
    figure.tight_layout()

    if save_path is not None:
        figure.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()

    return figure, axes

def plot_target_record_3d(target_list, ee_positions_recorded, save_path=None, show=True):
    """Plot the 3D trajectory of the target and recorded end-effector positions."""
    target_list = np.asarray(target_list, dtype=float)
    ee_positions_recorded = np.asarray(ee_positions_recorded, dtype=float)

    if target_list.ndim != 2 or target_list.shape[1] != 3:
        raise ValueError("target_list must have shape (number_of_samples, 3)")
    if ee_positions_recorded.ndim != 2 or ee_positions_recorded.shape[1] != 3:
        raise ValueError("ee_positions_recorded must have shape (number_of_samples, 3)")
    if target_list.shape[0] == 0 or ee_positions_recorded.shape[0] == 0:
        raise ValueError("target_list and ee_positions_recorded cannot be empty")

    # minx = 0.025, maxx = 

    figure = plt.figure(figsize=(10, 8))
    ax = figure.add_subplot(111, projection='3d')

    ax.plot(target_list[:, 0], target_list[:, 1], target_list[:, 2], label='Target', color='tab:blue')
    ax.plot(ee_positions_recorded[:, 0], ee_positions_recorded[:, 1], ee_positions_recorded[:, 2], label='Recorded EE', color='tab:red')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Trajectory')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Give X, Y and Z the same numeric span and the same visual scale.
    all_positions = np.vstack((target_list, ee_positions_recorded))
    data_min = np.min(all_positions, axis=0)
    data_max = np.max(all_positions, axis=0)
    axis_centers = (data_min + data_max) / 2.0
    equal_range = np.max(data_max - data_min)
    if equal_range == 0.0:
        equal_range = 1.0
    half_range = equal_range * 0.525

    ax.set_xlim(axis_centers[0] - half_range, axis_centers[0] + half_range)
    ax.set_ylim(axis_centers[1] - half_range, axis_centers[1] + half_range)
    ax.set_zlim(axis_centers[2] - half_range, axis_centers[2] + half_range)
    ax.set_box_aspect((1, 1, 1))
    figure.tight_layout()

    if save_path is not None:
        figure.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()

    return figure, ax

def find_good_translation(c_srs: C_SRS_fixedEnd, vert_list, tracking_ball_positions):
    tbp_sim_list = []
    for i in range(len(vert_list)):
        vert = vert_list[i]
        expected_tb = c_srs.get_tracker_pos(vert)
        tbp_sim_list.append(expected_tb)

    tbp_sim_list = get_dense_target(tbp_sim_list, len(tracking_ball_positions))
    # find the average translation between tbp_sim_list and tracking_ball_position
    translation_list = []
    for i in range(len(tbp_sim_list)):
        translation_list.append(tbp_sim_list[i] - tracking_ball_positions[i])
    average_translation = np.mean(translation_list, axis=0)
    print("average_translation: ", average_translation)

    return average_translation

def get_initial_translation(c_srs: C_SRS_fixedEnd, tracking_ball_positions, vert_list):
    tracking_ball_0 = tracking_ball_positions[0]
    vert0 = vert_list[0]
    expected_tb_0 = c_srs.get_tracker_pos(vert0)
    diff = expected_tb_0 - tracking_ball_0
    print("translation to be added: ", diff)
    return diff
        

if __name__ == "__main__":
    traj_file = 'data/IKD_traj_result_triangle.pkl'
    description_file = "./models/flat_tri_surface/C_SRS_description_bary.pkl"
    c_srs = C_SRS_fixedEnd(description_file)
    cut_time = 5.0
    project_dir = Path(__file__).resolve().parent.parent

    tracking_file = (
        project_dir / "data_tracking_ball" / "test_260725_tri_5_000.csv"
    )
    
    tracking_ball_positions = get_tracking_ball_pos(
        tracking_file,
        cut_time=cut_time,
    )
    tracking_ball_positions = tracking_ball_positions*1e-3
    with open(traj_file, 'rb') as f:
        cl_data = pickle.load(f)
        target_list = cl_data['target_list']
        length_cmd_list = cl_data['length_cmd_list']
        vert_list = cl_data['vert_list']
    # find_good_translation(c_srs, vert_list, tracking_ball_positions)
    # exit(0)
    # target_list = np.array([[0.26, 0.08, 0.03],
    #                            [0.25, 0.08, 0.07],
    #                            [0.23, 0.08, 0.08],
    #                            [0.24, 0.08, 0.04],
    #                            [0.26, 0.08, 0.03]]) # parallelogram (faked)

    # target_list = np.array([[0.26, 0.08, 0.03],
    #                            [0.24, 0.06, 0.07],
    #                            [0.24, 0.1, 0.07],
    #                            [0.26, 0.08, 0.03]]) # triangle

    tracking_ball_positions = tracking_ball_positions + translation_2add
    interpolated_normvecs = get_dense_normvec(c_srs, vert_list, tracking_ball_positions)
    ee_positions_recorded = get_ee_trackingball(c_srs, tracking_ball_positions, interpolated_normvecs)
    dense_target = get_dense_target(target_list, ee_positions_recorded.shape[0])
    plot_target_record_3d(dense_target, ee_positions_recorded, show=False)
    plot_xyz_target(dense_target, ee_positions_recorded, cut_time, show=False)
    plt.show()

    data_2save = {"traj": "triangle","cut_time": cut_time, "target_list": target_list,"dense_target": dense_target, "ee_positions_recorded": ee_positions_recorded}
    
    with open("data_tracking_ball/triangle_data_5s.pkl", "wb") as f:
        pickle.dump(data_2save, f)

    
    # plt.shoew()
