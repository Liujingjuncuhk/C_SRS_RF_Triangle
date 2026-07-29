import os
import sys
import inspect
import time
import numpy as np
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
if parentdir not in sys.path:
    sys.path.insert(0, parentdir)
from flying_carpet import Flying_carpet
from camera_driver_flying_carpet import D435CameraArray
try:
    from .flying_carpet_motor import FeetechUDPDriver
except ImportError:
    from flying_carpet_motor import FeetechUDPDriver
import pickle
import pyvista as pv
import cv2

class Flying_carpet_sys():
    def __init__(self, robot_description_file = "./models/flying_carpet/flying_carpet_description_bary.pkl"):
        self.robot = Flying_carpet(robot_description_file)
        self.motor_ids = [1,2,3,4,5,6,7,8]
        self.motor_driver = FeetechUDPDriver()
        self.camera_driver = D435CameraArray()
        time.sleep(0.1)
        self.stepPerm = 4096/(0.08*np.pi)
        self.mPerStep = 1/self.stepPerm
        self.calibrate_cable_length =  (np.array([512, 498, 527, 493, 438, 413, 448, 405])*1e-3).tolist()
        self.calibrate_motor_pos = [2048, 2048,2048,2048,2048,2048,2048,2048]
        self.initial_cable_length = self.get_cur_length()
        self.initial_motor_pos = self.motor_driver.read_positions()
        self.nCable = self.robot.nCable
        self.cur_cable_length = self.get_cur_length()
        # Q_list_all, self.starting_vertices, cable_tension = self.robot.FKD_time(self.cur_cable_length, 10, self.robot.vertices)
        # self.robot.visualize_vert(self.starting_vertices)

    def get_cur_length(self):
        cur_motor_pos = self.motor_driver.read_positions()
        cur_length = self.calibrate_cable_length.copy()
        for i in range(self.robot.nCable):
            # Calculate the current cable length based on motor position
            cur_length[i] = self.calibrate_cable_length[i] + (cur_motor_pos[i] - self.calibrate_motor_pos[i]) * self.mPerStep
        return cur_length

    def move_to_length(self, target_lengths, target_speed):
        # Implement the logic to move the cables to the target lengths
        tar_step = self.initial_motor_pos.copy()
        for i in range(self.robot.nCable):
            # print("length diff for cable", i+1, ":", target_lengths[i] - self.cur_cable_length[i])
            tar_step[i] = int((target_lengths[i] - self.calibrate_cable_length[i]) * self.stepPerm + self.calibrate_motor_pos[i] + 0.5) 
        v_step = [4500 for _ in range(self.robot.nCable)]
        if target_speed is None:
            v_step = [4500 for _ in range(self.robot.nCable)]
        else:
            for i in range(self.robot.nCable):
                v_step[i] = int(target_speed[i] * self.stepPerm + 0.5)
        v_step = [abs(v) for v in v_step]
        # self.checkpoint("Moving to target lengths. Press enter to continue or q to quit.")
        # input("Press enter to continue.")
        self.motor_driver.move_to_position(tar_step, speeds = v_step)

    def move_length_d(self, d_length, target_speed):
        cur_length = self.get_cur_length()
        for i in range(self.robot.nCable):
            cur_length[i] += d_length[i]
        self.move_to_length(cur_length, target_speed)

    def move_to_length_timed(self, target_lengths, duration):
        if duration == 0:
            self.move_to_length(target_lengths, None)
        else:
            cur_length = self.get_cur_length()
            v_length = [0 for _ in range(self.robot.nCable)]
            for i in range(self.robot.nCable):
                v_length[i] = abs((target_lengths[i] - cur_length[i]) / duration)
            self.move_to_length(target_lengths, v_length)

    def move_length_d_timed(self, d_length, duration):
        if duration == 0:
            self.move_length_d(d_length, None)
        else:
            cur_length = self.get_cur_length()
            for i in range(self.robot.nCable):
                cur_length[i] += d_length[i]
            self.move_to_length_timed(cur_length, duration)

    def execute_cable_length_traj_maxSpeed(self, cable_length_list, time_list, feedback = False):
        """
        Follow a cable length trajectory in real time at the highest motor speed.

        Each command in cable_length_list is dispatched at the scheduled time in
        time_list (measured from when this method is called).  Speed is set to the
        motor maximum (4500 steps/s, i.e. target_speed=None).

        Parameters
        ----------
        cable_length_list : list of list/array, shape (N, nCable)
            Cable length targets for each waypoint.
        time_list : list/array of float, length N
            Scheduled dispatch time for each waypoint, starting at 0.
        """
        BUSY_WAIT_THRESHOLD = 2e-3  # busy-wait for the last 2 ms for precision
        t_start = time.perf_counter()
        length_feedback_list = [self.initial_cable_length.copy() for _ in range(len(cable_length_list))]
        t_list_feedback = [0 for _ in range(len(cable_length_list))]
        i = 0
        
        for cmd, t in zip(cable_length_list, time_list):
            t_target = t_start + t
            # Coarse sleep to avoid spinning the full wait
            coarse = t_target - time.perf_counter() - BUSY_WAIT_THRESHOLD
            if coarse > 0:
                time.sleep(coarse)
            # Busy-wait for the remaining time
            while time.perf_counter() < t_target:
                pass
            self.move_to_length(cmd, None)
            if feedback:
                length_feedback_list[i] = self.get_cur_length()
                t_list_feedback[i] = time.perf_counter() - t_start
                i += 1
        return length_feedback_list, t_list_feedback

    def plot_cmd_vs_feedback(self, length_command_list, length_feedback_list, time_list, time_feedback_list):
        """
        Plot commanded vs feedback cable length for all 8 cables.

        Parameters
        ----------
        length_command_list : list of list/array, shape (N, nCable)
        length_feedback_list : list of list/array, shape (N, nCable)
        time_list : list/array of float, length N
        """
        import matplotlib.pyplot as plt
        cmd = np.array(length_command_list)
        fb  = np.array(length_feedback_list)
        t   = np.array(time_list)
        t_fb = np.array(time_feedback_list)
        fig, axes = plt.subplots(4, 2, figsize=(12, 10))
        axes = axes.flatten()
        for i in range(self.robot.nCable):
            axes[i].plot(t, cmd[:, i] * 1e3, label='Command', linewidth=1.2)
            axes[i].plot(t_fb, fb[:, i] * 1e3, label='Feedback', linewidth=1.2, linestyle='--')
            axes[i].set_title(f'Cable {i + 1}')
            axes[i].set_ylabel('Length (mm)')
            axes[i].set_xlabel('Time (s)')
            axes[i].legend(loc='upper right', fontsize=7)
            axes[i].grid(True)
        fig.suptitle('Cable Length: Command vs Feedback')
        plt.tight_layout()
        plt.show()

    def interpolate_cable_length_cmd(self, cable_length_list, time_list, hz):
        """
        Interpolate a cable length command trajectory to a uniform frequency.

        Parameters
        ----------
        cable_length_list : list of list/array, shape (N_waypoints, nCable)
            Cable length waypoints. The first entry must equal the current cable
            length and corresponds to time_list[0] = 0.
        time_list : list/array of float, length N_waypoints
            Timestamps for each waypoint, starting at 0.
        hz : float
            Desired output frequency in Hz.

        Returns
        -------
        new_cable_length_list : list of list, shape (N_new, nCable)
            Interpolated cable lengths at uniform 1/hz intervals.
        new_time_list : list of float, length N_new
            Uniform time stamps starting at 0, step = 1/hz.
        """
        t = np.array(time_list, dtype=float)
        cl = np.array(cable_length_list, dtype=float)  # (N_waypoints, nCable)
        dt = 1.0 / hz
        new_time = np.arange(0.0, t[-1] + dt * 0.5, dt)  # inclusive of final time
        new_time = np.clip(new_time, 0.0, t[-1])
        n_cables = cl.shape[1]
        new_cl = np.zeros((len(new_time), n_cables))
        for i in range(n_cables):
            new_cl[:, i] = np.interp(new_time, t, cl[:, i])
        return new_cl.tolist(), new_time.tolist()

    def retrive_last_executed_length(self, length_file: str = 'last_cable_length.pickle'):
        with open(length_file, 'rb') as f:
            last_lengths = pickle.load(f)
        time.sleep(0.1)
        self.cur_cable_length = last_lengths.copy()
        self.move_to_length(self.initial_cable_length, [0.08 for _ in range(self.robot.nCable)])
        return last_lengths

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
        for i in range(1, len(time_stamp)):
            prev_lengths = traj_length[i - 1]
            target_lengths = traj_length[i]
            dt = time_stamp[i] - time_stamp[i - 1]
            segment_speed = np.abs(target_lengths - prev_lengths) / dt
            self.move_to_length(target_lengths.tolist(), segment_speed.tolist())

            next_time = start_time + time_stamp[i]
            remaining = next_time - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)

    def visualize_mesh_w_feedback(self, verts, feedback_pts):
        # verts = self.robot.Q_to_vertices(Q)
        mesh = pv.PolyData(verts)
        
        mesh.faces = np.hstack([[4, *tet] for tet in self.robot.tetrahedra])
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, color='blue', show_edges=True, opacity=0.5)
        
        # Add pullpoints
        pullpoints = verts[self.robot.pp_idx]
        plotter.add_points(pullpoints, color='red', point_size=10, render_points_as_spheres=True)
        
        # Add pulley locations
        plotter.add_points(self.robot.pulley_location, color='green', point_size=10, render_points_as_spheres=True)
        # Add lines
        for i in range(self.robot.nCable):
            plotter.add_lines(np.array([self.robot.pulley_location[i], verts[self.robot.pp_idx[i]]]), color='green')
        
        # Add feedback points
        plotter.add_points(feedback_pts, color='yellow', point_size=8, render_points_as_spheres=True)
        
        plotter.show_grid()
        plotter.show_axes()
        # make axis equal
        plotter.set_scale(1, 1, 1)
        plotter.show()

    def visualize_mesh_w_feedback_2(self, verts, feedback_pts1, feedback_pts2):
        # verts = self.robot.Q_to_vertices(Q)
        mesh = pv.PolyData(verts)
        
        mesh.faces = np.hstack([[4, *tet] for tet in self.robot.tetrahedra])
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, color='blue', show_edges=True, opacity=0.5)
        
        # Add pullpoints
        pullpoints = verts[self.robot.pp_idx]
        plotter.add_points(pullpoints, color='red', point_size=10, render_points_as_spheres=True)
        
        # Add pulley locations
        plotter.add_points(self.robot.pulley_location, color='green', point_size=10, render_points_as_spheres=True)
        # Add lines
        for i in range(self.robot.nCable):
            plotter.add_lines(np.array([self.robot.pulley_location[i], verts[self.robot.pp_idx[i]]]), color='green')
        
        # Add feedback points from camera 1
        plotter.add_points(feedback_pts1, color='yellow', point_size=8, render_points_as_spheres=True)
        # Add feedback points from camera 2
        plotter.add_points(feedback_pts2, color='cyan', point_size=8, render_points_as_spheres=True)
        
        plotter.show_grid()
        plotter.show_axes()
        # make axis equal
        plotter.set_scale(1, 1, 1)
        plotter.show()
        
        # return plotter for saving screenshot later
        return plotter

    def visualize_mesh_w_feedback_all(self, verts, pts):
        # pts: list of point arrays, one per camera
        cam_colors = ['yellow', 'cyan', 'magenta', 'orange']
        mesh = pv.PolyData(verts)

        mesh.faces = np.hstack([[4, *tet] for tet in self.robot.tetrahedra])
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, color='blue', show_edges=True, opacity=0.5)

        # Add pullpoints
        pullpoints = verts[self.robot.pp_idx]
        plotter.add_points(pullpoints, color='red', point_size=10, render_points_as_spheres=True)

        # Add pulley locations
        plotter.add_points(self.robot.pulley_location, color='green', point_size=10, render_points_as_spheres=True)
        # Add lines
        for i in range(self.robot.nCable):
            plotter.add_lines(np.array([self.robot.pulley_location[i], verts[self.robot.pp_idx[i]]]), color='green')

        # Add feedback points from each camera
        print("\nColour legend:")
        for idx, pt_set in enumerate(pts):
            color = cam_colors[idx % len(cam_colors)]
            print(f"  cam{idx} → {color}")
            if pt_set is not None and len(pt_set) > 0:
                plotter.add_points(pt_set, color=color, point_size=8, render_points_as_spheres=True)

        plotter.show_grid()
        plotter.show_axes()
        # make axis equal
        plotter.set_scale(1, 1, 1)
        plotter.show()

        # return plotter for saving screenshot later
        return plotter

    def visualize_tar_pts_and_fb_pts(self, tar_pts, fb_pts):
        plotter = pv.Plotter()
        # Add target feedback points
        plotter.add_points(tar_pts, color='red', point_size=10, render_points_as_spheres=True, label='Target FB Points')
        # Add current feedback points
        plotter.add_points(fb_pts, color='blue', point_size=10, render_points_as_spheres=True, label='Current FB Points')
        plotter.add_points(self.robot.pulley_location, color='green', point_size=10, render_points_as_spheres=True)
        plotter.add_legend()
        plotter.show_grid()
        plotter.show_axes()
        plotter.set_scale(1, 1, 1)
        plotter.show()
        return plotter

    def visualize_3d_pts_in_rgb(self, pts_tar, pts_cur, cam_id):
        depth_images, color_images = self.camera_driver.get_pics()
        if cam_id == 1:
            color_image = color_images[0]
            R_global2cam = self.camera_driver.R_global2rgb_cam1
            T_global2cam = self.camera_driver.T_global2rgb_cam1
            color_K = self.camera_driver.color_K_1
            color_dist = self.camera_driver.color_distortion_1
        else:
            color_image = color_images[1]
            R_global2cam = self.camera_driver.R_global2rgb_cam2
            T_global2cam = self.camera_driver.T_global2rgb_cam2
            color_K = self.camera_driver.color_K_2
            color_dist = self.camera_driver.color_distortion_2
        
        for point in pts_tar:
            dist = color_dist
            rvec = cv2.Rodrigues(R_global2cam)[0]
            tvec = T_global2cam.reshape((3, 1))
            imgpts, _ = cv2.projectPoints(np.array([point]), rvec, tvec, color_K, dist)
            u, v = imgpts[0, 0]
            # print(f"Point {point} projects to pixel: ({u}, {v})")
            cv2.circle(color_image, (int(u), int(v)), 5, (255, 0, 0), -1)

        # for point in pts_cur:
        #     dist = color_dist
        #     rvec = cv2.Rodrigues(R_global2cam)[0]
        #     tvec = T_global2cam.reshape((3, 1))
        #     imgpts, _ = cv2.projectPoints(np.array([point]), rvec, tvec, color_K, dist)
        #     u, v = imgpts[0, 0]
        #     # print(f"Point {point} projects to pixel: ({u}, {v})")
        #     cv2.circle(color_image, (int(u), int(v)), 5, (0, 0, 255), -1)
        return color_image

    def icp_feedback_point_matching(self, starting_vert, cur_cable_length, pts_all, max_iter=100, tol=1e-4):
        fb_verts = starting_vert[self.robot.feedback_idx, :]
        fb_verts_new = fb_verts.copy()
        # vert_last = starting_vert.copy()
        starting_Q = self.robot.vertices_to_Q(starting_vert)
        Q_last = starting_Q.copy()
        for iter_count in range(max_iter):
            indices, R, t = icp(fb_verts, pts_all)
            for i in range(self.robot.n_feedback):
                fb_verts_new[i, :] = pts_all[indices[i], :]
            Q_new = self.robot.feedback_cg_weightedmatch(cur_cable_length, starting_Q, fb_verts_new)
            Q_last = starting_Q.copy()
            starting_Q = Q_new.copy()
            diff = np.linalg.norm(Q_new - Q_last)
            print(f"ICP Iteration {iter_count}, diff in Q: {diff}")
            vertices_cor = self.robot.Q_to_vertices(starting_Q)
            fb_verts = vertices_cor[self.robot.feedback_idx, :]
            # robot.visualize_mesh_cor_fbts(starting_vert, vertices_cor, fb_verts)
            if diff < tol:
                print(f"Converged at iteration {iter_count}, diff: {diff}")
                break
        return vertices_cor

    def get_feedback_pts(self, cam_indices: list[int] = [0,1,2,3], 
                         voxel_size: float = 0.005,
        x_range: tuple[float, float] | None = None,
        y_range: tuple[float, float] | None = None,
        z_range: tuple[float, float] | None = None,):
        pts = self.camera_driver.get_point_cloud(cam_indices, voxel_size, x_range, y_range, z_range)
        return_pts = np.asarray(pts.points)
        return return_pts

    def feedback_control(self, tar_vert_fb, starting_vert, max_iter=30, tol=1e-6, out_dir = "figures/feedback_control/"):
        def cal_diff(tar_fb, cur_fb): # both of shape (n_feedback, 3)
            # return sum of squared differences between vertices
            return np.sum((tar_fb - cur_fb)**2) * 0.5 / self.robot.n_feedback
        
        def cal_jacobian(cur_Q, tar_fb, cur_fb):
            Jac_all = self.robot.cal_jacobian_cg(cur_Q).T  # shape (nVert*3, nCable)
            Jac_this = np.zeros((self.robot.nCable, ))
            for i in range(self.robot.n_feedback):
                idx_fb = self.robot.feedback_idx[i]
                diff_vec = cur_fb[i, :] - tar_fb[i, :]
                for j in range(self.robot.nCable):
                    for k in range(3):
                        Jac_this[j] += diff_vec[k] * Jac_all[idx_fb*3 + k, j]
            return Jac_this
        step_size = 0.03
        diff = 1e3
        cur_cl = self.cur_cable_length.copy()
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        diff_list = []
        Q_list, vert_length, cable_tension, R_list_1212 = self.robot.FKD_static(cur_cl, starting_vert)
        for iter_count in range(max_iter):
            # take feedback verts
            Q_list, vert_length, cable_tension, R_list_1212 = self.robot.FKD_static(cur_cl, vert_length)
            minBox, maxBox = self.robot.get_feedback_box(vert_length)
            self.checkpoint("start capturing feedback points.")
            pts1, pts2 = self.camera_driver.get_pts_global()
            points_global_1 = self.camera_driver.filter_pts(pts1, minBox, maxBox)
            points_global_2 = self.camera_driver.filter_pts(pts2, minBox, maxBox)
            pts_all = np.vstack((points_global_1, points_global_2))
            vertices_cor = self.icp_feedback_point_matching(vert_length.copy(), cur_cl, pts_all, max_iter = 30)
            Q_cur = self.robot.vertices_to_Q(vertices_cor)
            cur_fb_verts = vertices_cor[self.robot.feedback_idx, :]
            plotter = self.visualize_tar_pts_and_fb_pts(tar_vert_fb, cur_fb_verts)
            screenshot_path = os.path.join(out_dir, f"fb_control_iter_{iter_count+1:03d}.png")
            plotter.screenshot(screenshot_path)
            color_image = self.visualize_3d_pts_in_rgb(tar_vert_fb, pts_all, cam_id=1)
            rgb_screenshot_path = os.path.join(out_dir, f"fb_control_rgb_iter_{iter_count+1:03d}.png")
            cv2.imwrite(rgb_screenshot_path, color_image)
            diff = cal_diff(tar_vert_fb, cur_fb_verts)
            diff_list.append(diff)
            if diff < tol:
                print(f"Feedback control converged at iteration {iter_count}, diff: {diff}")
                break
            Jac = cal_jacobian(Q_cur, tar_vert_fb, cur_fb_verts)
            # update cable lengths
            
            dl = step_size * Jac.flatten()
            print(f"Iteration {iter_count}, diff: {diff}, dl: {-dl}")
            check_i = input("Press 'q' to quit, any other key to continue: ")
            if check_i.lower() == 'q':
                print("Exiting feedback control.")
                break
            for i in range(self.robot.nCable):
                cur_cl[i] -= dl[i]
            self.move_to_length(cur_cl, [0.08 for _ in range(self.robot.nCable)])
        with open(os.path.join(out_dir, "diff_list.pickle"), 'wb') as f:
            pickle.dump(diff_list, f)
        
    def feedback_control_occulusion(self, tar_vert_fb, starting_vert, max_iter=30, tol=1e-6, out_dir = "figures/feedback_control_occulusion/"):
        def cal_diff(tar_fb, cur_fb): # both of shape (n_feedback, 3)
            # return sum of squared differences between vertices
            return np.sum((tar_fb - cur_fb)**2) * 0.5 / self.robot.n_feedback
        
        def cal_jacobian(cur_Q, tar_fb, cur_fb):
            Jac_all = self.robot.cal_jacobian_cg(cur_Q).T  # shape (nVert*3, nCable)
            Jac_this = np.zeros((self.robot.nCable, ))
            for i in range(self.robot.n_feedback):
                idx_fb = self.robot.feedback_idx[i]
                diff_vec = cur_fb[i, :] - tar_fb[i, :]
                for j in range(self.robot.nCable):
                    for k in range(3):
                        Jac_this[j] += diff_vec[k] * Jac_all[idx_fb*3 + k, j]
            return Jac_this
        step_size = 0.03
        diff = 1e3
        cur_cl = self.cur_cable_length.copy()
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        diff_list = []
        Q_list, vert_length, cable_tension, R_list_1212 = self.robot.FKD_static(cur_cl, starting_vert)
        for iter_count in range(max_iter):
            # take feedback verts
            Q_list, vert_length, cable_tension, R_list_1212 = self.robot.FKD_static(cur_cl, vert_length)
            minBox, maxBox = self.robot.get_feedback_box(vert_length)
            self.checkpoint("start capturing feedback points.")
            pts1, pts2 = self.camera_driver.get_pts_global()
            points_global_1 = self.camera_driver.filter_pts(pts1, minBox, maxBox)
            points_global_2 = self.camera_driver.filter_pts(pts2, minBox, maxBox)
            pts_all = np.vstack((points_global_1, points_global_2))
            vertices_cor = self.icp_feedback_point_matching(vert_length.copy(), cur_cl, pts_all, max_iter = 30)
            Q_cur = self.robot.vertices_to_Q(vertices_cor)
            cur_fb_verts = vertices_cor[self.robot.feedback_idx, :]
            plotter = self.visualize_tar_pts_and_fb_pts(tar_vert_fb, cur_fb_verts)
            screenshot_path = os.path.join(out_dir, f"fb_control_iter_{iter_count+1:03d}.png")
            plotter.screenshot(screenshot_path)
            color_image = self.visualize_3d_pts_in_rgb(tar_vert_fb, pts_all, cam_id=1)
            rgb_screenshot_path = os.path.join(out_dir, f"fb_control_rgb_iter_{iter_count+1:03d}.png")
            cv2.imwrite(rgb_screenshot_path, color_image)
            diff = cal_diff(tar_vert_fb, cur_fb_verts)
            diff_list.append(diff)
            if diff < tol:
                print(f"Feedback control converged at iteration {iter_count}, diff: {diff}")
                break
            Jac = cal_jacobian(Q_cur, tar_vert_fb, cur_fb_verts)
            # update cable lengths
            
            dl = step_size * Jac.flatten()
            print(f"Iteration {iter_count}, diff: {diff}, dl: {-dl}")
            check_i = input("Press 'q' to quit, any other key to continue: ")
            if check_i.lower() == 'q':
                print("Exiting feedback control.")
                break
            for i in range(self.robot.nCable):
                cur_cl[i] -= dl[i]
            self.move_to_length(cur_cl, [0.08 for _ in range(self.robot.nCable)])
        with open(os.path.join(out_dir, "diff_list.pickle"), 'wb') as f:
            pickle.dump(diff_list, f)


    def exit_all(self):
        """
        Exit all motors and close the driver.
        """
        self.move_to_length_timed(self.initial_cable_length, 2)

    def checkpoint(self, message:str):
        user_input = input(f"{message} (Press 'q' to quit, any other key to continue): ")
        if user_input.lower() == 'q':
            print("Exiting program.")
            self.exit_all()
            exit(0)

if __name__ == "__main__":
    flying_carpet_sys = Flying_carpet_sys()

    vert = flying_carpet_sys.starting_vertices
    xrange = (np.min(vert[:,0]-0.01), np.max(vert[:,0]+0.01))
    yrange = (np.min(vert[:,1]-0.01), np.max(vert[:,1]+0.01))
    zrange = (np.min(vert[:,2]-0.01), np.max(vert[:,2]+0.01))
    initial_pts = flying_carpet_sys.get_feedback_pts(cam_indices=[0,1], x_range=xrange, y_range=yrange, z_range=zrange)
    flying_carpet_sys.robot.visualize_vert_w_fb(flying_carpet_sys.starting_vertices, initial_pts)


