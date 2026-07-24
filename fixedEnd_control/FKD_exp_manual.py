import os
import sys
import inspect
import time
import numpy as np
from fixedEnd_sys import FixedEndSystem
import pickle

cl = (np.array([363,357,363, 333,305,335])*1e-3).tolist()

def filter_out_pts(pts, out_region):
    # filter the pts in the out region out!
    mask = (
        (pts[:, 0] < out_region[0]) | (pts[:, 0] > out_region[1]) |
        (pts[:, 1] < out_region[2]) | (pts[:, 1] > out_region[3]) |
        (pts[:, 2] < out_region[4]) | (pts[:, 2] > out_region[5])
    )
    return pts[mask]

if __name__ == "__main__":
    filtered_region = [0.02, 0.3, 0, 0.16, 0, 0.2]
    out_region = [0.22, 0.25, 0, 0.16, 0, 0.06]
    fixedEnd_sys = FixedEndSystem()

    pcd = fixedEnd_sys.camera.get_depth_pointcloud(region = filtered_region)
    pts = np.asarray(pcd.points)
    pts = filter_out_pts(pts, out_region)

    Q_list, cable_tension = fixedEnd_sys.c_srs.FKD_static_length(fixedEnd_sys.c_srs.vertices, cl, show_info=True)

    vert_length = fixedEnd_sys.c_srs.q_to_vertices(Q_list[-1])
    fixedEnd_sys.c_srs.visualize_fb_surface_w_gt(vert_length, pts)
    data_2save = {"cl": cl, "vert_length": vert_length, "pts": pts}
    with open("data/fixedEnd_FKD_exp_manual_data_1.pkl", "wb") as f:
        pickle.dump(data_2save, f)