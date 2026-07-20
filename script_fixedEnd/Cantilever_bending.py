import torch
import torch.nn as nn
import numpy as np
import joblib
import numpy as np
import pyvista as pv
import os
import sys
import inspect
import time
import open3d as o3d
import csv
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from C_SRS_fixedEnd import C_SRS_fixedEnd, IK_MLP
import pickle
from camera_driver_fixedEnd import FixedEndCamera

def take_cantilever_bending_pts(camera: FixedEndCamera) -> np.ndarray:
    """
    Take points along the cantilever bending surface.

    Args:
        camera (FixedEndCamera): The FixedEndCamera object.

    Returns:
        np.ndarray: Array of shape (num_points, 3) containing the sampled points.
    """
    # Define the range of x values along the cantilever
    filtered_region = [0.02, 0.265, 0, 0.13, -0.05, 0.02]
    rgb, depth = camera.read_rgb_depth()
    pcd = camera.get_depth_pointcloud(region=filtered_region)
    
    # pcd, geometries = camera.draw_points_depth(region=filtered_region)
    # return the filtered points as a numpy array
    points = np.asarray(pcd.points)
    # visualize the points using pyvista
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    o3d.visualization.draw_geometries([pcd, axes], window_name="Depth point cloud with coordinate frame")
    return points




if __name__ == "__main__":
    camera = FixedEndCamera(fps=15)
    time.sleep(3.0)  # Allow the camera to warm up
    points = take_cantilever_bending_pts(camera)
    print("type of points:", type(points))
    print("shape of points:", points.shape)
    # Save the points to a pickle file
    with open("cantilever_gravity_fixedEnd_flying_carpet.pkl", "wb") as f:
        pickle.dump(points, f)
    