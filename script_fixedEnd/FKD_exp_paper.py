import matplotlib
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
import csv
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from C_SRS_fixedEnd import C_SRS_fixedEnd, IK_MLP
import pickle

def draw_SOFA_compare(c_srs: C_SRS_fixedEnd):
    def get_fb_verts_SOFA(vert_SOFA, feedback_vert_idx):
        fb_verts_SOFA = vert_SOFA[feedback_vert_idx]
        return fb_verts_SOFA

    file_flat = "data/fixedEnd_FKD_exp_data_forpaper_final.pkl"
    with open(file_flat, "rb") as f:
        data = pickle.load(f)
    vert_list = data["vert_list"]
    pts_list = data["pts_list"]
    SOFA_file = "data/FKD_as_SOFA_compare.pickle"
    with open(SOFA_file, "rb") as f:
        data = pickle.load(f)
    tetrahedra = data["tetrahedra"]
    feedback_vert_idx = data["feedback_vert_idx"]
    vert_list_SOFA = data["vert_list"]

    for i in range(len(vert_list)):
        plotter = pv.Plotter()
        vert = vert_list[i]
        gt_pts = pts_list[i]
        vert_SOFA = vert_list_SOFA[i]
        # vert_SOFA[:, 2] -= 0.025
        # vert_SOFA[:, 0] += 0.02
        fb_vertices = c_srs.get_fb_surface(vert)
        fb_verts_SOFA = get_fb_verts_SOFA(vert_SOFA, feedback_vert_idx)
        faces = np.hstack((np.full((c_srs.mesh_triangles.shape[0], 1), 3), c_srs.mesh_triangles))
        mesh_surface = pv.PolyData(vert, faces)
        mesh_SOFA = pv.PolyData(vert_SOFA)
        mesh_SOFA.faces = np.hstack([[4, *tet] for tet in tetrahedra])
        mesh_fb = pv.PolyData(fb_vertices, faces)

        plotter.add_mesh(mesh_surface, color='lightblue', show_edges=True, opacity=0.95, label='Input Surface')
        plotter.add_mesh(mesh_fb, color='lightgrey', show_edges=True, opacity=0.35, label='FB Mid-Surface')
        plotter.add_mesh(mesh_SOFA, color='lightcoral', show_edges=True, opacity=0.5, label='SOFA Surface')
        plotter.add_points(gt_pts, color='green', point_size=10, label='Ground Truth Points')
        plotter.add_points(fb_verts_SOFA, color='red', point_size=10, label='FB points from SOFA')
        plotter.show_grid()
        plotter.show()


def read_VTK(file_path):
    """Read the vertices and tetrahedral connectivity from a VTK mesh.

    Parameters
    ----------
    file_path : str or os.PathLike
        Path to a VTK unstructured-grid file.

    Returns
    -------
    mesh_vtk : numpy.ndarray
        Vertex coordinates with shape ``(n_vertices, 3)``. Coordinates retain
        the units used by the input file.
    tetrahedron : numpy.ndarray
        Zero-based vertex indices with shape ``(n_tetrahedra, 4)``.
    """
    file_path = os.fspath(file_path)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"VTK file does not exist: {file_path}")

    vtk_grid = pv.read(file_path)
    mesh_vtk = np.asarray(vtk_grid.points).copy()
    if mesh_vtk.ndim != 2 or mesh_vtk.shape[1] != 3:
        raise ValueError(
            f"Expected 3-D VTK points, but received an array with shape "
            f"{mesh_vtk.shape} from {file_path}"
        )

    # VTK cell type 10 is a four-node linear tetrahedron.  Using cells_dict
    # avoids assuming that every cell has the same size or type.
    cells_dict = getattr(vtk_grid, "cells_dict", {})
    tetrahedron = cells_dict.get(pv.CellType.TETRA)
    if tetrahedron is None:
        raise ValueError(f"No tetrahedral cells were found in VTK file: {file_path}")

    tetrahedron = np.asarray(tetrahedron, dtype=np.int64).reshape(-1, 4).copy()
    return mesh_vtk, tetrahedron

if __name__ == "__main__":
    cl_list_1 = (np.array([424, 425, 424, 298, 274, 298]) * 1e-3).tolist()
    cl_list_2 = (np.array([416,430 , 436, 302, 270, 286]) * 1e-3).tolist()
    cl_list_3 = (np.array([443, 442, 433, 277,264,292]) * 1e-3).tolist()
    cl_list_4 = (np.array([363,357,363, 333,305,335]) * 1e-3).tolist()
    description_file = "./models/flat_tri_surface/C_SRS_description_bary.pkl"
    c_srs = C_SRS_fixedEnd(description_file)
    print("initial cable length: ", c_srs.initial_cable_length)
    cl_list = [cl_list_2, cl_list_3, cl_list_4]
    for i in range(len(cl_list)):
        cl = cl_list[i]
        Q_list, cable_tension = c_srs.FKD_static_length(c_srs.vertices, cl, show_info=True)
        vert_length = c_srs.q_to_vertices(Q_list[-1])
        c_srs.visualize_vert(vert_length)
    exit(0)


    # draw_SOFA_compare(c_srs)
    exit(0)

    data_file = "data/fixedEnd_FKD_exp_data.pkl"
    with open(data_file, "rb") as f:
        data = pickle.load(f)
    pts_list = data["pts_list"]
    fcl_list = data["fcl_list"]
    cl_lists = [cl_list_2, cl_list_3, cl_list_4]
    vert_list = []
    for i in range(len(cl_lists)):
        cl_list = cl_lists[i]
        pts = pts_list[i]
        # Add your processing code here for each cl_list and pts
        Q_list, cable_tension = c_srs.FKD_static_length(c_srs.vertices, cl_list, show_info=True)
        vert_length = c_srs.q_to_vertices(Q_list[-1])
        vert_list.append(vert_length)
        c_srs.visualize_fb_surface_w_gt(vert_length, pts)

    data_2save = {"original_cl_list": fcl_list, "cl_lists": cl_lists, "vert_list": vert_list, "pts_list": pts_list}
    with open("data/fixedEnd_FKD_exp_data_forpaper_final.pkl", "wb") as f:
        pickle.dump(data_2save, f)
