import os
import imageio
import cv2
import json
import numpy as np
import pybullet as p
from tqdm import tqdm
from argparse import ArgumentParser
import torch
from libinfer_3dgs import render
from libinfer_3dgs.gaussian_model import GaussianModel
from libinfer_3dgs.gaussian_model_depth import GaussianModelDepth
from libinfer_3dgs.cameras import MiniCam, Camera
from libinfer_3dgs.scene import Scene
from libinfer_3dgs.utils.graphics_utils import getProjectionMatrix
from libinfer_3dgs.utils.camera_utils import cameraList_from_camInfos
from libinfer_3dgs.arguments import ModelParams, PipelineParams, get_combined_args
from libinfer_3dgs.utils.system_utils import searchForMaxIteration
from libinfer_3dgs.dataset_readers import readColmapSceneInfo

from physical_lib.render_utils_3dgs import load_params_from_gs, apply_rotations, \
                                        generate_rotation_matrices, transform2origin, shift2center111, convert_SH, \
                                        initialize_resterize,  initialize_resterize_depth, particle_position_tensor_to_ply, \
                                        apply_inverse_rotations, undoshift2center111, undotransform2origin, \
                                        apply_cov_rotations, apply_inverse_cov_rotations

from physical_lib.vis import visualize_pointcloud_with_open3d, rotate_pointcloud_around_box_center, get_rotate_matrices, compute_view_matrix, create_view_metrix,\
                             denoise_3d_point_cloud, statistical_outlier_removal, visualize_3d_point_cloud, visualize_pointcloud_with_open3d_boxes8, visualize_points_in_3D  
from physical_lib.load import load_json_recursive,  point_cloud_filter                           

def generate_grid_index(depth):
    h, w = depth.shape
    grid = torch.meshgrid([torch.arange(h), torch.arange(w)])
    grid = torch.stack(grid, dim=-1)
    return grid

def load_checkpoint(model_path, sh_degree=3, iteration=-1, with_depth=False):
    checkpt_dir = os.path.join(model_path, "point_cloud")
    if iteration == -1:
        iteration = searchForMaxIteration(checkpt_dir)
    checkpt_path = os.path.join(
        checkpt_dir, f"iteration_{iteration}", "point_cloud.ply"
    )
    if with_depth:
        gaussians = GaussianModelDepth(sh_degree)
    else:
        gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(checkpt_path)
    return gaussians

if __name__ == "__main__":

    parser = ArgumentParser(description="Exporting script parameters")
    lp = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument('--iteration', type=int, default=30000)
    parser.add_argument('--config_file', default='scene_file/6e03a0a9-4.json')
    args = get_combined_args(parser)
    pipeline.compute_cov3D_python = True
    
    config = load_json_recursive(args.config_file)
    adjust_matrix = np.array(config["adjust_matrix"])
    sim_area = config["sim_area"]
    fovy = config["fovy"]
    fovx = config["fovx"]
    height = config["height"]
    width = config["width"]
    zfar = config["zfar"]
    znear = config["znear"]
    view_point = config['view_point']
    opacity_threshold = config["opacity_threshold"]
    use_cluster = config['use_cluster']
    
    scene_info = readColmapSceneInfo(args.source_path, 'images', eval=False)
    camera_list = scene_info.train_cameras
    #with open(os.path.join(args.model_path, "cameras.json"), 'r', encoding='utf-8') as file:
    #    json_cams = json.load(file)
    train_cameras = cameraList_from_camInfos(scene_info.train_cameras, 1, args)
    
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    gaussians = load_checkpoint(args.model_path, with_depth=True)
    params = load_params_from_gs(gaussians, pipeline)
    gaussians_mask = gaussians.get_mask
    
    
    init_pos = params["pos"]
    init_cov = params["cov3D_precomp"]
    init_screen_points = params["screen_points"]
    init_opacity = params["opacity"]
    init_shs = params["shs"]
    
    for cam_info in tqdm(train_cameras):
        rasterizer = initialize_resterize_depth(cam_info, gaussians, pipeline, background)
        

        # throw away low opacity kernels
        mask = init_opacity[:, 0] > opacity_threshold
        init_pos = init_pos[mask, :]
        init_cov = init_cov[mask, :]
        init_opacity = init_opacity[mask, :]
        init_screen_points = init_screen_points[mask, :]
        init_shs = init_shs[mask, :]
        
        colors_precomp = convert_SH(init_shs, cam_info, gaussians, init_pos, None)
        rendering, rendered_mask, rendered_depth, radii = rasterizer(
        means3D = init_pos,
        means2D = init_screen_points,
        shs = None,
        colors_precomp = colors_precomp,
        opacities = init_opacity,
        mask = gaussians_mask,
        scales = None,
        rotations = None,
        cov3D_precomp = init_cov)
        
        depth = rendered_depth.detach().cpu().squeeze()
        grid_index = generate_grid_index(depth)
        points_in_3D = torch.zeros(depth.shape[0], depth.shape[1], 3).cpu()
        points_in_3D[:,:,-1] = depth

        # caluculate cx cy fx fy with FoVx FoVy
        cx = depth.shape[1] / 2
        cy = depth.shape[0] / 2
        fx = cx / np.tan(fovx / 2)
        fy = cy / np.tan(fovy / 2)
        points_in_3D[:,:,0] = (grid_index[:,:,0] - cx) * depth / fx
        points_in_3D[:,:,1] = (grid_index[:,:,1] - cy) * depth / fy
        
        visualize_points_in_3D(points_in_3D)
        #depth_map = cv2.convertScaleAbs(rendered_depth[0].detach().cpu().numpy(), alpha=255.0 / rendered_depth.max().detach().cpu().numpy())   
        #cv2.imwrite(os.path.join('/home/xuyang/MaskClustering/data/colmap/drjohnson/depth',f'{cam_info.image_name}.png'),depth_map)
    print("\nViewing complete.")