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
from libinfer_3dgs.cameras import MiniCam
from libinfer_3dgs.scene import Scene
from libinfer_3dgs.utils.graphics_utils import getProjectionMatrix
from libinfer_3dgs.arguments import ModelParams, PipelineParams, get_combined_args
from libinfer_3dgs.utils.system_utils import searchForMaxIteration

from physical_lib.render_utils_3dgs import load_params_from_gs, apply_rotations, \
                                        generate_rotation_matrices, transform2origin, shift2center111, convert_SH, \
                                        initialize_resterize,  initialize_resterize_depth, particle_position_tensor_to_ply, \
                                        apply_inverse_rotations, undoshift2center111, undotransform2origin, \
                                        apply_cov_rotations, apply_inverse_cov_rotations

from physical_lib.vis import visualize_pointcloud_with_open3d, rotate_pointcloud_around_box_center, get_rotate_matrices, compute_view_matrix, create_view_metrix,\
                             denoise_3d_point_cloud, statistical_outlier_removal, visualize_3d_point_cloud, visualize_pointcloud_with_open3d_boxes8  
from physical_lib.load import load_json_recursive,  point_cloud_filter                           


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
    parser.add_argument('--config_file', default='scene_file/phone.json')
    args = get_combined_args(parser)

    
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


    print("View: " + args.model_path)
    pipeline.compute_cov3D_python = True
    view_matrix = np.array(p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=view_point[0:3], distance=1e-5, yaw=view_point[3], pitch=view_point[4], roll=view_point[5], upAxisIndex=2), dtype=np.float32).reshape(4,4)
    
    view_matrix[0:3,:] = adjust_matrix@view_matrix[0:3,:]
    world_view_transform = torch.tensor(view_matrix, dtype=torch.float32).cuda()
    projection_matrix_torch = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy).transpose(0, 1).cuda()
    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix_torch.unsqueeze(0))).squeeze(0)
    current_camera = MiniCam(width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform)
 
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    gaussians = load_checkpoint(args.model_path, with_depth=False)
    params = load_params_from_gs(gaussians, pipeline)
    #gaussians_mask = gaussians.get_mask
    #rasterizer = initialize_resterize_depth(current_camera, gaussians, pipeline, background)
    rasterizer = initialize_resterize(current_camera, gaussians, pipeline, background)
    
    init_pos = params["pos"]
    init_cov = params["cov3D_precomp"]
    init_screen_points = params["screen_points"]
    init_opacity = params["opacity"]
    init_shs = params["shs"]

    # throw away low opacity kernels
    
    # mask = init_opacity[:, 0] > opacity_threshold
    # init_pos = init_pos[mask, :]
    # init_cov = init_cov[mask, :]
    # init_opacity = init_opacity[mask, :]
    # init_screen_points = init_screen_points[mask, :]
    # init_shs = init_shs[mask, :]
    
    rotation_matrices = [torch.Tensor(adjust_matrix).cuda().inverse()]
    
    x_min,x_max,y_min,y_max,z_min,z_max = sim_area
    
    vertices = torch.tensor([
    [x_min, y_min, z_min],
    [x_max, y_min, z_min],
    [x_min, y_max, z_min],
    [x_max, y_max, z_min],
    [x_min, y_min, z_max],
    [x_max, y_min, z_max],
    [x_min, y_max, z_max],
    [x_max, y_max, z_max]]).cuda()
    
    
    
    # sim_area_inverse = apply_inverse_rotations(vertices, rotation_matrices)
    # visualize_pointcloud_with_open3d_boxes8(sim_area_inverse.clone().cpu().numpy(), init_pos.clone().detach().cpu().numpy())
    rotated_pos = apply_rotations(init_pos, rotation_matrices)
    
    #visualize_pointcloud_with_open3d(sim_area, rotated_pos.clone().detach().cpu().numpy())
    
    if sim_area is not []:
        boundary = sim_area
        assert len(boundary) == 6
        mask = torch.ones(rotated_pos.shape[0], dtype=torch.bool).to(device="cuda")
        #mask_obj =  torch.load('vast2.pt').cuda()
        for i in range(3):
            mask = torch.logical_and(mask, rotated_pos[:, i] > boundary[2 * i])
            mask = torch.logical_and(mask, rotated_pos[:, i] < boundary[2 * i + 1])

        #mask = torch.logical_and(mask, mask_obj)

        
        unselected_pos = init_pos[~mask, :]
        unselected_cov = init_cov[~mask, :]
        unselected_opacity = init_opacity[~mask, :]
        unselected_shs = init_shs[~mask, :]


        selected_rotated_pos_ori = init_pos[mask, :]
        selected_init_cov = init_cov[mask, :]
        selected_init_opacity = init_opacity[mask, :]
        selected_init_shs = init_shs[mask, :]


    gaussians.save_ply('crop.ply', mask)

    if use_cluster:
        
        obj_mask = point_cloud_filter(selected_rotated_pos_ori, eps=0.06, min_points=50, min_cluster_size=30)
        inliers = selected_rotated_pos_ori[obj_mask]
        outliers = selected_rotated_pos_ori[~obj_mask]
        #visualize_3d_point_cloud(inliers.detach().cpu().numpy(), outliers.detach().cpu().numpy())
        selected_rotated_pos_ori = selected_rotated_pos_ori[obj_mask,:]
        selected_init_cov = selected_init_cov[obj_mask,:]
        selected_init_opacity = selected_init_opacity[obj_mask,:]
        selected_init_shs=selected_init_shs[obj_mask,:]
   
    rotated_pos = apply_rotations(selected_rotated_pos_ori, rotation_matrices)
    rotated_cov = apply_cov_rotations(selected_init_cov, rotation_matrices)
    
    num_frames = 36  
    images_all = []
    depth_all = []
    
    x_delta = 0
    y_delta = 0
    z_delta = 0
    for i in tqdm(range(num_frames)):
        angle = i * 2 * np.pi / num_frames
        #angle = 0 
        #y_delta += i*0.003
        #x_delta -= i*0.001
        delta = torch.Tensor([x_delta,y_delta,z_delta]).cuda()
        selected_rotated_pos = rotate_pointcloud_around_box_center(sim_area, rotated_pos, angle, delta, axis='z')
         
        
        # transformed_pos, scale_origin, original_mean_pos = transform2origin(rotated_pos)
        # transformed_pos = shift2center111(transformed_pos)
               
        #visualize_pointcloud_with_open3d(sim_area, rotated_pos.clone().detach().cpu().numpy())
        if False:
            if not os.path.exists("./log"):
                os.makedirs("./log")
            particle_position_tensor_to_ply(
                rotated_pos,
                "./log/transformed_particles.ply",
            )
                
        # pos = unselected_pos
        # cov3D = unselected_cov
        # opacity = unselected_opacity
        # shs = unselected_shs
        
        
        # selected_rotated_pos = apply_inverse_rotations(
        #         undotransform2origin(
        #                 undoshift2center111(transformed_pos), scale_origin, original_mean_pos
        #             ),
        #             rotation_matrices,
        #         )
        #selected_init_cov = selected_init_cov / (scale_origin * scale_origin)
        #selected_init_cov = apply_inverse_cov_rotations(selected_init_cov, [rotation_matrices[0]])
        
        #visualize_pointcloud_with_open3d(sim_area, transformed_pos.clone().detach().cpu().numpy())
        
        # pos = rotated_pos
        # cov3D = init_cov
        # opacity = init_opacity
        # shs = init_shs
        
        R = get_rotate_matrices(angle, axis='z')
        selected_rotated_cov = apply_cov_rotations(rotated_cov, [R])
        selected_init_cov = apply_inverse_cov_rotations(selected_rotated_cov, rotation_matrices)
        selected_rotated_pos = apply_inverse_rotations(selected_rotated_pos, rotation_matrices)
        
        
        # pos = torch.cat([selected_rotated_pos, unselected_pos], dim=0)
        # cov3D = torch.cat([selected_init_cov, unselected_cov], dim=0)
        # opacity = torch.cat([selected_init_opacity, unselected_opacity], dim=0)
        # shs = torch.cat([selected_init_shs, unselected_shs], dim=0)
        
        pos = selected_rotated_pos
        cov3D = selected_init_cov
        opacity = selected_init_opacity
        shs = selected_init_shs
        
        # pos = unselected_pos
        # cov3D = unselected_cov
        # opacity = unselected_opacity
        # shs = unselected_shs
        
        colors_precomp = convert_SH(shs, current_camera, gaussians, pos, None)
        frame = 0
        rendering, raddi = rasterizer(
            means3D=pos,
            means2D=init_screen_points,
            shs=None,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=None,
            rotations=None,
            cov3D_precomp=cov3D)
        
        # rendering, rendered_mask, rendered_depth, radii = rasterizer(
        # means3D = pos,
        # means2D = init_screen_points,
        # shs = None,
        # colors_precomp = colors_precomp,
        # opacities = opacity,
        # mask = gaussians_mask,
        # scales = None,
        # rotations = None,
        # cov3D_precomp = cov3D)
        
        cv2_img = np.clip(rendering.permute(1, 2, 0).detach().cpu().numpy(),0,1)
        #depth_map = cv2.convertScaleAbs(rendered_depth[0].detach().cpu().numpy(), alpha=255.0 / rendered_depth.max().detach().cpu().numpy())
        #cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        #cv2.imwrite(f"{frame}.png".rjust(8, "0"), cv2_img*255)
        images_all.append(np.array(cv2_img*255, np.uint8))
        #depth_all.append(np.array(depth_map, np.uint8))
    #imageio.mimsave('move_animation_chair_depth.gif', depth_all, loop=True)    
    imageio.mimsave('move_animation_chair.gif', images_all, loop=True)
    #cv2.imwrite(f"{frame}.png".rjust(8, "0"), 255 * cv2_img)
    print("\nViewing complete.")