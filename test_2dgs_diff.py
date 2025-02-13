import os
import imageio
import cv2
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
import torch
from libinfer_3dgs import render
# from libinfer_3dgs.gaussian_model import GaussianModel
# from libinfer_3dgs.cameras import MiniCam
# from libinfer_3dgs.scene import Scene
# from libinfer_3dgs.utils.graphics_utils import getProjectionMatrix
# from libinfer_3dgs.arguments import ModelParams, PipelineParams, get_combined_args
# from libinfer_3dgs.utils.system_utils import searchForMaxIteration


from libinfer_2dgs.scene import Scene, GaussianModel
from libinfer_2dgs.scene.cameras import MiniCam
from libinfer_2dgs.arguments import ModelParams, PipelineParams, get_combined_args
from libinfer_2dgs.gaussian_renderer import render
from libinfer_2dgs.utils.graphics_utils import getProjectionMatrix
from libinfer_2dgs.utils.system_utils import searchForMaxIteration

from physical_lib.render_utils_2dgs import load_params_from_gs, apply_rotations, \
                                        generate_rotation_matrices, transform2origin, shift2center111, convert_SH, \
                                        initialize_resterize, particle_position_tensor_to_ply, \
                                        apply_inverse_rotations, undoshift2center111, undotransform2origin, \
                                        apply_cov_rotations, apply_inverse_cov_rotations

from physical_lib.vis import visualize_pointcloud_with_open3d, rotate_pointcloud_around_box_center,get_rotate_matrices





fovy = 1.1064156765004665
fovx = 1.6639937226014894
height = 540
width = 960
zfar = 100.0
znear = 0.01

adjust_matrix = np.array([
            [-0.01396038, -0.51966003, -0.85425907],
            [ 0.99977363,  0.00646452, -0.02027087],
            [ 0.01605634, -0.85434868,  0.51945214],
    ])

# adjust_matrix = np.array([
#             [1, 0, 0],
#             [0, 1, 0],
#             [0, 0, 1],
#     ])

opacity_threshold = 0.08
#opacity_threshold = 0
sim_area = [-1, 1, -4, -2, 0.4, 1.45]
#sim_area = None

def create_view_metrix(yaw, pitch, roll):
    yaw, pitch, roll = np.radians(float(yaw)), np.radians(float(pitch)), np.radians(360-float(roll))
    translate=np.array([.0, .0, .0])
    scale=1.0
    Rt = np.zeros((4, 4))
    t = np.array([0, 0, 0])
    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    R = np.dot(R_z, np.dot(R_y, R_x))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def load_checkpoint(model_path, sh_degree=3, iteration=-1):
    checkpt_dir = os.path.join(model_path, "point_cloud")
    if iteration == -1:
        iteration = searchForMaxIteration(checkpt_dir)
    checkpt_path = os.path.join(
        checkpt_dir, f"iteration_{iteration}", "point_cloud.ply"
    )
    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(checkpt_path)
    return gaussians

if __name__ == "__main__":

    parser = ArgumentParser(description="Exporting script parameters")
    lp = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument('--iteration', type=int, default=30000)
    args = get_combined_args(parser)

    print("View: " + args.model_path)
    pipeline.compute_cov3D_python = True
    view_matrix = create_view_metrix(0,0,90)
    view_matrix[0:3,:] = adjust_matrix@view_matrix[0:3,:]
    world_view_transform = torch.tensor(view_matrix).cuda()
    projection_matrix_torch = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy).transpose(0, 1).cuda()
    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix_torch.unsqueeze(0))).squeeze(0)
    current_camera = MiniCam(width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform)
 
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    gaussians = load_checkpoint(args.model_path)
    params = load_params_from_gs(gaussians, pipeline)
    
    rasterize = initialize_resterize(current_camera, gaussians, pipeline, background)
    
    
    
    init_pos = params["pos"]
    init_cov = params["cov3D_precomp"]
    init_screen_points = params["screen_points"]
    init_opacity = params["opacity"]
    init_shs = params["shs"]

    #init_cov = get_covariance()
    W, H = width, height
    near, far = znear, zfar
    ndc2pix = torch.tensor([
        [W / 2, 0, 0, (W-1) / 2],
        [0, H / 2, 0, (H-1) / 2],
        [0, 0, far-near, near],
        [0, 0, 0, 1]]).float().cuda().T
    world2pix =  full_proj_transform @ ndc2pix
    

    mask = init_opacity[:, 0] > opacity_threshold
    init_pos = init_pos[mask, :]
    init_cov = init_cov[mask, :]
    init_opacity = init_opacity[mask, :]
    init_screen_points = init_screen_points[mask, :]
    init_shs = init_shs[mask, :]

    
    #rotation_matrices = generate_rotation_matrices(torch.tensor([0]), [0])
    rotation_matrices = [torch.Tensor(adjust_matrix).cuda().inverse()]
    rotated_pos = apply_rotations(init_pos, rotation_matrices)
    
    #visualize_pointcloud_with_open3d(sim_area, rotated_pos.clone().detach().cpu().numpy())
    if sim_area is not None:
        boundary = sim_area
        assert len(boundary) == 6
        mask = torch.ones(rotated_pos.shape[0], dtype=torch.bool).to(device="cuda")
        for i in range(3):
            mask = torch.logical_and(mask, rotated_pos[:, i] > boundary[2 * i])
            mask = torch.logical_and(mask, rotated_pos[:, i] < boundary[2 * i + 1])

        unselected_pos = init_pos[~mask, :]
        unselected_cov = init_cov[~mask, :]
        unselected_opacity = init_opacity[~mask, :]
        unselected_shs = init_shs[~mask, :]

        selected_rotated_pos_ori = init_pos[mask, :]
        selected_init_cov = init_cov[mask, :]
        selected_init_opacity = init_opacity[mask, :]
        selected_init_shs = init_shs[mask, :]

    rotated_pos = apply_rotations(selected_rotated_pos_ori, rotation_matrices)
    #rotated_cov = apply_cov_rotations(selected_init_cov, rotation_matrices)
    
    num_frames = 36  # 总帧数
    images_all = []
    x_delta = 0
    y_delta = 0
    z_delta = 0
    
    
    for i in tqdm(range(num_frames)):
        angle = i * 2 * np.pi / num_frames
        delta = torch.Tensor([x_delta,y_delta,z_delta]).cuda()
        selected_rotated_pos = rotate_pointcloud_around_box_center(sim_area, rotated_pos, angle, delta, axis='z')
        
        #transformed_pos, scale_origin, original_mean_pos = transform2origin(rotated_pos)
        #transformed_pos = shift2center111(transformed_pos)
        # init_cov = apply_cov_rotations(init_cov, rotation_matrices)
        # init_cov = scale_origin * scale_origin * init_cov
        R = get_rotate_matrices(angle, axis='z')
        
        
        
        #selected_rotated_cov = apply_cov_rotations(rotated_cov, [R]) 
        #selected_init_cov = apply_inverse_cov_rotations(selected_rotated_cov, rotation_matrices)
        selected_rotated_pos = apply_inverse_rotations(selected_rotated_pos, rotation_matrices)
        #visualize_pointcloud_with_open3d(sim_area, rotated_pos.clone().detach().cpu().numpy())
        if False:
            if not os.path.exists("./log"):
                os.makedirs("./log")
            particle_position_tensor_to_ply(
                rotated_pos,
                "./log/transformed_particles.ply",
            )
                
        
        
        
        pos = torch.cat([selected_rotated_pos, unselected_pos], dim=0)
        cov3D = torch.cat([selected_init_cov, unselected_cov], dim=0)
        cov3D = (cov3D[:, [0,1,3]] @ world2pix[:,[0,1,3]]).permute(0,2,1).reshape(-1, 9)
        opacity = torch.cat([selected_init_opacity, unselected_opacity], dim=0)
        shs = torch.cat([selected_init_shs, unselected_shs], dim=0)  
        
        colors_precomp = convert_SH(shs, current_camera, gaussians, pos, None)
        
        rendering, raddi, allmap = rasterize(
            means3D=pos,
            means2D=init_screen_points,
            shs=None,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=None,
            rotations=None,
            cov3D_precomp=cov3D,
                )
        
        cv2_img = rendering.permute(1, 2, 0).detach().cpu().numpy()
        images_all.append(np.array(cv2_img*255, np.uint8))
        cv2.imwrite(f"{i}.png".rjust(8, "0"), 255 * cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))    
    imageio.mimsave('rotation_animation.gif', images_all, loop=True)
    #cv2.imwrite(f"{frame}.png".rjust(8, "0"), 255 * cv2_img)
    print("\nViewing complete.")