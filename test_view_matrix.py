import numpy as np
import torch
import pybullet as p
from plyfile import PlyData, PlyElement
from physical_lib.vis import create_view_metrix
from physical_lib.vis import visualize_pointcloud_with_open3d
from physical_lib.render_utils_3dgs import apply_rotations
from libinfer_3dgs.utils.graphics_utils import getProjectionMatrix

adjust_matrix = np.array([
    [ 0.99679286,  0.03824596,  0.07029395],
    [-0.08002495,  0.47639272,  0.87558323],
    [ 0        , -0.87840038,  0.47792549]
  ])

 
#view_matrix = create_view_metrix(0,0,0,[0.6, -0.1, 1.235])
sim_area = [-0.3, 0.7, -0.6, 0.4, 0.65, 1.82]


view_matrix = np.array(p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.6, 10, 2.235], distance=1e-5, yaw=0, pitch=0, roll=0, upAxisIndex=2), dtype=np.float32).reshape(4,4)
#view_matrix[0:3,:] = adjust_matrix@view_matrix[0:3,:]

world_view_transform = torch.tensor(view_matrix, dtype=torch.float32).cuda()
projection_matrix_torch = getProjectionMatrix(znear=0.01, zfar=100, fovX=1.6639937226014894, fovY=1.1064156765004665).transpose(0, 1).cuda()
full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix_torch.unsqueeze(0))).squeeze(0)

plydata = PlyData.read('model_file/3dgs/garden/point_cloud/iteration_30000/point_cloud.ply')
xyz = torch.Tensor(np.stack((np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"])),  axis=1)).cuda()


rotation_matrices = [torch.Tensor(adjust_matrix).cuda().inverse()]
rotated_pos = apply_rotations(xyz, rotation_matrices)

visualize_pointcloud_with_open3d(sim_area, rotated_pos.cpu().numpy(), view_matrix=full_proj_transform.cpu().numpy())