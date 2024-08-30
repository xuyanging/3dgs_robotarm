import cv2
import numpy as np
import pybullet as p
from argparse import ArgumentParser
import torch

from libinfer_2dgs.scene import Scene, GaussianModel
from libinfer_2dgs.scene.cameras import MiniCam
from libinfer_2dgs.arguments import ModelParams, PipelineParams, get_combined_args
from libinfer_2dgs.gaussian_renderer import render
from libinfer_2dgs.utils.graphics_utils import getProjectionMatrix



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

def view(dataset, pipe, iteration):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        view_matrix = np.eye(4, dtype=np.float32)
        view_matrix[0:3,:] = adjust_matrix@view_matrix[0:3,:]
        world_view_transform = torch.tensor(view_matrix).cuda()
        projection_matrix_torch = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy).transpose(0, 1).cuda()
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix_torch.unsqueeze(0))).squeeze(0)
        custom_cam = MiniCam(width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform)
        rebot_view_rendering = render(custom_cam, gaussians, pipe, background)["render"]
        rebot_view_img_buffer = rebot_view_rendering.permute(1, 2, 0).cpu().numpy()
        cv2.imwrite('test.jpg', np.array(rebot_view_img_buffer*255, np.uint8))
        pass
    

if __name__ == "__main__":

    # Set up command line argument parser
    parser = ArgumentParser(description="Exporting script parameters")
    lp = ModelParams(parser, sentinel=True)
    pp = PipelineParams(parser)
    parser.add_argument('--iteration', type=int, default=30000)
    args = get_combined_args(parser)
    #args = parser.parse_args(sys.argv[1:])
    print("View: " + args.model_path)
    view(lp.extract(args), pp.extract(args), args.iteration)

    print("\nViewing complete.")