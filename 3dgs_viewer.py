import threading
import time
import math
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import dearpygui.dearpygui as dpg
import pybullet as p
import random
from argparse import ArgumentParser


is_2dgs = False

if is_2dgs:
    from libinfer_2dgs.scene import Scene, GaussianModel
    from libinfer_2dgs.scene.cameras import MiniCam
    from libinfer_2dgs.arguments import ModelParams, PipelineParams, get_combined_args
    from libinfer_2dgs.gaussian_renderer import render
    from libinfer_2dgs.utils.graphics_utils import getProjectionMatrix
else:
    from libinfer_3dgs import render
    from libinfer_3dgs.gaussian_model import GaussianModel
    from libinfer_3dgs.cameras import MiniCam
    from libinfer_3dgs.scene import Scene
    from libinfer_3dgs.utils.graphics_utils import getProjectionMatrix
    from libinfer_3dgs.arguments import ModelParams, PipelineParams, get_combined_args

def compute_projection_matrix_fov(fovy, fovx, near_val, far_val):
    y_scale = 1.0 / np.tan(fovy / 2)
    x_scale = 1.0 / np.tan(fovx / 2)

    projection_matrix = np.zeros((4, 4), dtype=np.float32)
    projection_matrix[0, 0] = x_scale
    projection_matrix[1, 1] = y_scale
    projection_matrix[2, 2] = (near_val + far_val) / (near_val - far_val)
    projection_matrix[2, 3] = -1
    projection_matrix[3, 2] = (2 * far_val * near_val) / (near_val - far_val)
    return projection_matrix

class KeyInputHandler:
    def __init__(self):
        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.lock = threading.Lock()

    def set_euler_angles(self, yaw, pitch, roll):
        with self.lock:
            self.yaw = yaw
            self.pitch = pitch
            self.roll = roll

    def get_euler_angles(self):
        with self.lock:
            return self.yaw, self.pitch, self.roll

    def set_position(self, x, y, z):
        with self.lock:
            self.x = x
            self.y = y
            self.z = z

    def get_position(self):
        with self.lock:
            return self.x, self.y, self.z


def main():
    
    key_input_handler = KeyInputHandler()
          
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
   
    projection_matrix_torch = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy).transpose(0, 1).cuda()

    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))

    dataset = model.extract(args)
    iteration = args.iteration
    pipeline = pipeline.extract(args)

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    prev_time = time.time()
    fps = 0
    frame_count = 0
    
    
    def update_image():
       
        nonlocal prev_time, fps, frame_count
        with torch.no_grad():
            yaw, pitch, roll = key_input_handler.get_euler_angles()
            x, y, z = key_input_handler.get_position()
            view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[x, y, z], distance=1e-5, yaw=yaw, pitch=pitch, roll=roll, upAxisIndex=2)
            view_matrix = np.array(view_matrix, np.float32).reshape(4,4)
            view_matrix[0:3,:] = adjust_matrix@view_matrix[0:3,:]
            world_view_transform = torch.tensor(view_matrix).cuda()
            full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix_torch.unsqueeze(0))).squeeze(0)
            custom_cam = MiniCam(width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform)
            rendering = render(custom_cam, gaussians, pipeline, background)["render"]
            img_buffer = rendering.permute(1, 2, 0).cpu().numpy()            
            dpg.set_value("rendered_image", img_buffer.reshape(-1))
            dpg.set_value("euler_angles_text", f'FPS: {fps:.2f}')
            
            frame_count += 1
            current_time = time.time()
            elapsed_time = current_time - prev_time
            if elapsed_time >= 1.0:
                fps = frame_count / elapsed_time
                prev_time = current_time
                frame_count = 0


    def key_callback(sender, app_data):
        if sender == "yaw_slider":
            key_input_handler.set_euler_angles(app_data, key_input_handler.pitch, key_input_handler.roll)
        elif sender == "pitch_slider":
            key_input_handler.set_euler_angles(key_input_handler.yaw, app_data, key_input_handler.roll)
        elif sender == "roll_slider":
            key_input_handler.set_euler_angles(key_input_handler.yaw, key_input_handler.pitch, app_data)
        elif sender == "x_slider":
            key_input_handler.set_position(app_data, key_input_handler.y, key_input_handler.z)
        elif sender == "y_slider":
            key_input_handler.set_position(key_input_handler.x, app_data, key_input_handler.z)
        elif sender == "z_slider":
            key_input_handler.set_position(key_input_handler.x, key_input_handler.y, app_data)

    dpg.create_context()

    with dpg.texture_registry(show=False):
        dpg.add_raw_texture(width, height, np.zeros((height, width, 3)), format=dpg.mvFormat_Float_rgb, tag="rendered_image")

    with dpg.window(label="Main Window", width=1800, height=1200):
        with dpg.group(horizontal=False):
            with dpg.child_window(width=width, height=height*2):
                dpg.add_image("rendered_image")
            with dpg.child_window(width=600, height=300, pos=(width+10,10)):
                dpg.add_text("FPS: 0.00 yaw: pitch: roll:", tag="euler_angles_text")
                dpg.add_slider_float(label="Yaw", default_value=0, min_value=-180, max_value=180, tag="yaw_slider", callback=key_callback)
                dpg.add_slider_float(label="Pitch", default_value=0, min_value=-180, max_value=180, tag="pitch_slider", callback=key_callback)
                dpg.add_slider_float(label="Roll", default_value=0, min_value=-180, max_value=180, tag="roll_slider", callback=key_callback)
                dpg.add_slider_float(label="X", default_value=0, min_value=-10, max_value=10, tag="x_slider", callback=key_callback)
                dpg.add_slider_float(label="Y", default_value=0, min_value=-10, max_value=10, tag="y_slider", callback=key_callback)
                dpg.add_slider_float(label="Z", default_value=0, min_value=-10, max_value=10, tag="z_slider", callback=key_callback)
    dpg.create_viewport(title='Render Viewer', width=width, height=height)
    dpg.setup_dearpygui()
    dpg.show_viewport()

    while dpg.is_dearpygui_running():
        update_image()
        dpg.render_dearpygui_frame()

    dpg.destroy_context()
    p.disconnect()

if __name__ == "__main__":
    main()
