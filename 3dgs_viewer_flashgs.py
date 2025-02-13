import os
import threading
import time
import json
import math
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import dearpygui.dearpygui as dpg
import pybullet as p
import random
from argparse import ArgumentParser
import flash_gaussian_splatting
import cv2

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


class Scene:
    def __init__(self, device):
        self.device = device
        self.num_vertex = 0
        self.position = None
        self.shs = None
        self.opacity = None
        self.cov3d = None

    def loadPly(self, scene_path):
        self.num_vertex, self.position, self.shs, self.opacity, self.cov3d = flash_gaussian_splatting.ops.loadPly(
            scene_path)
        print("num_vertex = %d" % self.num_vertex)
        # 58*4byte
        self.position = self.position.to(self.device)  # 3
        self.shs = self.shs.to(self.device)  # 48
        self.opacity = self.opacity.to(self.device)  # 1
        self.cov3d = self.cov3d.to(self.device)  # 6

class Rasterizer:
    # 构造函数中分配内存
    def __init__(self, scene, MAX_NUM_RENDERED, MAX_NUM_TILES):
        # 24 bytes
        self.gaussian_keys_unsorted = torch.zeros(MAX_NUM_RENDERED, device=scene.device, dtype=torch.int64)
        self.gaussian_values_unsorted = torch.zeros(MAX_NUM_RENDERED, device=scene.device, dtype=torch.int32)
        self.gaussian_keys_sorted = torch.zeros(MAX_NUM_RENDERED, device=scene.device, dtype=torch.int64)
        self.gaussian_values_sorted = torch.zeros(MAX_NUM_RENDERED, device=scene.device, dtype=torch.int32)

        self.MAX_NUM_RENDERED = MAX_NUM_RENDERED
        self.MAX_NUM_TILES = MAX_NUM_TILES
        self.SORT_BUFFER_SIZE = flash_gaussian_splatting.ops.get_sort_buffer_size(MAX_NUM_RENDERED)
        self.list_sorting_space = torch.zeros(self.SORT_BUFFER_SIZE, device=scene.device, dtype=torch.int8)
        self.ranges = torch.zeros((MAX_NUM_TILES, 2), device=scene.device, dtype=torch.int32)
        self.curr_offset = torch.zeros(1, device=scene.device, dtype=torch.int32)

        # 40 bytes
        self.points_xy = torch.zeros((scene.num_vertex, 2), device=scene.device, dtype=torch.float32)
        self.rgb_depth = torch.zeros((scene.num_vertex, 4), device=scene.device, dtype=torch.float32)
        self.conic_opacity = torch.zeros((scene.num_vertex, 4), device=scene.device, dtype=torch.float32)

    # 前向传播（应用层封装）
    def forward(self, scene, camera, bg_color):
        # 属性预处理 + 键值绑定
        self.curr_offset.fill_(0)
        flash_gaussian_splatting.ops.preprocess(scene.position, scene.shs, scene.opacity, scene.cov3d,
                                                camera.width, camera.height, 16, 16,
                                                camera.position, camera.rotation,
                                                camera.focal_x, camera.focal_y, camera.zFar, camera.zNear,
                                                self.points_xy, self.rgb_depth, self.conic_opacity,
                                                self.gaussian_keys_unsorted, self.gaussian_values_unsorted,
                                                self.curr_offset)

        # 键值对数量判断 + 处理键值对过多的异常情况
        num_rendered = int(self.curr_offset.cpu()[0])
        # print(num_rendered)
        if num_rendered >= self.MAX_NUM_RENDERED:
            raise "Too many k-v pairs!"

        flash_gaussian_splatting.ops.sort_gaussian(num_rendered, camera.width, camera.height, 16, 16,
                                                   self.list_sorting_space,
                                                   self.gaussian_keys_unsorted, self.gaussian_values_unsorted,
                                                   self.gaussian_keys_sorted, self.gaussian_values_sorted)
        # 排序 + 像素着色 + 混色阶段
        out_color = torch.zeros((camera.height, camera.width, 3), device=scene.device, dtype=torch.int8)
        flash_gaussian_splatting.ops.render_16x16(num_rendered, camera.width, camera.height,
                                                  self.points_xy, self.rgb_depth, self.conic_opacity,
                                                  self.gaussian_keys_sorted, self.gaussian_values_sorted,
                                                  self.ranges, bg_color, out_color)
        return out_color


class Camera:
    def __init__(self, width, height, focal_x, focal_y, zFar, zNear, position, rotation):
        self.width = width
        self.height = height
        self.position = torch.tensor(position)
        self.rotation = torch.tensor(rotation)
        self.focal_x = focal_x
        self.focal_y = focal_y
        self.zFar = zFar
        self.zNear = zNear

def main():
    
    key_input_handler = KeyInputHandler()
          
    fovy = 1.1064156765004665
    fovx = 1.6639937226014894
    height = 540
    width = 960
    zfar = 100.0
    znear = 0.01
    
    focal_x = width / (2.0*np.tan(fovx / 2))
    focal_y = height / (2.0*np.tan(fovy / 2))


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
    
    scene_path = os.path.join(args.model_path, "point_cloud", "iteration_30000", "point_cloud.ply")
    print(scene_path)
    camera_path = os.path.join(args.model_path, "cameras.json")
    print(camera_path)


    device = torch.device('cuda:0')
    bg_color = torch.zeros(3, dtype=torch.float32)  # black
    scene = Scene(device)
    scene.loadPly(scene_path)

    MAX_NUM_RENDERED = 2 ** 27
    MAX_NUM_TILES = 2 ** 20
    rasterizer = Rasterizer(scene, MAX_NUM_RENDERED, MAX_NUM_TILES)

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
            camera = Camera(width, height, focal_x, focal_y, zfar, znear, np.array([x,y,z], dtype=np.float32), view_matrix[0:3,0:3])
            #torch.cuda.synchronize()
            image = rasterizer.forward(scene, camera, bg_color)
            #torch.cuda.synchronize()
            img_buffer = np.array(image.cpu().numpy().astype(np.uint8)/255, dtype=np.float32)         
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

if __name__ == "__main__":
    main()
