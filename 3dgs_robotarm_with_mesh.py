import threading
import time
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import dearpygui.dearpygui as dpg
from argparse import ArgumentParser
import pybullet as p
import random
import functools


is_2dgs = True

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



useMaximalCoordinates = 0

last_position = None
last_angles = None

def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper



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
        self.delta_x = 0.0
        self.delta_y = 0.0
        self.delta_z = 0.0
        self.robot_yaw = 0.0
        self.robot_pitch = 0.0
        self.robot_roll = 0.0
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

    def update_deltas(self, delta_x, delta_y, delta_z):
        with self.lock:
            self.delta_x = delta_x
            self.delta_y = delta_y
            self.delta_z = delta_z

    def get_deltas(self):
        with self.lock:
            return self.delta_x, self.delta_y, self.delta_z

    def set_robot_euler_angles(self, yaw, pitch, roll):
        with self.lock:
            self.robot_yaw = yaw
            self.robot_pitch = pitch
            self.robot_roll = roll

    def get_robot_euler_angles(self):
        with self.lock:
            return self.robot_yaw, self.robot_pitch, self.robot_roll
       
def kuka_camera(w, h, view_matrix, proj_matrix):
    projection_matrix = tuple(proj_matrix.reshape(-1))
    view_matrix = tuple(view_matrix.reshape(-1))
    print('start: ', time.time())
    img = p.getCameraImage(w, h, view_matrix, projection_matrix, 
                           renderer=p.ER_BULLET_HARDWARE_OPENGL)
                           #renderer = p.ER_TINY_RENDERER)
    print('end: ', time.time())    
    return img

def main():
    
    adjust_matrix = np.array([
            [-0.01396038, -0.51966003, -0.85425907],
            [ 0.99977363,  0.00646452, -0.02027087],
            [ 0.01605634, -0.85434868,  0.51945214],
    ])
    
    p.connect(p.GUI)
    
    eglPluginId = -1
    import pkgutil
    egl = pkgutil.get_loader('eglRenderer')
    if (egl):
        eglPluginId = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
    else:
        eglPluginId = p.loadPlugin("eglRendererPlugin")
    
    
    
    #p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    #p.setTimeStep(1./50.)
    p.setRealTimeSimulation(1)
    
    monastryId = concaveEnv = p.createCollisionShape(p.GEOM_MESH,
                                                    fileName="env_mesh_model/output_compressed.obj",
                                                    flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
    rotation = R.from_matrix(np.linalg.inv(adjust_matrix))

    euler_angles = rotation.as_euler('xyz', degrees=False)
    euler_angles[1] += np.pi 
    adjusted_rotation = R.from_euler('xyz', euler_angles)
    rotation_quaternion = adjusted_rotation.as_quat()
    rotation_quaternion = [rotation_quaternion[0], rotation_quaternion[1], rotation_quaternion[2], rotation_quaternion[3]]
    p.createMultiBody(0, monastryId, baseOrientation=rotation_quaternion)
    
    
    x_rebot = 0
    y_rebot = 2.9
    z_rebot = -0.65
 
    start_pos = [x_rebot, y_rebot, z_rebot]
    start_orientation = p.getQuaternionFromEuler([0, 0, 0])
    kuka_id = p.loadURDF("rm65/urdf/rm_65.urdf", start_pos, start_orientation, useFixedBase=True)    
    numJoints = p.getNumJoints(kuka_id)
    key_input_handler = KeyInputHandler()
          
    fovy = 1.1064156765004665
    fovx = 1.6639937226014894
    height = 540
    width = 960
    zfar = 100.0
    znear = 0.01
    
    
    projection_matrix_torch = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy).transpose(0, 1).cuda()
    projection_matrix_fov = compute_projection_matrix_fov(fovy=fovy, fovx=fovx, near_val=znear, far_val=zfar)

    global_images = {"3dgs": np.zeros((height,width,3)), "bullet": np.zeros((height,width,3)), "mask": np.zeros((height,width))}

    def generate_ball_event():
        sphereRadius = 0.05
        colSphereId = p.createCollisionShape(p.GEOM_SPHERE, radius=sphereRadius)
        colBoxId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[sphereRadius, sphereRadius, sphereRadius])


        mass = 1
        useMaximalCoordinates = 0
        #visualShapeId = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=sphereRadius, rgbaColor=[0, 1, 1, 1])
        visualShapeId = -1
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    x = -i * 2 * sphereRadius + x_rebot  
                    y = j * 2 * sphereRadius + y_rebot
                    z = k * 2 * sphereRadius + 1.0 

                    if (k & 2):
                        sphereUid = p.createMultiBody(
                            mass,
                            colSphereId,
                            visualShapeId, [x, y, z],
                            useMaximalCoordinates=useMaximalCoordinates)
                    else:
                        sphereUid = p.createMultiBody(
                            mass,
                            colBoxId,
                            visualShapeId, [x, y, z],
                            useMaximalCoordinates=useMaximalCoordinates)
                    p.changeDynamics(sphereUid,
                       -1,
                       spinningFriction=0.001,
                       rollingFriction=0.001,
                       linearDamping=0.0)
                    # p.changeDynamics(
                    #     sphereUid,
                    #     -1,
                    #     restitution=0.9,
                    #     spinningFriction=0.001,
                    #     rollingFriction=0.001,
                    #     linearDamping=0.0,
                    #     ccdSweptSphereRadius=0.01
                    # )

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
        model_path = dataset.model_path
            
    
    
    def render_thread_func(gaussians, pipeline, background):
        global last_position, last_angles
        while True:
            yaw, pitch, roll = key_input_handler.get_euler_angles()
            x, y, z = key_input_handler.get_position()
            if (x, y, z) == last_position and (yaw, pitch, roll) == last_angles:
                continue 
            last_position = (x, y, z)
            last_angles = (yaw, pitch, roll)
            
            view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[x, y, z], distance=1e-5, yaw=yaw, pitch=pitch, roll=roll, upAxisIndex=2)
            view_matrix = np.array(view_matrix, dtype=np.float32).reshape(4, 4)
            view_matrix[0:3, :] = adjust_matrix @ view_matrix[0:3, :]
            world_view_transform = torch.tensor(view_matrix, device='cuda')
            full_proj_transform = world_view_transform @ projection_matrix_torch
            custom_cam = MiniCam(width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform)
            with torch.no_grad():
                rendering = render(custom_cam, gaussians, pipeline, background)["render"]
                img_buffer = rendering.permute(1, 2, 0).cpu().numpy()
            global_images["3dgs"]=img_buffer
            
    
    def process_thread_func(projection_matrix_fov):
        while True:
            yaw, pitch, roll = key_input_handler.get_euler_angles()
            x, y, z = key_input_handler.get_position()
            view_matrix_rebot = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[x, -y, -z], distance=1e-5, yaw=-yaw, pitch=pitch, roll=-roll, upAxisIndex=2)
            img = kuka_camera(width, height, np.array(view_matrix_rebot, dtype=np.float32).reshape(4,4), projection_matrix_fov)
            img_buffer_2 = np.array(img[2][:,:,:3], dtype=np.float32) / 255
            mask = np.array((img[4] != 0), np.uint8)
            
            global_images["bullet"] = img_buffer_2
            global_images["mask"] = mask
            p.stepSimulation()
            
    def update_image():
        prev_time = time.time()
        img_buffer_2 = global_images["bullet"]
        mask = global_images["mask"] 
        img_buffer = global_images["3dgs"]
        mask_c3 = mask[:, :, None].astype(bool)
        img_blend = np.where(mask_c3, 0, img_buffer) + np.where(mask_c3, img_buffer_2, 0)
        current_time = time.time()
        elapsed_time = current_time - prev_time
        fps = 1 / elapsed_time
        dpg.set_value("rendered_robotarm", img_blend.ravel())
        dpg.set_value("euler_angles_text", f'FPS: {fps:.2f}')
        
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
        elif sender == "delta_x_slider":
            key_input_handler.update_deltas(app_data, key_input_handler.delta_y, key_input_handler.delta_z)
        elif sender == "delta_y_slider":
            key_input_handler.update_deltas(key_input_handler.delta_x, app_data, key_input_handler.delta_z)
        elif sender == "delta_z_slider":
            key_input_handler.update_deltas(key_input_handler.delta_x, key_input_handler.delta_y, app_data)
        elif sender == "robot_yaw_slider":
            key_input_handler.set_robot_euler_angles(app_data, key_input_handler.robot_pitch, key_input_handler.robot_roll)
        elif sender == "robot_pitch_slider":
            key_input_handler.set_robot_euler_angles(key_input_handler.robot_yaw, app_data, key_input_handler.robot_roll)
        elif sender == "robot_roll_slider":
            key_input_handler.set_robot_euler_angles(key_input_handler.robot_yaw, key_input_handler.robot_pitch, app_data)

    dpg.create_context()

    with dpg.texture_registry(show=False):
        dpg.add_raw_texture(width, height, np.zeros((height, width, 3)), format=dpg.mvFormat_Float_rgb, tag="rendered_robotarm")

    with dpg.window(label="Main Window", width=1800, height=600):
        with dpg.group(horizontal=False):
            with dpg.child_window(width=width, height=height):
                dpg.add_image("rendered_robotarm")
            with dpg.child_window(width=600, height=250, pos=(width+10,10)):
                dpg.add_text("FPS: 0.00", tag="euler_angles_text")
                dpg.add_slider_float(label="Yaw", default_value=0, min_value=-180, max_value=180, tag="yaw_slider", callback=key_callback)
                dpg.add_slider_float(label="Pitch", default_value=0, min_value=-180, max_value=180, tag="pitch_slider", callback=key_callback)
                dpg.add_slider_float(label="Roll", default_value=0, min_value=-180, max_value=180, tag="roll_slider", callback=key_callback)
                dpg.add_slider_float(label="X", default_value=0, min_value=-10, max_value=10, tag="x_slider", callback=key_callback)
                dpg.add_slider_float(label="Y", default_value=0, min_value=-10, max_value=10, tag="y_slider", callback=key_callback)
                dpg.add_slider_float(label="Z", default_value=0, min_value=-10, max_value=10, tag="z_slider", callback=key_callback)
            with dpg.child_window(width=600, height=250, pos=(width+10,280)):
                dpg.add_slider_float(label="Delta X", default_value=0, min_value=-1.5, max_value=1.5, width=500, tag="delta_x_slider", callback=key_callback)
                dpg.add_slider_float(label="Delta Y", default_value=0, min_value=-1.5, max_value=1.5, width=500,tag="delta_y_slider", callback=key_callback)
                dpg.add_slider_float(label="Delta Z", default_value=0, min_value=-1.5, max_value=1.5, width=500,tag="delta_z_slider", callback=key_callback)
                dpg.add_slider_float(label="Robot Yaw", default_value=0, min_value=-180, max_value=180, tag="robot_yaw_slider", callback=key_callback)
                dpg.add_slider_float(label="Robot Pitch", default_value=0, min_value=-180, max_value=180, tag="robot_pitch_slider", callback=key_callback)
                dpg.add_slider_float(label="Robot Roll", default_value=0, min_value=-180, max_value=180, tag="robot_roll_slider", callback=key_callback)

    dpg.create_viewport(title='Render Viewer', width=width, height=height)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    
    
    
    render_thread = threading.Thread(target=render_thread_func, args=(gaussians, pipeline, background), daemon=True)
    process_thread = threading.Thread(target=process_thread_func, args=(projection_matrix_fov,), daemon=True)
    render_thread.start()
    process_thread.start()
    timer = threading.Timer(5.0, generate_ball_event)
    timer.start()
    
    while dpg.is_dearpygui_running():
        update_image()
        dpg.render_dearpygui_frame()

    dpg.destroy_context()
    p.disconnect()

if __name__ == "__main__":
    main()
