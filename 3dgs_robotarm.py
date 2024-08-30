import threading
import time
import math
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import dearpygui.dearpygui as dpg
import pybullet as p
import pybullet_data
import random
from argparse import ArgumentParser


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
    img = p.getCameraImage(w, h, view_matrix, projection_matrix)
    return img

def main():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(1./50.)
    p.setRealTimeSimulation(1)
    
    
    eglPluginId = -1
    import pkgutil
    egl = pkgutil.get_loader('eglRenderer')
    if (egl):
        eglPluginId = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
    else:
        eglPluginId = p.loadPlugin("eglRendererPlugin")
    
    
    
    
    x_rebot = 0
    y_rebot = 2.9
    z_rebot = -0.65
 
    start_pos = [x_rebot, y_rebot, z_rebot]
    start_orientation = p.getQuaternionFromEuler([0, 0, 0])
    #plane_id = p.loadURDF("plane.urdf", [x_rebot, y_rebot, -1.65])
    #p.changeDynamics(plane_id, -1, restitution=1)
    kuka_id = p.loadURDF("rm65/urdf/rm_65.urdf", start_pos, start_orientation, useFixedBase=True)    
    numJoints = p.getNumJoints(kuka_id)
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
    projection_matrix_fov = compute_projection_matrix_fov(fovy=fovy, fovx=fovx, near_val=znear, far_val=zfar)


    #------ create mesh for physical activity -----------------------------------------------
    monastryId = concaveEnv = p.createCollisionShape(p.GEOM_MESH,
                                                    fileName="env_mesh_model/output_compressed.obj",
                                                    flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
    rotation = R.from_matrix(np.linalg.inv(adjust_matrix))

    euler_angles = rotation.as_euler('xyz', degrees=False)  # 使用xyz顺序得到欧拉角
    euler_angles[1] += np.pi 
    adjusted_rotation = R.from_euler('xyz', euler_angles)
    rotation_quaternion = adjusted_rotation.as_quat()  # 返回格式为 [x, y, z, w]

    # 将四元数转换为 PyBullet 格式 (x, y, z, w)
    rotation_quaternion = [rotation_quaternion[0], rotation_quaternion[1], rotation_quaternion[2], rotation_quaternion[3]]
    p.createMultiBody(0, monastryId, baseOrientation=rotation_quaternion)
    #------------------------------------------------------------------------------------------



    #---------------------------------------ball-----------------------------------------------
    ball_radius = 0.1
    collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_SPHERE, radius=ball_radius)
    visual_shape_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=ball_radius, rgbaColor=[1, 0, 0, 1])  # 红色


    ball_id = p.createMultiBody(baseMass=0.1, baseCollisionShapeIndex=collision_shape_id,
                                baseVisualShapeIndex=visual_shape_id, basePosition=[x_rebot+0.5, y_rebot, z_rebot])
    
    p.changeDynamics(ball_id, -1, mass=0.1,
                restitution=0.6, 
                lateralFriction=1.5,  
                rollingFriction=1.5,  
                spinningFriction=1.5,  
                linearDamping=0.9,     
                angularDamping=0.9
                ) 

    #--------------------------------------------------------------------------------------------------
    
    def generate_ball_event():
        sphereRadius = 0.05
        colSphereId = p.createCollisionShape(p.GEOM_SPHERE, radius=sphereRadius)
        colBoxId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[sphereRadius, sphereRadius, sphereRadius])

        mass = 0.1
        useMaximalCoordinates = 0
        visualShapeId = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=sphereRadius, rgbaColor=[0, 1, 1, 1])

        for i in range(5):
            for j in range(5):
                for k in range(5):
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

                    # 设置弹性和摩擦力
                    p.changeDynamics(
                        sphereUid,
                        -1,
                        restitution=0.95,  # 设置弹性系数，1.0表示完全弹性碰撞
                        spinningFriction=0.001,  # 旋转摩擦
                        rollingFriction=0.001,  # 滚动摩擦
                        linearDamping=0.0  # 线性阻尼
                    )

            time.sleep(0.1)  # 控制球的创建速度

    #--------------------------------------------------------------------------------------------------
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
    
    event_thread = threading.Thread(target=generate_ball_event)
    
    def release_ball():
        global constraint_id, attached
        if attached:
            p.removeConstraint(constraint_id)
            #attached = False
            print("Ball released!")
    
    
    def update_image():
        global constraint_id, attached
        nonlocal prev_time, fps, frame_count
        keys = p.getKeyboardEvents()
        if p.B3G_SPACE in keys and keys[p.B3G_SPACE] & p.KEY_WAS_TRIGGERED:
           event_thread.start()
        #print(keys) 
        with torch.no_grad():
            yaw, pitch, roll = key_input_handler.get_euler_angles()
            x, y, z = key_input_handler.get_position()
            
            delta_x, delta_y, delta_z = key_input_handler.get_deltas()
            robot_yaw, robot_pitch, robot_roll = key_input_handler.get_robot_euler_angles()
            

            target_orientation = p.getQuaternionFromEuler([math.radians(robot_yaw), math.radians(robot_pitch), math.radians(robot_roll)])
            jointPoses = p.calculateInverseKinematics(kuka_id, 5, [x_rebot + delta_x, y_rebot + delta_y, z_rebot + delta_z], target_orientation)
        
            for i in range(numJoints):
                p.setJointMotorControl2(bodyIndex=kuka_id,
                                        jointIndex=i,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPosition=jointPoses[i],
                                        targetVelocity=1,
                                        force=1000,
                                        positionGain=0.1,
                                        velocityGain=1)
            
            view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[x, y, z], distance=1e-5, yaw=yaw, pitch=pitch, roll=roll, upAxisIndex=2)
            view_matrix = np.array(view_matrix, np.float32).reshape(4,4)
            view_matrix[0:3,:] = adjust_matrix@view_matrix[0:3,:]
            world_view_transform = torch.tensor(view_matrix).cuda()
            full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix_torch.unsqueeze(0))).squeeze(0)
            custom_cam = MiniCam(width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform)
            rendering = render(custom_cam, gaussians, pipeline, background)["render"]
            img_buffer = rendering.permute(1, 2, 0).cpu().numpy()
            view_matrix_rebot = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[x, -y, -z], distance=1e-5, yaw=-yaw, pitch=pitch, roll=-roll, upAxisIndex=2)
                                               
            com_p, com_o, _, _, _, _ = p.getLinkState(kuka_id, 5, computeForwardKinematics=True)
            rebot_cam_x, rebot_cam_y, rebot_cam_z = com_p
            euler_angles = p.getEulerFromQuaternion(com_o)
            rebot_cam_yaw = math.degrees(euler_angles[2])
            rebot_cam_pitch = math.degrees(euler_angles[1])
            rebot_cam_roll = math.degrees(euler_angles[0])
            #original_cam_quaternion = p.getQuaternionFromEuler([rebot_cam_yaw, rebot_cam_pitch, rebot_cam_roll])
            #difference = np.array(target_orientation) - np.array(com_o)
            #print(f"Results Quaternion Difference: {difference}")
            
            ball_pos, _ = p.getBasePositionAndOrientation(ball_id)
            distance = ((com_p[0] - ball_pos[0]) ** 2 + (com_p[1] - ball_pos[1]) ** 2 + (com_p[2] - ball_pos[2]) ** 2) ** 0.5
            
            #print(distance)
            if distance < 0.24:
                if not attached:
                    
                    constraint_id = p.createConstraint(kuka_id, 5, ball_id, -1, p.JOINT_FIXED, 
                                    jointAxis=[0, 0, 0],
                                    parentFramePosition=[0, 0, 0],
                                    childFramePosition=[0, 0, 0])
                    attached = True
            else:
                attached = False
            #view_matrix_rebot_cam = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[rebot_cam_x, -rebot_cam_y, -rebot_cam_z], distance=1e-5, yaw=-rebot_cam_roll+90, pitch=-rebot_cam_pitch+90, roll=rebot_cam_yaw, upAxisIndex=2)
            view_matrix_rebot_cam = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[rebot_cam_x, -rebot_cam_y, -rebot_cam_z], distance=1e-5, yaw=-robot_roll+90, pitch=-robot_pitch+90, roll=robot_yaw, upAxisIndex=2) #debug

            view_matrix_rebot_cam = np.array(view_matrix_rebot_cam,np.float32).reshape(4,4)
            view_matrix_rebot_cam[0:3,:] = adjust_matrix@view_matrix_rebot_cam[0:3,:]
            rebot_view_transform = torch.tensor(view_matrix_rebot_cam).cuda()
            rebot_proj_transform = (rebot_view_transform.unsqueeze(0).bmm(projection_matrix_torch.unsqueeze(0))).squeeze(0)
            rebot_cam = MiniCam(width, height, fovy, fovx, znear, zfar, rebot_view_transform, rebot_proj_transform)
            rebot_view_rendering = render(rebot_cam, gaussians, pipeline, background)["render"]
            rebot_view_img_buffer = rebot_view_rendering.permute(1, 2, 0).cpu().numpy()
            

            img = kuka_camera(width, height, np.array(view_matrix_rebot, np.float32).reshape(4,4), projection_matrix_fov)
            img_buffer_2 = np.array(img[2][:,:,:3], dtype=np.float32) / 255
            mask = np.array((img[4] == 0)+(img[4] == 2)+ (img[4] >= 3), np.uint8)
            #cv2.imwrite('test.jpg',np.array((img[4] == 0), np.uint8)*255)  # 0 robot arm   many items   >=3 many items       
            mask_c3 = np.repeat(mask[..., np.newaxis], 3, 2)
            img1_masked = np.where(mask_c3 == 0, img_buffer * 255, 0)
            img2_masked = np.where(mask_c3 == 1, img_buffer_2 * 255, 0)
            img_blend = img1_masked + img2_masked
            
            dpg.set_value("rendered_image", rebot_view_img_buffer.reshape(-1))
            dpg.set_value("rendered_robotarm", (img_blend / 255).reshape(-1))
            dpg.set_value("euler_angles_text", f'FPS: {fps:.2f}, Yaw: {rebot_cam_yaw:.2f}, Pitch: {rebot_cam_pitch:.2f}, Roll: {rebot_cam_roll:.2f}')
            
            frame_count += 1
            current_time = time.time()
            elapsed_time = current_time - prev_time
            if elapsed_time >= 1.0:
                fps = frame_count / elapsed_time
                prev_time = current_time
                frame_count = 0

        p.stepSimulation()

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
        dpg.add_raw_texture(width, height, np.zeros((height, width, 3)), format=dpg.mvFormat_Float_rgb, tag="rendered_image")

    with dpg.window(label="Main Window", width=1800, height=1200):
        with dpg.group(horizontal=False):
            with dpg.child_window(width=width, height=height*2):
                dpg.add_image("rendered_image")
                dpg.add_image("rendered_robotarm")
            with dpg.child_window(width=600, height=300, pos=(width+10,10)):
                dpg.add_text("FPS: 0.00 yaw: pitch: roll:", tag="euler_angles_text")
                dpg.add_slider_float(label="Yaw", default_value=0, min_value=-180, max_value=180, tag="yaw_slider", callback=key_callback)
                dpg.add_slider_float(label="Pitch", default_value=0, min_value=-180, max_value=180, tag="pitch_slider", callback=key_callback)
                dpg.add_slider_float(label="Roll", default_value=0, min_value=-180, max_value=180, tag="roll_slider", callback=key_callback)
                dpg.add_slider_float(label="X", default_value=0, min_value=-10, max_value=10, tag="x_slider", callback=key_callback)
                dpg.add_slider_float(label="Y", default_value=0, min_value=-10, max_value=10, tag="y_slider", callback=key_callback)
                dpg.add_slider_float(label="Z", default_value=0, min_value=-10, max_value=10, tag="z_slider", callback=key_callback)
            with dpg.child_window(width=600, height=300, pos=(width+10,400)):
                dpg.add_slider_float(label="Delta X", default_value=0, min_value=-1.5, max_value=1.5, width=500, tag="delta_x_slider", callback=key_callback)
                dpg.add_slider_float(label="Delta Y", default_value=0, min_value=-1.5, max_value=1.5, width=500,tag="delta_y_slider", callback=key_callback)
                dpg.add_slider_float(label="Delta Z", default_value=0, min_value=-1.5, max_value=1.5, width=500,tag="delta_z_slider", callback=key_callback)
                dpg.add_slider_float(label="Robot Yaw", default_value=0, min_value=-180, max_value=180, tag="robot_yaw_slider", callback=key_callback)
                dpg.add_slider_float(label="Robot Pitch", default_value=0, min_value=-180, max_value=180, tag="robot_pitch_slider", callback=key_callback)
                dpg.add_slider_float(label="Robot Roll", default_value=0, min_value=-180, max_value=180, tag="robot_roll_slider", callback=key_callback)
                dpg.add_button(label="Release Ball", callback=lambda: release_ball()) 
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
