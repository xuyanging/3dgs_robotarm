import os
import json
import threading
import cv2
import csv
import re
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
from sklearn.decomposition import PCA
import asyncio
import websockets
import threading
from websocket import create_connection


from libinfer_3dgs import render
from libinfer_3dgs.gaussian_model import GaussianModel
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
from physical_lib.vis import visualize_pointcloud_with_open3d_boxes8

is_recording = False
playing = False
finished_ik = True
record_xy = [] 
file_path = "coordinates.csv"

n_xyz = [0, 1, 0] # 垂直xy平面的单位向量
# 镜像矩阵
rotation_matrices_y = torch.Tensor(np.array([
    [1-2*n_xyz[0]**2, -2*n_xyz[0]*n_xyz[1], -2*n_xyz[0]*n_xyz[2]],
    [-2*n_xyz[0]*n_xyz[1], 1-2*n_xyz[1]**2, -2*n_xyz[1]*n_xyz[2]],
    [-2*n_xyz[0]*n_xyz[2], -2*n_xyz[1]*n_xyz[2], 1-2*n_xyz[2]**2]])).cuda()

n_xyz = [1, 0, 0] 
rotation_matrices_x = torch.Tensor(np.array([
    [1-2*n_xyz[0]**2, -2*n_xyz[0]*n_xyz[1], -2*n_xyz[0]*n_xyz[2]],
    [-2*n_xyz[0]*n_xyz[1], 1-2*n_xyz[1]**2, -2*n_xyz[1]*n_xyz[2]],
    [-2*n_xyz[0]*n_xyz[2], -2*n_xyz[1]*n_xyz[2], 1-2*n_xyz[2]**2]])).cuda()

n_xyz = [0, 0, 1] 
rotation_matrices_z = torch.Tensor(np.array([
    [1-2*n_xyz[0]**2, -2*n_xyz[0]*n_xyz[1], -2*n_xyz[0]*n_xyz[2]],
    [-2*n_xyz[0]*n_xyz[1], 1-2*n_xyz[1]**2, -2*n_xyz[1]*n_xyz[2]],
    [-2*n_xyz[0]*n_xyz[2], -2*n_xyz[1]*n_xyz[2], 1-2*n_xyz[2]**2]])).cuda()


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

def compute_projection_matrix_fov(fovy, fovx, near_val, far_val):
    y_scale = 1.0 / math.tan(fovy / 2)
    x_scale = 1.0 / math.tan(fovx / 2)

    projection_matrix = np.zeros((4, 4), dtype=np.float32)
    projection_matrix[0, 0] = x_scale
    projection_matrix[1, 1] = y_scale
    projection_matrix[2, 2] = (near_val + far_val) / (near_val - far_val)
    projection_matrix[2, 3] = -1
    projection_matrix[3, 2] = (2 * far_val * near_val) / (near_val - far_val)
    return projection_matrix

def get_intrinsic_matrix(image_width, image_height, fovX, fovY):
    
    f_x = image_width / (2.0 * math.tan(fovX / 2))
    f_y = image_height / (2.0 * math.tan(fovY / 2))
    c_x = image_width / 2.0
    c_y = image_height / 2.0
    K = torch.zeros(3, 3)
    K[0, 0] = f_x
    K[1, 1] = f_y
    K[0, 2] = c_x
    K[1, 2] = c_y
    K[2, 2] = 1.0
    return K

def get_rotated_bounding_box(points):
    
    assert points.ndim == 2 and points.shape[1] == 3, "Input should be of shape [N, 3]"
    pca = PCA(n_components=3)
    pca.fit(points.cpu().numpy()) 
    rotation_matrix = torch.tensor(pca.components_.T, dtype=points.dtype, device=points.device)
    rotated_points = points @ rotation_matrix
    min_coords = torch.min(rotated_points, dim=0)[0]
    max_coords = torch.max(rotated_points, dim=0)[0]
    bounding_box = torch.tensor([[min_coords[0], min_coords[1], min_coords[2]],
                                  [max_coords[0], min_coords[1], min_coords[2]],
                                  [min_coords[0], max_coords[1], min_coords[2]],
                                  [max_coords[0], max_coords[1], min_coords[2]],
                                  [min_coords[0], min_coords[1], max_coords[2]],
                                  [max_coords[0], min_coords[1], max_coords[2]],
                                  [min_coords[0], max_coords[1], max_coords[2]],
                                  [max_coords[0], max_coords[1], max_coords[2]]], 
                                  dtype=points.dtype, device=points.device)
    bounding_box = bounding_box @ rotation_matrix.T
    return bounding_box

def move_plane(surface, distance):
    v1 = surface[1] - surface[0]
    v2 = surface[2] - surface[0]
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)  
    moved_surface = surface + distance * normal
    return moved_surface

def getProjectionMatrix_inverse(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = -1.0  

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign  
    P[2, 2] = z_sign * (zfar+znear) / (zfar - znear)  
    P[2, 3] = -2*(zfar * znear) / (zfar - znear)  

    return P

def calculate_3d_box_points(xyz, yaw, pitch, roll, w, h):
    
    def euler_to_rotation_matrix(yaw, pitch, roll):

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
        
        R = R_z @ R_y @ R_x
        return R

    xyz = np.array(xyz)

    half_w, half_h = w / 2, h / 2
    local_points = np.array([
        [-half_w, -half_h, 0],  
        [half_w, -half_h, 0],   
        [half_w, half_h, 0],    
        [-half_w, half_h, 0]    
    ])

    R = euler_to_rotation_matrix(yaw, pitch, roll)
    global_points = []
    for point in local_points:
        rotated_point = R @ point
        global_point = rotated_point + xyz
        global_points.append(global_point)

    return np.array(global_points)

def check_orientation(points):
    points = np.unique(points, axis=0)
    if len(points) < 3:
        return "Invalid shape"  # 点数不足，无法形成多边形
    elif len(points) == 3:
        return "Triangle"  # 三个点，形成三角形
    elif len(points) > 4:
        return "Invalid shape"  # 超出4个点，无法判断

    def cross_product(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
    cross_products = []
    for i in range(len(points)):
        p1 = points[i]
        p2 = points[(i + 1) % len(points)]
        p3 = points[(i + 2) % len(points)]
        cross_products.append(cross_product(p1, p2, p3))

    if all(cp > 0 for cp in cross_products) or all(cp < 0 for cp in cross_products):
        print("Facing Outward")
    else:
        print("Facing Inward")

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

frame = None

async def receive_frame():
    global frame
    uri = "ws://localhost:8765"
    async with websockets.connect(uri) as websocket:
        while True:
            frame_data = await websocket.recv()
            np_arr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def start_receive_frame():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(receive_frame())


thread = threading.Thread(target=start_receive_frame, daemon=True)
thread.start()

def send_click_coordinates_sync(x, y):
    uri = "ws://localhost:8766"
    try:
        # 建立同步 WebSocket 连接
        websocket = create_connection(uri)
        
        # 创建点击数据并发送
        click_data = {
            "x": x,
            "y": y
        }
        websocket.send(json.dumps(click_data))
        print(is_recording)
        if is_recording:
            with open(file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([x, y]) 
                print(f"Saved coordinates: ({x}, {y})")
        print(f"x={x}, y={y}")
        
        # 关闭连接
        websocket.close()
    except Exception as e:
        print(f"Error occurred: {e}")

def send_coordinates():
    global u_warp, v_warp
    if u_warp > 0 and v_warp > 0:
        send_click_coordinates_sync(int(u_warp), int(v_warp)) 
         


def project_3D_to_2D(surface, extrinsic_matrix, K):
    points_3D_homogeneous = np.hstack((surface, np.ones((surface.shape[0], 1))))
    points_camera = points_3D_homogeneous @ extrinsic_matrix.T
    X_camera, Y_camera, Z_camera = points_camera[:, 0], points_camera[:, 1], points_camera[:, 2]
    points_image_homogeneous = (K @ np.vstack((X_camera, Y_camera, Z_camera))).numpy()
    u = np.array(points_image_homogeneous[0] / points_image_homogeneous[2], dtype=np.float32)
    v = np.array(points_image_homogeneous[1] / points_image_homogeneous[2], dtype=np.float32)
    return u, v

def map_dst_to_src(x, y, M_inv):
    dst_coord = np.array([x, y, 1], dtype=np.float32).reshape(3, 1)
    src_coord = M_inv.dot(dst_coord)
    src_coord /= src_coord[2]  
    return int(src_coord[0]), int(src_coord[1])

def record_start():
    global is_recording
    is_recording = True
    print('Recording...')
    if os.path.exists(file_path):
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["x", "y"])  # 写入表头
    else:
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["x", "y"])
def record_stop():
    global is_recording
    is_recording = False
    print('Finish!')

def replay():
    global playing, finished_ik
    playing = True
    finished_ik = False
    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        
        # 读取表头
        header = next(reader)
        print("Header:", header)
        
        # 读取每一行数据
        for row in reader:
            x, y = row
            record_xy.append([x,y])
            print(f"Coordinates: x={x}, y={y}")

def rotate_pointcloud_around_box_center(points, angle, axis='z'):
    points_centered = points
    # 构造旋转矩阵
    if axis == 'x':
        R = torch.tensor([[1, 0, 0],
                          [0, np.cos(angle), -np.sin(angle)],
                          [0, np.sin(angle), np.cos(angle)]], dtype=points.dtype)
    elif axis == 'y':
        R = torch.tensor([[np.cos(angle), 0, np.sin(angle)],
                          [0, 1, 0],
                          [-np.sin(angle), 0, np.cos(angle)]], dtype=points.dtype)
    elif axis == 'z':
        R = torch.tensor([[np.cos(angle), -np.sin(angle), 0],
                          [np.sin(angle), np.cos(angle), 0],
                          [0, 0, 1]], dtype=points.dtype)
    else:
        raise ValueError("轴应为 'x', 'y' 或 'z'")
    points_rotated = torch.matmul(points_centered, R.cuda().T)
    points_rotated = points_rotated 
    return points_rotated

def get_rotate_matrices(angle, axis='z'):
    if axis == 'x':
        R = torch.tensor([[1, 0, 0],
                          [0, np.cos(angle), -np.sin(angle)],
                          [0, np.sin(angle), np.cos(angle)]], dtype=torch.float32)
    elif axis == 'y':
        R = torch.tensor([[np.cos(angle), 0, np.sin(angle)],
                          [0, 1, 0],
                          [-np.sin(angle), 0, np.cos(angle)]], dtype=torch.float32)
    elif axis == 'z':
        R = torch.tensor([[np.cos(angle), np.sin(-angle), 0],
                          [np.sin(angle), np.cos(angle), 0],
                          [0, 0, 1]], dtype=torch.float32)
    else:
        raise ValueError("轴应为 'x', 'y' 或 'z'")

    return R.cuda()

def main():   
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(1./50.)
    p.setRealTimeSimulation(1)
     
    x_rebot = -0.5
    y_rebot = -2
    z_rebot = -0.7
 
    start_pos = [x_rebot, y_rebot, z_rebot]
    start_orientation = p.getQuaternionFromEuler([0, 0, 0])
    kuka_id = p.loadURDF("rm65/urdf/rm_65.urdf", start_pos, start_orientation, useFixedBase=True, globalScaling=4.0)    
    numJoints = p.getNumJoints(kuka_id)
    key_input_handler = KeyInputHandler()
          
    fovy = 1.6639937226014894
    fovx = 1.1064156765004665*0.9
    height = 1920
    width = 1080
    zfar = 100.0
    znear = 0.01

    basePosition=[0, 0, 0]
    baseorientation=[np.pi, 0, 0]


    K= get_intrinsic_matrix(width, height, fovx, fovy)
    projection_matrix_torch = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy).cuda().transpose(0, 1)
    projection_matrix_fov = compute_projection_matrix_fov(fovy=fovy, fovx=fovx, near_val=znear, far_val=zfar)
    #------ create mesh for physical activity -----------------------------------------------
    concaveEnv = p.createCollisionShape(p.GEOM_MESH,
                                        fileName="env_mesh_model/sculpture_custom_nw_iterations30000_DLNR_Middlebury_baseline7_0p_mask0_occ1_scale1_0_voxel2_512_trunc4_20_cleaned_mesh.obj",  
                                        flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
    monastryId = p.createMultiBody(baseMass=0,  
                                baseCollisionShapeIndex=concaveEnv,
                                basePosition=basePosition,  
                                baseOrientation=p.getQuaternionFromEuler(baseorientation))  
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
    pipeline.compute_cov3D_python = True
    with torch.no_grad():
        gaussians = load_checkpoint(args.model_path)
        params = load_params_from_gs(gaussians, pipeline)
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    init_pos = params["pos"]
    init_cov = params["cov3D_precomp"]
    init_screen_points = params["screen_points"]
    init_opacity = params["opacity"]
    init_shs = params["shs"]
    global rotated_pos, rotated_cov
    rotated_pos = init_pos
    rotated_cov = init_cov
    
    surface = calculate_3d_box_points([0.74, 1.52, 0.7], -np.pi/40, -np.pi/36-np.pi/72, np.pi/12, w=1.2, h=2.55)
    surface = move_plane(surface, 0.1)
    transform1 = np.array([
        [1,0,0,0],
        [0,-1,0,0],
        [0,0,-1,0],
        [0,0,0,1]])
    view_trans = np.array([
        [1,-1,-1,-1],
        [-1,1,1,1],
        [-1,1,1,1],
        [1,-1,-1,1]])
    points_homogeneous = np.hstack((surface, np.ones((4, 1))))  
    rotated_points_homogeneous = (transform1 @ points_homogeneous.T).T
    surface = rotated_points_homogeneous[:, :3]

    prev_time = time.time()
    fps = 0
    frame_count = 0
    angle = 0      
    def update_image():
        #global rotated_cov, rotated_pos
        #nonlocal angle
        #angle+=0.01
        #rotated_pos_r = rotate_pointcloud_around_box_center(rotated_pos, angle, axis='x')
        #R = get_rotate_matrices(angle, axis='x')
        #rotated_cov_r = apply_cov_rotations(rotated_cov, [R])
        #rotated_cov_r = rotated_cov
        #baseorientation[0]=np.pi+angle
        #p.resetBasePositionAndOrientation(monastryId, basePosition, p.getQuaternionFromEuler(baseorientation))
        collision_points = p.getClosestPoints(bodyA=kuka_id, bodyB=monastryId, distance=0.005)   
        global constraint_id, attached, frame, M
        global u_warp, v_warp
        nonlocal prev_time, fps, frame_count
        global record_xy, finished_ik
        with torch.no_grad():
            yaw, pitch, roll = key_input_handler.get_euler_angles()
            x, y, z = key_input_handler.get_position()     
            view_matrix = np.array(p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[x, y, z], distance=5, yaw=yaw, pitch=pitch, roll=roll, upAxisIndex=2), np.float32).reshape(4,4)
            view_matrix_mirror = np.array(view_matrix*view_trans, np.float32).reshape(4,4)
            world_view_transform = torch.tensor(view_matrix_mirror).cuda()
            full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix_torch.unsqueeze(0))).squeeze(0)
            custom_cam = MiniCam(width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform)
            rasterizer = initialize_resterize(custom_cam, gaussians, pipeline, background)
            colors_precomp = convert_SH(init_shs, custom_cam, gaussians, rotated_pos, None)
            rendering, _ = rasterizer(
                        means3D=rotated_pos,
                        means2D=init_screen_points,
                        shs=None,
                        colors_precomp=colors_precomp,
                        opacities=init_opacity,
                        scales=None,
                        rotations=None,
                        cov3D_precomp=rotated_cov)
                       
            extrinsic_matrix = np.hstack((view_matrix[:3, :3].T, view_matrix[3, :3].reshape(3, 1)))
            extrinsic_matrix = rotation_matrices_x.cpu().numpy() @ extrinsic_matrix
            u,v = project_3D_to_2D(surface=surface, extrinsic_matrix=extrinsic_matrix, K=K) 
            img_buffer = np.ascontiguousarray(np.array(rendering.permute(1, 2, 0).cpu().numpy())*255) 
            img = kuka_camera(width, height, np.array(view_matrix, np.float32).reshape(4,4), projection_matrix_fov)
            img_buffer_2 = np.array(img[2][:,:,:3], dtype=np.float32) 

            if frame is not None:
                frame_dis = cv2.cvtColor(np.array(cv2.resize(frame,(400,600)), dtype=np.float32),cv2.COLOR_BGR2RGB)
                src_points = np.float32([[0, 0], [frame.shape[1], 0], [frame.shape[1], frame.shape[0]], [0, frame.shape[0]]])
                dst_points = np.float32([[u[0],v[0]],[u[1],v[1]],[u[2],v[2]],[u[3],v[3]]])
                #check_orientation(dst_points)

                R = extrinsic_matrix[:,:3]
                t = extrinsic_matrix[:, 3]
                vec_ab = surface[1] - surface[0]  
                vec_ac = surface[2] - surface[0]  
                normal = np.cross(vec_ab, vec_ac)         
                normal = normal / np.linalg.norm(normal)  
                camera_center = -np.linalg.inv(R) @ t  
                plane_center = np.mean(surface, axis=0)  
                to_plane = plane_center - camera_center     
                to_plane_normalized = to_plane / np.linalg.norm(to_plane)  
                cos_theta = np.dot(normal, to_plane_normalized)  
                if cos_theta > 0:
                    M = cv2.getPerspectiveTransform(src_points, dst_points)
                    warped_img = cv2.warpPerspective(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB), M, (width, height))
                    mask = np.zeros((height, width), dtype=np.uint8)
                    cv2.fillConvexPoly(mask, dst_points.astype(int), 255)
                    output_img = cv2.bitwise_and(warped_img, warped_img, mask=mask)
                    img_buffer[mask == 255] = output_img[mask == 255]
                
            if collision_points:
                for point in collision_points:
                    contact_position = point[5]  
                    u_touch, v_touch = project_3D_to_2D(surface=np.array(contact_position).reshape(1,3), extrinsic_matrix=extrinsic_matrix, K=K)
                    u_warp, v_warp = map_dst_to_src(u_touch[0], v_touch[0], np.linalg.inv(M))
                    cv2.circle(frame_dis, (int(u_warp*400/frame.shape[1]),int(v_warp*600/frame.shape[0])), 3, (255,0,0), -1)
            else:
                u_warp, v_warp = -1, -1
            
            mask = np.array((img[4] == 0), np.uint8)     
            mask_c3 = np.repeat(mask[..., np.newaxis], 3, 2)
            img1_masked = np.where(mask_c3 == 0, img_buffer, 0)
            img2_masked = np.where(mask_c3 == 1, img_buffer_2, 0)
            img_blend = img1_masked + img2_masked
          
        if len(record_xy)>0 and finished_ik is False:
            pass
        else:
            delta_x, delta_y, delta_z = key_input_handler.get_deltas()
            robot_yaw, robot_pitch, robot_roll = key_input_handler.get_robot_euler_angles()     

            target_orientation = p.getQuaternionFromEuler([math.radians(robot_yaw), math.radians(robot_pitch), math.radians(robot_roll)])
            jointPoses = p.calculateInverseKinematics(kuka_id, 5, [x_rebot + delta_x, y_rebot + delta_y, z_rebot + delta_z], target_orientation)
        
            for i in range(numJoints):
                p.setJointMotorControl2(bodyIndex=kuka_id,
                                        jointIndex=i,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPosition=jointPoses[i],
                                        targetVelocity=0,
                                        force=500,
                                        positionGain=0.05,
                                        velocityGain=0.5)
        p.stepSimulation()

        frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - prev_time
        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time
            prev_time = current_time
            frame_count = 0
        dpg.set_value("rendered_image", (frame_dis/255).reshape(-1))
        dpg.set_value("rendered_robotarm", (img_blend/255).reshape(-1))
        dpg.set_value("euler_angles_text", f'FPS: {fps:.2f}, X: {u_warp:.2f}, Y: {v_warp:.2f}')

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
        dpg.add_raw_texture(400, 600, np.zeros((600, 400, 3)), format=dpg.mvFormat_Float_rgb, tag="rendered_image")

    with dpg.window(label="Main Window", width=1800, height=1600):
        with dpg.group(horizontal=False):
            with dpg.child_window(width=width, height=height):
                #dpg.add_image("rendered_image")
                dpg.add_image("rendered_robotarm")
            with dpg.child_window(width=600, height=400, pos=(width+10,10)):
                dpg.add_text("FPS: 0.00 yaw: pitch: roll:", tag="euler_angles_text")
                dpg.add_slider_float(label="Yaw", default_value=0, min_value=-180, max_value=180, tag="yaw_slider", callback=key_callback)
                dpg.add_slider_float(label="Pitch", default_value=0, min_value=-180, max_value=180, tag="pitch_slider", callback=key_callback)
                dpg.add_slider_float(label="Roll", default_value=0, min_value=-180, max_value=180, tag="roll_slider", callback=key_callback)
                dpg.add_slider_float(label="X", default_value=0, min_value=-10, max_value=10, tag="x_slider", callback=key_callback)
                dpg.add_slider_float(label="Y", default_value=0, min_value=-10, max_value=10, tag="y_slider", callback=key_callback)
                dpg.add_slider_float(label="Z", default_value=0, min_value=-10, max_value=10, tag="z_slider", callback=key_callback)
                dpg.add_slider_float(label="Delta X", default_value=0, min_value=-2, max_value=2, width=500, tag="delta_x_slider", callback=key_callback)
                dpg.add_slider_float(label="Delta Y", default_value=0, min_value=-2, max_value=2, width=500,tag="delta_y_slider", callback=key_callback)
                dpg.add_slider_float(label="Delta Z", default_value=0, min_value=-2, max_value=2, width=500,tag="delta_z_slider", callback=key_callback)
                dpg.add_slider_float(label="Robot Yaw", default_value=0, min_value=-180, max_value=180, tag="robot_yaw_slider", callback=key_callback)
                dpg.add_slider_float(label="Robot Pitch", default_value=0, min_value=-180, max_value=180, tag="robot_pitch_slider", callback=key_callback)
                dpg.add_slider_float(label="Robot Roll", default_value=0, min_value=-180, max_value=180, tag="robot_roll_slider", callback=key_callback)
            with dpg.child_window(width=400, height=600, pos=(width+10,400+10)):
                dpg.add_image("rendered_image")
            with dpg.child_window(width=300, height=300, pos=(width+410,400+10)): 
                dpg.add_button(label="Click", callback=send_coordinates, width=150, height=50)
                dpg.add_button(label="Start Record", callback=record_start, width=150, height=50)
                dpg.add_button(label="Stop Record", callback=record_stop, width=150, height=50)
                dpg.add_button(label="Replay", callback=replay, width=150, height=50)
                     
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
