import pybullet as p
import websockets
import asyncio
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import json
import threading
from scipy.spatial.transform import Rotation as R
import pybullet_data
import torch
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


p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)
p.setTimeStep(1./100.)
#p.setRealTimeSimulation(1)

adjust_matrix = np.array([
    [-0.01396038, -0.51966003, -0.85425907],
    [0.99977363, 0.00646452, -0.02027087],
    [0.01605634, -0.85434868, 0.51945214],
])

monastryId = p.createCollisionShape(
    p.GEOM_MESH,
    fileName="env_mesh_model/output_compressed.obj",
    flags=p.GEOM_FORCE_CONCAVE_TRIMESH
)

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
key_input_handler = {'yaw': 0, 'pitch': 0, 'roll': 0, 'x': 0, 'y': 0, 'z': 0}  # 初始控制参数

parser = ArgumentParser(description="Testing script parameters")
model = ModelParams(parser, sentinel=True)
pipeline = PipelineParams(parser)
parser.add_argument("--iteration", default=-1, type=int)
args = get_combined_args(parser)

torch.manual_seed(0)
torch.cuda.set_device(torch.device("cuda:0"))

dataset = model.extract(args)
iteration = args.iteration
pipeline = pipeline.extract(args)

gaussians = GaussianModel(dataset.sh_degree)
scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

fovy = 1.1064156765004665
fovx = 1.6639937226014894
height = 540
width = 960
zfar = 100.0
znear = 0.01

projection_matrix_torch = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy).transpose(0, 1).cuda()
projection_matrix_fov = compute_projection_matrix_fov(fovy=fovy, fovx=fovx, near_val=znear, far_val=zfar)

def kuka_camera(w, h, view_matrix, proj_matrix):
    projection_matrix = tuple(proj_matrix.reshape(-1))
    view_matrix = tuple(view_matrix.reshape(-1))
    img = p.getCameraImage(w, h, view_matrix, projection_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)    
    return img


def generate_ball_event():
    sphereRadius = 0.05
    # colSphereId = p.createCollisionShape(p.GEOM_SPHERE, radius=sphereRadius)
    # colBoxId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[sphereRadius, sphereRadius, sphereRadius])
    
    colSphereId = p.createCollisionShape(p.GEOM_SPHERE, radius=sphereRadius)
    colBoxId = p.createCollisionShape(p.GEOM_BOX,
                                    halfExtents=[sphereRadius, sphereRadius, sphereRadius])


    mass = 1
    useMaximalCoordinates = 0
    visualShapeId = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=sphereRadius, rgbaColor=[0, 1, 1, 1])
    #visualShapeId = -1
    for i in range(3):
        for j in range(3):
            for k in range(3):
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
                # p.changeDynamics(sphereUid,
                #     -1,
                #     spinningFriction=0.001,
                #     rollingFriction=0.001,
                #     linearDamping=0.0)
                p.changeDynamics(
                    sphereUid,
                    -1,
                    restitution=0.9,
                    spinningFriction=0.001,
                    rollingFriction=0.001,
                    linearDamping=0.0,
                    ccdSweptSphereRadius=0.01
                )


async def send_and_receive_data(websocket, path):
    try:
        while True:
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[key_input_handler['x'], key_input_handler['y'], key_input_handler['z']],
                distance=1e-5, yaw=key_input_handler['yaw'], pitch=key_input_handler['pitch'], roll=key_input_handler['roll'], upAxisIndex=2
            )
            view_matrix = np.array(view_matrix, dtype=np.float32).reshape(4, 4)
            view_matrix[0:3, :] = adjust_matrix @ view_matrix[0:3, :]
            world_view_transform = torch.tensor(view_matrix, device='cuda')
            full_proj_transform = world_view_transform @ projection_matrix_torch
            custom_cam = MiniCam(width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform)

            with torch.no_grad():
                rendering = render(custom_cam, gaussians, pipeline, background)["render"]
                img_buffer = rendering.permute(1, 2, 0).cpu().numpy()


            view_matrix_rebot = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[key_input_handler['x'], -key_input_handler['y'], -key_input_handler['z']], distance=1e-5, yaw=-key_input_handler['yaw'], pitch=key_input_handler['pitch'], roll=-key_input_handler['roll'], upAxisIndex=2)
            img = kuka_camera(width, height, np.array(view_matrix_rebot, dtype=np.float32).reshape(4,4), projection_matrix_fov)
            img_buffer_2 = np.array(img[2][:,:,:3], dtype=np.float32) / 255
            mask = np.array((img[4] == 0),np.int8)

            mask_c3 = ~mask[:, :, None].astype(bool)
            img_blend = np.where(mask_c3, 0, img_buffer) + np.where(mask_c3, img_buffer_2, 0)
            #img_blend = np.where(mask_c3, img_buffer_2, 0)
            img_pil = Image.fromarray(np.array(img_blend*255, np.uint8))
            buffer = BytesIO()
            img_pil.save(buffer, format="JPEG")
            img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
            await websocket.send(img_str)
            message = await websocket.recv()
            controls = json.loads(message)
            key_input_handler.update(controls)

            p.stepSimulation()
            await asyncio.sleep(0.1)
    except websockets.ConnectionClosed:
        print("客户端连接已关闭")
    finally:
        p.disconnect()
        
timer = threading.Timer(5.0, generate_ball_event)
timer.start()
start_server = websockets.serve(send_and_receive_data, "", 8769)
print("WebSocket 服务器已启动，等待客户端连接...")
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
