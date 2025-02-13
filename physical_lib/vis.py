import torch
import numpy as np
import open3d as o3d
from physical_lib.render_utils_3dgs import get_mat_from_upper
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

def visualize_pointcloud_with_open3d(sim_area, points, line_width=2.0, point_size=1.0, point_alpha=0.5, view_matrix=None):
    # 创建点云对象
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    # 设置点云颜色（可调整透明度）
    colors = np.zeros_like(points) + [0, 0, 1]  # 蓝色
    point_cloud.colors = o3d.utility.Vector3dVector(colors * point_alpha + (1 - point_alpha))  # 设置点的透明度

    # 定义3D框范围
    x_min, x_max, y_min, y_max, z_min, z_max = sim_area
    points_box = [
        [x_min, y_min, z_min], [x_min, y_min, z_max],
        [x_min, y_max, z_min], [x_min, y_max, z_max],
        [x_max, y_min, z_min], [x_max, y_min, z_max],
        [x_max, y_max, z_min], [x_max, y_max, z_max]
    ]

    # 创建线框边界
    lines = [[0, 1], [1, 3], [3, 2], [2, 0],
             [4, 5], [5, 7], [7, 6], [6, 4],
             [0, 4], [1, 5], [2, 6], [3, 7]]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points_box)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(lines))])  # 红色边框

    # 创建坐标系
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])

    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(point_cloud)
    vis.add_geometry(line_set)
    vis.add_geometry(coordinate_frame)

    # 如果提供了view_matrix，计算并显示相机视锥体
    if view_matrix is not None:
        # 视锥体的参数（视场角、纵横比、近远平面）
        fov = np.rad2deg(1.1064156765004665)  # 视场角度
        aspect_ratio = 1.6639937226014894/1.1064156765004665
        near_plane = 0.01  # 近平面距离
        far_plane = 1  # 远平面距离

        # 计算近平面和远平面的宽高
        near_height = 2 * np.tan(np.radians(fov / 2)) * near_plane
        near_width = near_height * aspect_ratio
        far_height = 2 * np.tan(np.radians(fov / 2)) * far_plane
        far_width = far_height * aspect_ratio

        # 定义相机的四个平面角点（近平面和远平面）
        frustum_points = np.array([
            [-near_width / 2, -near_height / 2, near_plane],
            [near_width / 2, -near_height / 2, near_plane],
            [near_width / 2, near_height / 2, near_plane],
            [-near_width / 2, near_height / 2, near_plane],
            [-far_width / 2, -far_height / 2, far_plane],
            [far_width / 2, -far_height / 2, far_plane],
            [far_width / 2, far_height / 2, far_plane],
            [-far_width / 2, far_height / 2, far_plane]
        ])

        # 转换视锥体点到相机坐标系
        frustum_points = np.hstack((frustum_points, np.ones((8, 1))))  # 增加齐次坐标
        frustum_points = (view_matrix @ frustum_points.T).T[:, :3]  # 应用视图矩阵

        # 创建相机视锥体的线条
        frustum_lines = [[0, 1], [1, 2], [2, 3], [3, 0],  # 近平面
                         [4, 5], [5, 6], [6, 7], [7, 4],  # 远平面
                         [0, 4], [1, 5], [2, 6], [3, 7]]  # 连接近平面和远平面

        frustum_line_set = o3d.geometry.LineSet()
        frustum_line_set.points = o3d.utility.Vector3dVector(frustum_points)
        frustum_line_set.lines = o3d.utility.Vector2iVector(frustum_lines)
        frustum_line_set.colors = o3d.utility.Vector3dVector([[0, 1, 0] for _ in range(len(frustum_lines))])  # 绿色视锥体

        # 添加视锥体到可视化窗口
        vis.add_geometry(frustum_line_set)

    # 调整点大小和线宽
    opt = vis.get_render_option()
    opt.point_size = point_size
    opt.line_width = line_width
    opt.background_color = np.asarray([1, 1, 1])  # 设置背景为白色

    # 可视化
    vis.run()
    vis.destroy_window()

def visualize_pointcloud_with_open3d_boxes8(sim_area, points, line_width=2.0, point_size=1.0, point_alpha=0.5, view_matrix=None):
    # 创建点云对象
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    # 设置点云颜色（可调整透明度）
    colors = np.zeros_like(points) + [0, 0, 1]  # 蓝色
    point_cloud.colors = o3d.utility.Vector3dVector(colors * point_alpha + (1 - point_alpha))  # 设置点的透明度

    # 使用 sim_area 作为 3D 盒子的 8 个顶点
    points_box = np.array(sim_area).reshape(8, 3)

    # 创建线框边界
    lines = [[0, 1], [1, 3], [3, 2], [2, 0],  # 近平面
             [4, 5], [5, 7], [7, 6], [6, 4],  # 远平面
             [0, 4], [1, 5], [2, 6], [3, 7]]  # 连接近平面和远平面

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points_box)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(lines))])  # 红色边框

    # 创建坐标系
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])

    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(point_cloud)
    vis.add_geometry(line_set)
    vis.add_geometry(coordinate_frame)

    # 如果提供了view_matrix，计算并显示相机视锥体
    if view_matrix is not None:
        # 视锥体的参数（视场角、纵横比、近远平面）
        fov = np.rad2deg(1.1064156765004665)  # 视场角度
        aspect_ratio = 1.6639937226014894/1.1064156765004665
        near_plane = 0.01  # 近平面距离
        far_plane = 1  # 远平面距离

        # 计算近平面和远平面的宽高
        near_height = 2 * np.tan(np.radians(fov / 2)) * near_plane
        near_width = near_height * aspect_ratio
        far_height = 2 * np.tan(np.radians(fov / 2)) * far_plane
        far_width = far_height * aspect_ratio

        # 定义相机的四个平面角点（近平面和远平面）
        frustum_points = np.array([
            [-near_width / 2, -near_height / 2, near_plane],
            [near_width / 2, -near_height / 2, near_plane],
            [near_width / 2, near_height / 2, near_plane],
            [-near_width / 2, near_height / 2, near_plane],
            [-far_width / 2, -far_height / 2, far_plane],
            [far_width / 2, -far_height / 2, far_plane],
            [far_width / 2, far_height / 2, far_plane],
            [-far_width / 2, far_height / 2, far_plane]
        ])

        # 转换视锥体点到相机坐标系
        frustum_points = np.hstack((frustum_points, np.ones((8, 1))))  # 增加齐次坐标
        frustum_points = (view_matrix @ frustum_points.T).T[:, :3]  # 应用视图矩阵

        # 创建相机视锥体的线条
        frustum_lines = [[0, 1], [1, 2], [2, 3], [3, 0],  # 近平面
                         [4, 5], [5, 6], [6, 7], [7, 4],  # 远平面
                         [0, 4], [1, 5], [2, 6], [3, 7]]  # 连接近平面和远平面

        frustum_line_set = o3d.geometry.LineSet()
        frustum_line_set.points = o3d.utility.Vector3dVector(frustum_points)
        frustum_line_set.lines = o3d.utility.Vector2iVector(frustum_lines)
        frustum_line_set.colors = o3d.utility.Vector3dVector([[0, 1, 0] for _ in range(len(frustum_lines))])  # 绿色视锥体

        # 添加视锥体到可视化窗口
        vis.add_geometry(frustum_line_set)

    # 调整点大小和线宽
    opt = vis.get_render_option()
    opt.point_size = point_size
    opt.line_width = line_width
    opt.background_color = np.asarray([1, 1, 1])  # 设置背景为白色

    # 可视化
    vis.run()
    vis.destroy_window()




def rotate_pointcloud_around_box_center(sim_area, points, angle, delta, axis='z'):
    """
    使用 PyTorch 实现绕3D框的中心旋转点云。
    
    参数:
    - sim_area: list或array，表示3D框的范围 [x_min, x_max, y_min, y_max, z_min, z_max]
    - points: Nx3的点云坐标张量 (torch.Tensor)
    - angle: 旋转角度（弧度）
    - axis: 旋转轴 ('x', 'y' 或 'z')
    
    返回:
    - 旋转后的点云 (torch.Tensor)
    """
    # 计算3D框的中心
    x_min, x_max, y_min, y_max, z_min, z_max = sim_area
    center = torch.tensor([(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2], dtype=points.dtype).cuda()

    # 将点云平移到框的中心作为原点
    points_centered = points - center

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

    # 应用旋转矩阵
    points_rotated = torch.matmul(points_centered, R.cuda().T)

    # 将点云平移回原位置
    points_rotated = points_rotated + center

    return points_rotated+delta


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
        R = torch.tensor([[np.cos(angle), -np.sin(angle), 0],
                          [np.sin(angle), np.cos(angle), 0],
                          [0, 0, 1]], dtype=torch.float32)
    else:
        raise ValueError("轴应为 'x', 'y' 或 'z'")

    return R.cuda()

def compute_view_matrix(yaw, pitch, roll, camera_target_position, distance=1e-5, up_axis_index=2, use_torch=False):
    # 将欧拉角转换为弧度
    yaw = np.deg2rad(yaw)
    pitch = np.deg2rad(pitch)
    roll = np.deg2rad(roll)

    # 计算旋转矩阵
    if use_torch:
        # 使用 PyTorch
        cos_yaw, sin_yaw = torch.cos(torch.tensor(yaw)), torch.sin(torch.tensor(yaw))
        cos_pitch, sin_pitch = torch.cos(torch.tensor(pitch)), torch.sin(torch.tensor(pitch))
        cos_roll, sin_roll = torch.cos(torch.tensor(roll)), torch.sin(torch.tensor(roll))

        # 构造旋转矩阵（旋转顺序为 Yaw (Z), Pitch (Y), Roll (X)）
        rot_matrix = torch.tensor([
            [cos_yaw * cos_pitch, cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll, cos_yaw * sin_pitch * cos_roll + sin_yaw * sin_roll],
            [sin_yaw * cos_pitch, sin_yaw * sin_pitch * sin_roll + cos_yaw * cos_roll, sin_yaw * sin_pitch * cos_roll - cos_yaw * sin_roll],
            [-sin_pitch, cos_pitch * sin_roll, cos_pitch * cos_roll]
        ], dtype=torch.float32)
    else:
        # 使用 NumPy
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        cos_pitch, sin_pitch = np.cos(pitch), np.sin(pitch)
        cos_roll, sin_roll = np.cos(roll), np.sin(roll)

        # 构造旋转矩阵（旋转顺序为 Yaw (Z), Pitch (Y), Roll (X)）
        rot_matrix = np.array([
            [cos_yaw * cos_pitch, cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll, cos_yaw * sin_pitch * cos_roll + sin_yaw * sin_roll],
            [sin_yaw * cos_pitch, sin_yaw * sin_pitch * sin_roll + cos_yaw * cos_roll, sin_yaw * sin_pitch * cos_roll - cos_yaw * sin_roll],
            [-sin_pitch, cos_pitch * sin_roll, cos_pitch * cos_roll]
        ], dtype=float)

    # 计算相机位置
    camera_position = np.array(camera_target_position, dtype=float) - distance * rot_matrix[:, 2]

    # 构造视图矩阵
    up_vector = np.zeros(3, dtype=float)
    up_vector[up_axis_index] = 1.0

    # 确保z_axis非零向量
    z_axis = (np.array(camera_target_position, dtype=float) - camera_position)
    if np.linalg.norm(z_axis) < 1e-8:
        z_axis = np.array([0, 0, 1], dtype=float)  # 默认向上

    z_axis /= np.linalg.norm(z_axis)

    x_axis = np.cross(up_vector, z_axis)
    if np.linalg.norm(x_axis) < 1e-8:  # 如果x轴的计算结果接近0,0,0
        x_axis = np.array([1, 0, 0], dtype=float)  # 默认向右

    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    if use_torch:
        view_matrix = torch.eye(4, dtype=torch.float32)
        view_matrix[:3, 0] = torch.tensor(x_axis, dtype=torch.float32)
        view_matrix[:3, 1] = torch.tensor(y_axis, dtype=torch.float32)
        view_matrix[:3, 2] = torch.tensor(-z_axis, dtype=torch.float32)
        view_matrix[:3, 3] = torch.tensor(-camera_position, dtype=torch.float32)
    else:
        view_matrix = np.eye(4, dtype=float)
        view_matrix[:3, 0] = x_axis
        view_matrix[:3, 1] = y_axis
        view_matrix[:3, 2] = -z_axis
        view_matrix[:3, 3] = -camera_position

    return view_matrix

def create_view_metrix(yaw, pitch, roll, translate=np.array([.0, .0, .0])):
    yaw, pitch, roll = np.radians(float(yaw)), np.radians(float(pitch)), np.radians(360-float(roll))
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




def denoise_3d_point_cloud(point_cloud):
    X = point_cloud[:, :3]  # 使用所有三维点进行拟合
    Y = np.ones(X.shape[0])  # 平面方程的常数项作为输出

    # 使用RANSAC进行三维平面拟合
    ransac = RANSACRegressor(residual_threshold=0.1)
    ransac.fit(X, Y)

    # 获取内点和外点的掩码
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)

    # 根据掩码获取内点和外点
    inliers = point_cloud[inlier_mask]
    outliers = point_cloud[outlier_mask]

    return inliers, outliers

def statistical_outlier_removal(point_cloud, nb_neighbors=10, std_ratio=0.05):
    # 将点云数据转换为Open3D的点云格式
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    
    # 统计离群点去除
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    
    inlier_cloud = pcd.select_by_index(ind)
    outlier_cloud = pcd.select_by_index(ind, invert=True)

    return inlier_cloud, outlier_cloud

# 可视化点云
def visualize_3d_point_cloud(inliers, outliers):
    inlier_cloud = o3d.geometry.PointCloud()
    inlier_cloud.points = o3d.utility.Vector3dVector(inliers)
    inlier_cloud.paint_uniform_color([0, 1, 0])  # 绿色表示内点

    outlier_cloud = o3d.geometry.PointCloud()
    outlier_cloud.points = o3d.utility.Vector3dVector(outliers)
    outlier_cloud.paint_uniform_color([1, 0, 0])  # 红色表示外点
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

    # inliers.paint_uniform_color([0, 1, 0])  # 绿色表示内点
    # outliers.paint_uniform_color([1, 0, 0])  # 红色表示外点
    # o3d.visualization.draw_geometries([inliers, outliers])

def visualize_points_in_3D(points_in_3D):
    """
    可视化 3D 点云数据
    :param points_in_3D: 形状为 (H, W, 3) 的 tensor
    """
    # 将 tensor 转换为 numpy 数组
    points_in_3D_np = points_in_3D.cpu().numpy()

    # 展平为 (N, 3) 的数组
    points_in_3D_flat = points_in_3D_np.reshape(-1, 3)

    # 创建 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_in_3D_flat)

    # 可视化点云
    o3d.visualization.draw_geometries([pcd])    
   
