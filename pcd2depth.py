import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def generate_depth_map(points, intrinsic, width, height, depth_scale=1, depth_max=50.0):
    # 提取内参
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    
    # 初始化深度图
    depth_map = np.full((height, width), np.inf)
    valid_points = []
    depth_values = []
    
    for point in points:
        x, y, z = point
        
        # 投影到图像平面
        u = int((x * fx) / z + cx)
        v = int((y * fy) / z + cy)
        
        if 0 <= u < width and 0 <= v < height and z > 0 and z < depth_max:
            depth_value = z * depth_scale
            depth_map[v, u] = min(depth_map[v, u], depth_value)
            valid_points.append([u, v])
            depth_values.append(depth_value)
    
    valid_points = np.array(valid_points)
    depth_values = np.array(depth_values)
    
    # 使用 griddata 插值
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    interpolated_depth_map = griddata(valid_points, depth_values, (grid_x, grid_y), method='linear', fill_value=depth_max * depth_scale)

    return interpolated_depth_map

file_path = 'model_file/3dgs/6e03a0a9-4/point_cloud/iteration_30000/point_cloud.ply'

pcd_legacy = o3d.io.read_point_cloud(file_path)
points = np.asarray(pcd_legacy.points)

intrinsic = np.array([[1.03549660e+03, 0.00000000e+00, 6.66000000e+02],
                     [0.00000000e+00, 1.03497186e+03, 4.38000000e+02],
                     [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

depth_image = generate_depth_map(points, intrinsic, 1332, 876)
depth = o3d.geometry.Image(depth_image)
pcld = o3d.geometry.PointCloud.create_from_depth_image(depth, intrinsic, depth_scale=1, depth_trunc=50)

# 使用matplotlib保存深度图
plt.imsave('depth_map_interpolated.png', depth_image, cmap='gray')

# 显示深度图
plt.imshow(depth_image, cmap='gray')
plt.show()
