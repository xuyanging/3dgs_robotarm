from plyfile import PlyData
import numpy as np
import trimesh
from sklearn.cluster import KMeans

# 读取 .ply 文件
ply_data = PlyData.read('env_mesh_model/fuse_0.01_v2.ply')

# 提取顶点和面数据
vertices = np.vstack([ply_data['vertex'].data['x'],
                      ply_data['vertex'].data['y'],
                      ply_data['vertex'].data['z']]).T

faces = np.vstack(ply_data['face'].data['vertex_indices'])

# 检查并移除无效的面片索引
if np.max(faces) >= vertices.shape[0]:
    print("Error: Face index exceeds vertex array bounds. Filtering invalid faces.")
    faces = faces[np.all(faces < vertices.shape[0], axis=1)]

# 提取顶点颜色（假设颜色存在）
vertex_colors = np.vstack([ply_data['vertex'].data['red'],
                           ply_data['vertex'].data['green'],
                           ply_data['vertex'].data['blue']]).T

# 确保 vertex_colors 大小匹配 vertices
if vertex_colors.shape[0] != vertices.shape[0]:
    print("Warning: Vertex colors and vertices count mismatch. Truncating or expanding vertex colors.")
    vertex_colors = vertex_colors[:vertices.shape[0]]

# 创建 Trimesh 对象
ply_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=vertex_colors)

# 对模型进行简化，减少顶点数目
simplified_ply_mesh = ply_mesh.simplify_quadric_decimation(100000)  # 目标是减少到10000个面

# 计算每个面的平均颜色
face_colors = np.mean(vertex_colors[simplified_ply_mesh.faces], axis=1).astype(np.uint8)

# 聚类颜色，减少颜色数量
num_colors = 10  # 目标颜色种类数量
kmeans = KMeans(n_clusters=num_colors)
kmeans.fit(face_colors)
compressed_colors = kmeans.cluster_centers_[kmeans.labels_].astype(np.uint8)

# 创建最终的 Trimesh 对象
obj_mesh = trimesh.Trimesh(vertices=simplified_ply_mesh.vertices, faces=simplified_ply_mesh.faces, face_colors=compressed_colors)

# 导出到 .obj 文件
obj_mesh.export('env_mesh_model/output_compressed.obj')

print("Model has been simplified and exported as 'output_compressed.obj'")
