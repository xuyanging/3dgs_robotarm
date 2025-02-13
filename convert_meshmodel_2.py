import trimesh
import pymeshlab

# 读取 PLY 文件
mesh = trimesh.load("sculpture_custom_nw_iterations30000_DLNR_Middlebury_baseline7_0p_mask0_occ1_scale1_0_voxel2_512_trunc4_20_cleaned_mesh.ply")

# 创建一个 pymeshlab project
ms = pymeshlab.MeshSet()

# 将 trimesh 对象转换为 pymeshlab Mesh
mesh_data = pymeshlab.Mesh(vertex_matrix=mesh.vertices, 
                           face_matrix=mesh.faces,
                           v_rgb_color_matrix=mesh.visual.vertex_colors[:, :3])  # 取前3列作为RGB颜色

# 将 mesh_data 添加到 pymeshlab 中
ms.add_mesh(mesh_data)

# 保存为 OBJ 文件并保留颜色
ms.save_current_mesh("sculpture_custom_nw_iterations30000_DLNR_Middlebury_baseline7_0p_mask0_occ1_scale1_0_voxel2_512_trunc4_20_cleaned_mesh.obj", save_vertex_color=True)