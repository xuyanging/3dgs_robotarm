import json
import torch
import numpy as np
from sklearn.cluster import DBSCAN

def load_json_recursive(file_path):
    # 读取 JSON 文件
    with open(file_path, 'r') as file:
        config = json.load(file)
    
    # 创建一个字典用于保存加载的引用配置
    additional_config = {}

    # 检查是否有引用的 JSON 文件
    for key, value in list(config.items()):  # 使用 list(config.items()) 防止迭代过程中修改字典大小
        if isinstance(value, str) and value.endswith('.json'):
            # 递归读取引用的 JSON 文件
            referenced_config = load_json_recursive(value)
            # 将引用的内容添加到附加配置字典中
            additional_config.update(referenced_config)
    
    # 合并附加配置到原始配置
    config.update(additional_config)

    return config

def point_cloud_filter(points: torch.Tensor, eps=0.1, min_points=10, min_cluster_size=100):
    """
    对输入的Nx3的点云数据进行DBSCAN聚类，并返回剔除无用点的掩码。
    
    参数:
    - points: 输入的点云数据，形状为 (N, 3)，类型为 torch.Tensor。
    - eps: DBSCAN算法中的邻域半径。
    - min_points: DBSCAN算法中的每个簇的最小点数。
    - min_cluster_size: 保留的最小聚类大小，小于此大小的簇将被剔除。

    返回:
    - mask: 剔除无用点后的掩码，形状为 (N, 1)，类型为 torch.Tensor。
    """
    # 将点云转换为numpy数组以使用scipy的DBSCAN
    points_np = points.detach().cpu().numpy()

    # 使用DBSCAN进行聚类
    db = DBSCAN(eps=eps, min_samples=min_points).fit(points_np)
    labels = db.labels_

    # 统计每个簇的大小
    unique, counts = np.unique(labels, return_counts=True)
    cluster_sizes = dict(zip(unique, counts))

    # 创建一个掩码，标记所有需要保留的点
    mask = torch.zeros(points.size(0), dtype=torch.bool)

    for label in cluster_sizes:
        if label == -1:
            # -1 表示噪声点
            continue
        if cluster_sizes[label] >= min_cluster_size:
            # 保留点数大于等于 min_cluster_size 的簇
            mask[torch.from_numpy(np.where(labels == label)[0])] = True

    return mask.cuda()