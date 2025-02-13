import numpy as np

# 原始 view_matrix
view_matrix = np.array([
    [1.00000000e+00, 0.00000000e+00, -0.00000000e+00, 0.00000000e+00],
    [0.00000000e+00, 0.00000000e+00, -1.00000000e+00, 0.00000000e+00],
    [0.00000000e+00, 1.00000000e+00, -0.00000000e+00, 0.00000000e+00],
    [-0.00000000e+00, -0.00000000e+00, -9.99999975e-06, 1.00000000e+00]
])

# 给定的 view_matrix_mirror
view_matrix_mirror = np.array([
    [1.00000000e+00, 0.00000000e+00, -0.00000000e+00, 0.00000000e+00],
    [-0.00000000e+00, 0.00000000e+00, -1.00000000e+00, 0.00000000e+00],
    [0.00000000e+00, 1.00000000e+00, 0.00000000e+00, 0.00000000e+00],
    [-0.00000000e+00, -0.00000000e+00, -9.99999975e-06, 1.00000000e+00]
])

# 求 view_matrix 的逆矩阵
view_matrix_inv = np.linalg.inv(view_matrix)

# 计算变换矩阵 T
T = view_matrix_mirror @ view_matrix_inv

# 打印结果
print("Transformation matrix T (view_matrix_mirror @ inv(view_matrix)):")
print(T)
