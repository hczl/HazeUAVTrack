import cv2
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt

# === Step 1: 读取图像并灰度化 ===
img = cv2.imread('img000010.jpg')  # 替换为你自己的图像路径
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# === Step 2: 提取FAST角点 ===
fast = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
keypoints = fast.detect(gray, None)
corners = np.array([kp.pt for kp in keypoints], dtype=np.float32)

# === Step 3: 模拟前几帧检测框（可换成你实际的框） ===
# 格式：(x1, y1, x2, y2)
prev_boxes = []

# === Step 4: 框内角点加权 ===
weights = np.ones(len(corners))
for i, (x, y) in enumerate(corners):
    for x1, y1, x2, y2 in prev_boxes:
        if x1 <= x <= x2 and y1 <= y <= y2:
            weights[i] *= 1.5  # 增加权重

# === Step 5: Mean-Shift 聚类 ===
bandwidth = estimate_bandwidth(corners, quantile=0.2, n_samples=500)
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
replicated = []
for i, pt in enumerate(corners):
    repeat = int(weights[i])  # 权重 1.5 会变为1
    for _ in range(repeat):
        replicated.append(pt)
replicated = np.array(replicated)

ms.fit(replicated)
cluster_centers = ms.cluster_centers_

# === Step 6: 可视化（聚类框可作为候选区域） ===
output = img.copy()
for pt in corners:
    cv2.circle(output, tuple(int(x) for x in pt), 1, (0, 255, 0), -1)
for center in cluster_centers:
    cx, cy = int(center[0]), int(center[1])
    cv2.rectangle(output, (cx-30, cy-30), (cx+30, cy+30), (0, 0, 255), 2)

# === 显示图像 ===
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title("Corners & Mean-Shift Candidate Boxes")
plt.axis("off")
plt.show()
