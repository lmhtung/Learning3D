import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_voxel_matplotlib(voxel, threshold=0.7, filename="voxel.png"):
    """
    Render voxel bằng matplotlib (3D scatter)
    voxel: torch.Tensor [B,H,W,D] hoặc [H,W,D]
    threshold: ngưỡng để xác định voxel có / không
    filename: tên file ảnh để lưu
    """

    # bỏ batch dimension nếu có
    if voxel.dim() == 4:
        voxel = voxel[0]

    voxel_np = voxel.detach().cpu().numpy()
    coords = np.argwhere(voxel_np > threshold)  # (N,3)

    if coords.shape[0] == 0:
        print("⚠️ Không có voxel nào được chọn (tất cả <= threshold).")
        return

    # Vẽ scatter 3D
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(coords[:,0], coords[:,1], coords[:,2], 
               c="blue", marker='s', s=20, alpha=0.8)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Voxel (Blue)")

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"✅ Saved voxel scatter plot (blue) -> {filename}")
