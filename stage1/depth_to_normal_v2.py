import torch
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import argparse

class DepthToNormalMap:
    def __init__(self, image_path: str):
        # Đọc ảnh và chuyển sang tensor
        depth_map = Image.open(image_path).convert('L')
        depth_tensor = transforms.ToTensor()(depth_map).float()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.depthMap = depth_tensor.to(self.device).squeeze(0)  # (H, W)
        self.h, self.w = self.depthMap.shape
        self.normalMap = torch.zeros(3, self.h, self.w, device=self.device)

    def compute_normal_map(self):
    # 1. Chuẩn hóa depth map
        self.depthMap = (self.depthMap - self.depthMap.min()) / (self.depthMap.max() - self.depthMap.min() + 1e-8)
        depth = self.depthMap.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

    # 2. Sobel kernel
        sobel_x = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=torch.float32, device=self.device).view(1, 1, 3, 3) / 8.0
        sobel_y = torch.tensor([[-1, -2, -1],
                            [ 0,  0,  0],
                            [ 1,  2,  1]], dtype=torch.float32, device=self.device).view(1, 1, 3, 3) / 8.0

    # 3. Tính gradient
        dzdx = F.conv2d(depth, sobel_x, padding=1).squeeze()
        dzdy = F.conv2d(depth, sobel_y, padding=1).squeeze()

    # 4. Scale gradient mạnh để rõ ràng hơn
        scale = 50.0
        nx = -dzdx * scale
        ny = -dzdy * scale
        nz = torch.ones_like(nx)

    # 5. Chuẩn hóa và tăng tương phản normal map
        normal = torch.stack([nx, ny, nz], dim=0)
        normal = normal / torch.norm(normal, dim=0, keepdim=True)
        normal = (normal + 1) / 2  # Đưa về [0,1] để hiển thị
        normal = torch.pow(normal, 0.5)  # Gamma correction
        normal = torch.clamp(normal, 0, 1)

        self.normalMap = normal



    def save(self, output_path: str):
        normal_np = (self.normalMap.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        normal_img = Image.fromarray(normal_np)
        normal_img.save(output_path)

    def view(self, show: str = 'y'):
        if show == 'y':
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            axs[0].imshow(self.depthMap.cpu().numpy(), cmap='gray')
            axs[0].set_title('Depth Map')
            axs[0].axis('off')

            axs[1].imshow(self.normalMap.permute(1, 2, 0).cpu().numpy())
            axs[1].set_title('Normal Map')
            axs[1].axis('off')

            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert depth map to normal map (fast version)")
    parser.add_argument("--input", type=str, required=True, help="Path to depth map image")
    parser.add_argument("--output", type=str, default="normal_map.png", help="Output path for normal map image")
    parser.add_argument("--view", type=str, choices=['y', 'n'], default='y', help="Show maps (y/n)")
    args = parser.parse_args()

    converter = DepthToNormalMap(args.input)
    converter.compute_normal_map()
    converter.save(args.output)
    converter.view(args.view)
