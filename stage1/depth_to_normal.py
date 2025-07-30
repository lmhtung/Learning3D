import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import argparse

class DepthMap:
    def __init__(self, image_path : str):
        """
        image_path (str): path of depth-map picture
        """
        depth_map = Image.open(image_path).convert('L')
        d = transforms.ToTensor()(depth_map).float()
        self.depthMap = d.squeeze(0)
        self.h = self.depthMap.shape[0]
        self.w = self.depthMap.shape[1]
        self.normalMap = torch.zeros(3, self.h, self.w)
    
    def toNormal(self, output : str):
        for i in range (1, self.h-1):
            for j in range(1, self.w -1):
                dzdx = (self.depthMap[i+1, j] - self.depthMap[i-1, j]) / 0.0002
                dzdy = (self.depthMap[i, j+1] - self.depthMap[i, j-1]) / 0.0002
                normal = torch.tensor([-dzdx, -dzdy, 1.0])
                normal = normal / normal.norm()
                normal = (normal +1) / 2 # 
                self.normalMap[:, i, j] = normal
        normal_img = Image.fromarray((self.normalMap.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        normal_img.save(output)
    def view(self, v:str = 'n'):
        if v == 'y':
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            # Depth map
            axs[0].imshow(self.depthMap.numpy(), cmap='gray')
            axs[0].set_title('Depth Map')
            axs[0].axis('off')
            # Normal map
            axs[1].imshow(self.normalMap.permute(1, 2, 0).numpy())
            axs[1].set_title('Normal Map')
            axs[1].axis('off')

            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert depth map to normal map")
    parser.add_argument("--input", type=str, help="Path to depth map image")
    parser.add_argument("--output", type=str, default="normalmap.png", help="Output path for normal map image (default: normal_map.png)",)
    parser.add_argument("--view", type=str, choices=['y', 'n'], default='y', help="Visualize output (y/n)")
    args = parser.parse_args()

    depthmap = DepthMap(args.input)
    depthmap.toNormal(args.output)
    depthmap.view(args.view)
