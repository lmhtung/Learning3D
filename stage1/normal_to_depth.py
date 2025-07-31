import torch
import torch.fft
from PIL import Image
from torchvision import transforms
import numpy as np
from torch.nn import functional as F
import matplotlib.pyplot as plt
import argparse

'''
        Sử dụng phương pháp Poisson div(p,q) = dp/dx + dq/dy + 0
            Tính Fourier bằng công thức: 
                    F(div_pq) = -4 * pi^2 * (u^2 + v^2) * F(z(x,y))
            Mục tiêu cần tính là z(x,y) 
            Từ đó, tính hàm ngược của F(z(x,y)) và ta được:
                    z(x,y) = F^-1( F(div_pq) / (4 * (sin^2(pi * u)+ sin^2(pi * v))))
'''
class NormalToDepth:
    def __init__ (self, path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        normal_map = Image.open(path).convert('RGB')
        Map = transforms.ToTensor()(normal_map).to(self.device)
        Map = Map*2 - 1 # Scale giá trị về [-1,1]
        # n = (n_x, n_y, n_z) tương ứng (R, G, B) khi chuyển từ Depth sang Normal
        n_x, n_y, n_z = Map[0], Map[1], Map[2]
        self.h, self.w = Map.shape[1], Map.shape[2]
        self.normalMap = normal_map
        self.depthMap = torch.zeros(1, self.h, self.w, device=self.device)
        self.p = -n_x / n_z
        self.q = -n_y / n_z
        
    def toDepthMap(self, scale = 0.5):
        ''' 
            Tính đạo hàm, sau đó:
            Thêm pad(input, pad, mode='constant', value=0) để đảm bảo không sai về kích thước ảnh
                        (left, right, top, bottom)
        '''
        dp_dx = self.p[:, 1:] - self.p[:, :-1] 
        dp_dx = F.pad(dp_dx, (1,0))
        dq_dy = self.q[1:, :] - self.q[:-1, :]
        dq_dy = F.pad(dq_dy, (0,0,1,0))
        div_pq = dp_dx + dq_dy
            
        # Tính z(x,y)
        freq_y = torch.fft.fftfreq(self.h, device = self.device).reshape(self.h, 1)
        freq_x = torch.fft.fftfreq(self.w, device = self.device).reshape(1, self.w)
        denom = 4 * (torch.sin(np.pi * freq_x) ** 2 + torch.sin(np.pi * freq_y) ** 2)
        denom[0,0] = 1e-8
        div_fft = torch.fft.fft2(div_pq)
        depth_fft = div_fft / denom
        depth = torch.fft.ifft2(depth_fft).real # Lấy phần thực
            
        # Chuẩn hóa Depth Map
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
        depth = 1.0 - depth
        depth = torch.clamp(depth * scale, 0.0, 1.0)
        # Lưu ảnh 
        depth_img_tensor = (depth * 255.0).to(torch.uint8).unsqueeze(0)  # Shape [1, H, W]
        self.depthMap = transforms.ToPILImage()(depth_img_tensor.cpu())

    def save(self, output: str):
        self.depthMap.save(output)
    def view(self, show:str ='y'):
        if show == 'y':
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            # Hiển thị Normal Map
            axs[0].imshow(self.normalMap)
            axs[0].set_title('Normal Map')
            axs[0].axis('off')
            # Hiển thị Depth Map
            axs[1].imshow(np.array(self.depthMap), cmap ='gray')
            axs[1].set_title('Depth Map')
            axs[1].axis('off')

            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert depth map to normal map (fast version)")
    parser.add_argument("--input", type=str, required=True, help="Path to depth map image")
    parser.add_argument("--output", type=str, default="depth_map_reverted.png", help="Output path for normal map image")
    parser.add_argument("--view", type=str, choices=['y', 'n'], default='y', help="Show maps (y/n)")
    args = parser.parse_args()

    converter = NormalToDepth(args.input)
    converter.toDepthMap()
    converter.save(args.output)
    converter.view(args.view)
