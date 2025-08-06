import imageio
import numpy as np
import pytorch3d
import torch
import pytorch3d.io
from PIL import Image, ImageDraw
from tqdm.auto import tqdm
import math
from pytorch3d.renderer import (look_at_view_transform, 
                                FoVPerspectiveCameras,
                                TexturesVertex,
                                HardPhongShader,
                                SoftPhongShader,
                                MeshRenderer,
                                MeshRasterizer,
                                RasterizationSettings,
                                PointLights)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def _mesh_render(image_size = 512, lights = None, blur = None):
    '''
        bin_size:           ảnh được chia thành các ô, bin_size = n ảnh được chia thành n ô, 
                            bin_size = None ảnh được chia tự động theo image_size
        max_faces_per_bin:  số face tối đa trên 1 bin, max_faces_per_bin = None số face tối đa tự set
    '''
    blur_radius = blur if blur else 0
    raster_set = RasterizationSettings(image_size = image_size, blur_radius = blur_radius, faces_per_pixel = 1, bin_size = None)
    rasterizer = MeshRasterizer(raster_settings= raster_set)
    shader = HardPhongShader(device = device, lights = lights) # soft hoặc hard (+ lights)
    renderer = MeshRenderer(rasterizer = rasterizer, shader= shader ) # rasterizer + shader
    return renderer

def dolly_effect(image_size = 512, path :str = "data/cow_on_plane.obj", device = device, blur = None,
                 num_frames = 30, duration =3, output = "output/dolly.gif"):
    lights = PointLights(location= [[0, 0, 3]], device= device)
    blur_radius = blur if blur else 0
    meshes = pytorch3d.io.load_objs_as_meshes([path])
    meshes = meshes.to(device)

    mesh_render = _mesh_render(image_size= image_size, lights= lights, blur= blur_radius)
    fovs = torch.linspace(5, 120, num_frames)

    renders = []
    for fov in fovs:
        distance  = 5 / (2* math.tan(math.radians(fov /2)))
        T = [[0, 0, distance]]
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(fov=fov, T=T, device=device)
        rend = mesh_render(meshes, cameras=cameras, lights=lights)
        rend = rend[0, ..., :3].cpu().numpy()  # (N, H, W, 3)
        renders.append(rend)

    images = []
    for i, r in enumerate(renders):
        image = Image.fromarray((r * 255).astype(np.uint8))
        draw = ImageDraw.Draw(image)
        draw.text((20, 20), f"fov: {fovs[i]:.2f}", fill=(255, 0, 0))
        images.append(np.array(image))
    imageio.mimsave(output, images, duration=duration)

if __name__ == "__main__":
    dolly_effect()