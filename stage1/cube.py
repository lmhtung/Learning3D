import torch
import imageio
import numpy as np
from tqdm import tqdm
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras, PointLights,
    RasterizationSettings, MeshRasterizer,
    HardPhongShader, MeshRenderer, TexturesVertex,
    look_at_view_transform
)
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_cube():
    verts = torch.tensor([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ], dtype=torch.float32)
    verts = verts - 0.5
    faces = torch.tensor([
        [0, 1, 2], [0, 2, 3],
        [4, 5, 6], [4, 6, 7],  
        [0, 1, 5], [0, 5, 4], 
        [1, 2, 6], [1, 6, 5],  
        [2, 3, 7], [2, 7, 6],  
        [3, 0, 4], [3, 4, 7],  
    ], dtype=torch.int64)


    verts = verts.unsqueeze(0).to(device)
    faces = faces.unsqueeze(0).to(device)
    texture_rgb = torch.ones_like(verts) 
    texture_rgb = texture_rgb * torch.tensor([0.1, 0.7, 1], device= device)
    textures = TexturesVertex(texture_rgb)
    mesh = Meshes(verts=verts, faces=faces, textures=textures)
    return mesh

def get_renderer(image_size=512):
    lights = PointLights(device=device, location=[[2, 2, 2]])
    raster_settings = RasterizationSettings(image_size=image_size)
    rasterizer = MeshRasterizer(raster_settings=raster_settings)
    shader = HardPhongShader(device=device, lights=lights)
    renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)
    return renderer

def render_cube(output_path="cube.gif", num_frames=36, image_size=256):
    mesh = create_cube()
    renderer = get_renderer(image_size=image_size)
    images = []

    angles = torch.linspace(0, 360, steps=num_frames)

    for azim in tqdm(angles):
        R, T = look_at_view_transform(dist=2.7, elev=0.0, azim=azim.item())
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
        image = renderer(mesh, cameras=cameras)
        img = image[0, ..., :3].cpu().numpy()
        img = (img * 255).astype(np.uint8)
        images.append(img)

    imageio.mimsave(output_path, images, duration=0.1)
    print(f"saved gif to: {output_path}")

render_cube()
