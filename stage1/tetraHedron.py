import imageio
import numpy as np
import pytorch3d
import torch
import pytorch3d.io
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from pytorch3d.structures import Meshes
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

def _render(image_size = 512, output = "output/tetrahedron.png"):

    R = torch.eye(3).unsqueeze(0)
    T = [[0, 0, 3]]
    cameras = FoVPerspectiveCameras( R= R,
                                     T=T,
                                     fov = 60,
                                     device =device)
    vertices = torch.tensor([
    [1, 1, 1],     # Vertex 0
    [-1, -1, 1],   # Vertex 1
    [-1, 1, -1],   # Vertex 2
    [1, -1, -1],   # Vertex 3
    ], dtype=torch.float32, device=device)
    vertices = vertices.unsqueeze(0)
    faces = torch.tensor([
    [0, 1, 2],  # Face 0
    [0, 3, 1],  # Face 1
    [0, 2, 3],  # Face 2
    [1, 3, 2],  # Face 3
    ], dtype=torch.int64, device=device)
    faces = faces.unsqueeze(0)
    texture_rgb = torch.ones_like(vertices) # N X 3
    texture_rgb = texture_rgb * torch.tensor([1.0, 1.0, 0], device= device) #[0.7, 0.7, 0.1] hệ số màu rgb 
    textures = TexturesVertex(texture_rgb) # important

    meshes = Meshes(verts=vertices,
                    faces=faces,
                    textures= textures)
    meshes = meshes.to(device)
    lights  = PointLights (location = [[0, 0, -3]] ,device = device)

    raster_settings = RasterizationSettings(image_size = image_size)
    rasterizer = MeshRasterizer(raster_settings = raster_settings)
    shader = SoftPhongShader(device = device)

    renderer = MeshRenderer(rasterizer = rasterizer, shader = shader)

    image = renderer(meshes, cameras= cameras, lights=lights)
    plt.imshow(image[0].cpu().numpy())

_render()