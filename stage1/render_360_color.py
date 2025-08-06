import torch
import numpy as np
import pytorch3d
import pytorch3d.io
from pytorch3d.renderer import (look_at_view_transform, 
                                FoVPerspectiveCameras,
                                TexturesVertex,
                                HardPhongShader,
                                SoftPhongShader,
                                MeshRenderer,
                                MeshRasterizer,
                                RasterizationSettings,
                                PointLights)
from pytorch3d.structures import Meshes
import imageio
import argparse

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def set_cam(dist = 1.0, elev = 0.0, azim = 30, degrees = True):
    R, T = look_at_view_transform(dist= dist, elev= elev, azim= azim, degrees= degrees)
    cameras = FoVPerspectiveCameras(R= R, T= T, device= device)
    return cameras

def load_mesh(path:str):
    color1 = torch.tensor([1.0, 0.0, 0.0])
    color2 = torch.tensor([0.0, 0.0, 1.0])
    vertices, face, textures = pytorch3d.io.load_obj("data/cow.obj")
    z = vertices[..., 2]
    z_min = z.min()
    z_max = z.max()
    alpha = (z - z_min) / (z_max - z_min)
    vertex_colors = (1 - alpha).unsqueeze(1) * color1 + alpha.unsqueeze(1) * color2
    vertex_colors = vertex_colors.unsqueeze(0) 

    faces = face.verts_idx
    vertices = vertices.unsqueeze(0) # 1 x n x3
    faces = faces.unsqueeze(0) # 1 x n x 3
    
    textures = TexturesVertex(vertex_colors) # important
    meshes = Meshes(verts = vertices, faces = faces, textures = textures)
    meshes = meshes.to(device)
    return meshes

def render_image( path:str , azim = 20, dist =3.0, elev = 0.0 , image_size = 512):
    lights = PointLights(location=[[0, 0, -3]], device=device)
    raster_settings = RasterizationSettings(image_size = image_size)
    rasterizer = MeshRasterizer(raster_settings = raster_settings)
    shader = SoftPhongShader(device = device)
    renderer = MeshRenderer(rasterizer = rasterizer, shader = shader)
    cameras = set_cam(dist= dist , azim= azim, elev= elev)
    meshes = load_mesh(path= path)
    image = renderer(meshes, cameras= cameras, lights=lights)
    return image

def render_360(path :str, num_frames = 12, dist = 3.0, elev = 0.0, output = "output/cow_color.gif"):
    images = []
    angles = torch.linspace(0, 360, steps = num_frames, dtype=torch.float32)
    for a in angles:
        image = render_image(path= path, azim= a, dist= dist, elev =elev)
        i = image[0, ..., :3].cpu().numpy()
        i = (i * 255).astype(np.uint8)
        images.append(i)

    duration = 1000 // 15  # Convert FPS (frames per second) to duration (ms per frame)
    imageio.mimsave(output, images, duration=duration)

if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description="Rendering gif 360")
    parser.add_argument("--input", type=str, default= "data/cow.obj", help= "Path to object.obj")
    parser.add_argument("--output", type=str, default= "output/cow_color.gif", help="Output path for gif")
    parser.add_argument("--num_frames", type= int, default = 30, help="Number of frames")
    parser.add_argument("--dist", type= float, default = 3.0, help="Distance between object and camera")
    parser.add_argument("--elev", type= float, default = 0.0, help="The angles between the vector from the object to camera and horizontal plane")
    args = parser.parse_args()
    render_360(args.input, args.num_frames, args.dist, args.elev, args.output)