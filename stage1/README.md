# Learning 3D

## Assignment 0: Convert Depth to Normal Map and vice versa
  Depth -> Normal map: <br>
For example, using the command line:  
```bash
python .\depth_to_normal.py --input B:\FIL\3D\depthmap.png --output normalmap_2.png --view y
```
  Normal -> Depth map: <br>
For example, using the command line:
```bash
python .\normal_to_depth.py --input B:\FIL\3D\normalmap\normalmap3.png --output B:\FIL\3D\depthmap\depthmap3_reverted.png --view y
```
## Assignment 1: Rendering First Object
  Setting up for Windows with Conda: <br>
  Install Build Tools: https://visualstudio.microsoft.com/visual-cpp-build-tools/ <br>
  Choose: Desktop development with C++
```bash
conda create -n learning3d python=3.10 
conda activate learning3d 
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0  pytorch-cuda=11.8 -c pytorch -c nvidia
pip install fvcore iopath
conda install numpy matplotlib imageio scikit-image plotly
pip install opencv-python
pip install black usort flake8 flake8-bugbear flake8-comprehensions
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
python setup.py install
```
  Setting up for Linux with Conda: <br>
```
conda create -n learning3d python=3.10 
conda activate learning3d 
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0  pytorch-cuda=11.8 -c pytorch -c nvidia
pip install fvcore iopath
conda install numpy matplotlib imageio scikit-image plotly
pip install opencv-python
MAX_JOBS=8 pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
pip install -r requirements.txt
```
  Setting up with pip:
```
python3.10 -m venv learning3d 
source learning3d/bin/activate # Linux
learning3d\Scripts\activate.bat # Window cmd
learning3d\Scripts\Activate.ps1 # Window powershell
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install fvcore iopath
pip install numpy matplotlib imageio scikit-image plotly
pip install opencv-python
MAX_JOBS=8 pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
pip install -r requirements.txt
```
### Create a mesh
```bash 
vertices, face_props, text_props = pytorch3d.io.load_obj("data/cow.obj") # Load object.obj from folder 
faces = face_props.verts_idx # Getting vertices index (Including 3 vertices)
vertices = vertices.unsqueeze(0)  # 1 x N_v x 3
faces = faces.unsqueeze(0)  # 1 x N_f x 3

texture_rgb = torch.ones_like(vertices) # N x 3
texture_rgb = texture_rgb * torch.tensor([1.0, 1.0, 0]) # RGB color coefficient such as [1.0, 1.0, 0]: yellow 
textures = pytorch3d.renderer.TexturesVertex(texture_rgb) # Creating texture for each pixel
print(texture_rgb)
print(texture_rgb.shape)

meshes = pytorch3d.structures.Meshes(
    verts = vertices,
    faces = faces,
    textures = textures,
)
meshes = meshes.to(device)
``` 
### Setting up camera
``` bash
R = torch.eye(3).unsqueeze(0) # Creating rotation matrix E for rotation 
T = torch.tensor([[0, 0, 3]]) # Translation vector 
fov = 60
cameras = pytorch3d.renderer.FoVPerspectiveCameras(
    R = R,
    T = T,
    fov = fov,
    device = device,
)

cameras.get_camera_center() #

transform = cameras.get_world_to_view_transform()
transform.get_matrix()
```
*References: http://www.codinglabs.net/article_world_view_projection_matrix.aspx <br>
### Setting up pipeline renderer 
``` bash 
image_size = 512

raster_settings = pytorch3d.renderer.RasterizationSettings(image_size = image_size)
rasterizer = pytorch3d.renderer.MeshRasterizer(
    raster_settings = raster_settings,
)
shader = pytorch3d.renderer.HardPhongShader(device = device)
renderer = pytorch3d.renderer.MeshRenderer(
    rasterizer = rasterizer,
    shader = shader,
)
```
### Render an image 2D (+lights)
``` bash
lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)
image = renderer(meshes, cameras=cameras, lights=lights)
plt.imshow(image[0].cpu().numpy())
```
<img width="856" height="821" alt="image" src="https://github.com/user-attachments/assets/07ba6836-ee60-469a-b7b1-f19208536895" /> <br>

Plot scene: <br>
```bash
plot_scene({
    "figure": {
        "Mesh": meshes,
        "Camera": cameras,
    }
})
```
<img width="778" height="671" alt="image" src="https://github.com/user-attachments/assets/3b7b494d-cd78-4b45-b3dd-6ee48f60c035" /> <br>
### Render an gif 
For example: 
``` bash
python render_360.py --num_frames 60
```
![cow_gif](https://github.com/user-attachments/assets/9ac5f0bc-104e-4a71-8339-04ac7756fe7f) <br>

