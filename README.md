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
  Set up: <br>
```bash
conda create -n learning3d python=3.10 <br>
conda activate learning3d <br>
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0  pytorch-cuda=11.8 -c pytorch -c nvidia
```
  Normal -> Depth map: <br>
For example, using the command line:
```bash
python .\normal_to_depth.py --input B:\FIL\3D\normalmap\normalmap3.png --output B:\FIL\3D\depthmap\depthmap3_reverted.png --view y
```
