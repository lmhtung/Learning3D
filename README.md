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
  Set up for Windows: <br>
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
  Set up for Ubuntu: <br>
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
