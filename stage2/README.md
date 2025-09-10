# 16-825 Assignment 2: Single View to 3D

Goals: In this assignment, you will explore the types of loss and decoder functions for regressing to voxels, point clouds, and mesh representation from single view RGB input. 

## Table of Contents
0. [Setup](#0-setup)
1. [Exploring Loss Functions](#1-exploring-loss-functions)
2. [Reconstructing 3D from single view](#2-reconstructing-3d-from-single-view)
3. [Exploring other architectures / datasets](#3-exploring-other-architectures--datasets-choose-at-least-one-more-than-one-is-extra-credit)
## 0. Setup

Please download and extract the dataset for this assigment. We provide two versions for the dataset, which are hosted on huggingface. 

* [here](https://huggingface.co/datasets/learning3dvision/r2n2_shapenet_dataset) for a single-class dataset which contains one class of chair. Total size 7.3G after unzipping.

Download the dataset using the following commands:

```
$ sudo apt install git-lfs
$ git lfs install
$ git clone https://huggingface.co/datasets/learning3dvision/r2n2_shapenet_dataset
```

* [here](https://huggingface.co/datasets/learning3dvision/r2n2_shapenet_dataset_full) for an extended version which contains three classes, chair, plane, and car.  Total size 48G after unzipping. Download this dataset with the following command:

```
$ git lfs install
$ git clone https://huggingface.co/datasets/learning3dvision/r2n2_shapenet_dataset_full
```

Downloading the datasets may take a few minutes. After unzipping, set the appropriate path references in `dataset_location.py` file [here](dataset_location.py).

The extended version is required for Q3.3; for other parts, using single-class version is sufficient.

Make sure you have installed the packages mentioned in `requirements.txt`.
This assignment will need the GPU version of pytorch.

## 1. Exploring loss functions
This section will involve defining a loss function, for fitting voxels, point clouds and meshes.

### 1.1. Fitting a voxel grid (5 points)
In this subsection, we will define binary cross entropy loss that can help us <b>fit a 3D binary voxel grid</b>.
Define the loss functions `voxel_loss` in [`losses.py`](losses.py) file. 
For this you can use the pre-defined losses in pytorch library.

Run the file `python fit_data.py --type 'vox'`, to fit the source voxel grid to the target voxel grid. 
#### RESULT: 
```
def voxel_loss(voxel_src,voxel_tgt):
	# voxel_src: b x h x w x d
	# voxel_tgt: b x h x w x d
	# loss = 
	# implement some loss for binary voxel grids
	# use BCE binary cross entropy 
	eps = 1e-5 
	voxel_src = torch.clamp(voxel_src, eps, 1 - eps)
	s = voxel_tgt * torch.log(voxel_src) + (1- voxel_tgt)* torch.log(1- voxel_src)
	loss = -1* s.mean()
	return loss
```

Visualize the optimized voxel grid along-side the ground truth voxel grid using the tools learnt in previous section.

### 1.2. Fitting a point cloud (5 points)
In this subsection, we will define chamfer loss that can help us <b> fit a 3D point cloud </b>.
Define the loss functions `chamfer_loss` in [`losses.py`](losses.py) file.
<b>We expect you to write your own code for this and not use any pytorch3d utilities. You are allowed to use functions inside pytorch3d.ops.knn such as knn_gather or knn_points</b>

Run the file `python fit_data.py --type 'point'`, to fit the source point cloud to the target point cloud. 
#### RESULT:
```
def chamfer_loss(point_cloud_src,point_cloud_tgt):
	# point_cloud_src, point_cloud_src: b x n_points x 3  
	# loss_chamfer = 
	# implement chamfer loss from scratch
	src_1 = knn_points(point_cloud_src, point_cloud_tgt, K= 1)
	src_2 = knn_points(point_cloud_tgt, point_cloud_src, K= 1)
	s_1 = src_1.dists.sum()
	s_2 = src_2.dists.sum()
	loss_chamfer = s_1 + s_2
	return loss_chamfer
```

Visualize the optimized point cloud along-side the ground truth point cloud using the tools learnt in previous section.

### 1.3. Fitting a mesh (5 points)
In this subsection, we will define an additional smoothening loss that can help us <b> fit a mesh</b>.
Define the loss functions `smoothness_loss` in [`losses.py`](losses.py) file.

For this you can use the pre-defined losses in pytorch library.

Run the file `python fit_data.py --type 'mesh'`, to fit the source mesh to the target mesh. 
#### RESULT:
```
def smoothness_loss(mesh_src):
	loss_laplacian = mesh_laplacian_smoothing(mesh_src)
	# implement laplacian smoothening loss
	return loss_laplacian
```

Visualize the optimized mesh along-side the ground truth mesh using the tools learnt in previous section.

## 2. Reconstructing 3D from single view
This section will involve training a single view to 3D pipeline for voxels, point clouds and meshes.
Refer to the `save_freq` argument in `train_model.py` to save the model checkpoint quicker/slower. 

We also provide pretrained ResNet18 features of images to save computation and GPU resources required. Use `--load_feat` argument to use these features during training and evaluation. This should be False by default, and only use this if you are facing issues in getting GPU resources. You can also enable training on a CPU by the `device` argument. Also indiciate in your submission if you had to use this argument. 

### 2.1. Image to voxel grid (20 points)
In this subsection, we will define a neural network to decode binary voxel grids.
Define the decoder network in [`model.py`](model.py) file for `vox` type, then reference your decoder in [`model.py`](model.py) file

Run the file `python train_model.py --type 'vox'`, to train single view to voxel grid pipeline, feel free to tune the hyperparameters as per your need.

After trained, visualize the input RGB, ground truth voxel grid and predicted voxel in `eval_model.py` file using:
`python eval_model.py --type 'vox' --load_checkpoint`

You need to add the respective visualization code in `eval_model.py`

On your webpage, you should include visuals of any three examples in the test set. For each example show the input RGB, render of the predicted 3D voxel grid and a render of the ground truth mesh.
#### RESULT:
```
class MLPDecoder(nn.Module):
    def __init__(self, latent_dim= 512, hidden_dim= 256, resolution= 32):
        super(MLPDecoder, self).__init__()
        # self.latent_dim = latent_dim
        # self.hidden_dim = hidden_dim
        self.resolution = resolution

        self.fc1 = nn.Linear(latent_dim +3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim,1)

        x = torch.linspace(-1,1,resolution)
        y = torch.linspace(-1,1,resolution)
        z = torch.linspace(-1,1,resolution)

        grid = torch.stack (torch.meshgrid(x ,y, z, indexing= "ij"), dim= -1)
        self.register_buffer("coords", grid.reshape(-1,3))

    def forward(self, z):
        """
            z: (B, latent_dim) latent vector from encoder
            return: voxel occupancy (B, res, res, res)
        """
        B = z.size(0)
        N = self.coords.size(0)

        # expand z to all coords
        z_expand = z.unsqueeze(1).expand(B, N, z.size(-1))   # (B, N, latent_dim)
        coords_expand = self.coords.unsqueeze(0).expand(B, N, 3)  # (B, N, 3)

        x = torch.cat([coords_expand, z_expand], dim=-1)     # (B, N, latent_dim+3)

        # run through MLP
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))   # occupancy probability
        return x.view(B, self.resolution, self.resolution, self.resolution)

```

### 2.2. Image to point cloud (20 points)
In this subsection, we will define a neural network to decode point clouds.
Similar as above, define the decoder network in [`model.py`](model.py) file for `point` type, then reference your decoder in [`model.py`](model.py) file.

Run the file `python train_model.py --type 'point'`, to train single view to pointcloud pipeline, feel free to tune the hyperparameters as per your need.

After trained, visualize the input RGB, ground truth point cloud and predicted  point cloud in `eval_model.py` file using:
`python eval_model.py --type 'point' --load_checkpoint`

You need to add the respective visualization code in `eval_model.py`.

On your webpage, you should include visuals of any three examples in the test set. For each example show the input RGB, render of the predicted 3D point cloud and a render of the ground truth mesh.
#### RESULT:
```
class PointDecoder(nn.Module):
    def __init__(self, n_points, latent_dim = 512, hidden_dim =512):
        super(PointDecoder, self).__init__()
        self.n_points = n_points
        # self.device = args.device
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim*2)
        self.fc3 = nn.Linear(hidden_dim*2, self.n_points*3)

    def forward(self, z):
        layer1 = F.relu(self.fc1(z))
        layer2 = F.relu(self.fc2(layer1))
        layer3 = torch.tanh(self.fc3(layer2))
        return layer3.view(z.size(0), self.n_points, 3)
```

### 2.3. Image to mesh (20 points)
In this subsection, we will define a neural network to decode mesh.
Similar as above, define the decoder network in [`model.py`](model.py) file for `mesh` type, then reference your decoder in [`model.py`](model.py) file.

Run the file `python train_model.py --type 'mesh'`, to train single view to mesh pipeline, feel free to tune the hyperparameters as per your need. We also encourage the student to try different mesh initializations (i.e. replace `ico_sphere` by other shapes).


After trained, visualize the input RGB, ground truth mesh and predicted mesh in `eval_model.py` file using:
`python eval_model.py --type 'mesh' --load_checkpoint`

You need to add the respective visualization code in `eval_model.py`.

On your webpage, you should include visuals of any three examples in the test set. For each example show the input RGB, render of the predicted mesh and a render of the ground truth mesh.
#### RESULT:
```
class MeshDecoder(nn.Module):
    def __init__(self, n_verts ,latent_dim = 512, hidden_dim= 256):
        super(MeshDecoder, self).__init__()
        self.n_verts = n_verts
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim*2)
        self.fc3 = nn.Linear(hidden_dim*2, self.n_verts *3)
    def forward(self, z):

        layer1 = F.relu(self.fc1(z))
        layer2 = F.relu(self.fc2(layer1))
        layer3 = torch.tanh(self.fc3(layer2))
        return layer3.view(-1, self.n_verts, 3)
```

### 2.4. Quantitative comparisions(10 points)
Quantitatively compare the F1 score of 3D reconstruction for meshes vs pointcloud vs voxelgrids.
Provide an intutive explaination justifying the comparision.

For evaluating you can run:
`python eval_model.py --type voxel|mesh|point --load_checkpoint`


On your webpage, you should include the f1-score curve at different thresholds for voxelgrid, pointcloud and the mesh network. The plot is saved as `eval_{type}.png`.
#### VOXEL:
<img width="563" height="453" alt="image" src="https://github.com/user-attachments/assets/7067d27a-a639-4343-9d39-048846963f5d" /> <br>
#### POINTCLOUD:
<img width="563" height="453" alt="image" src="https://github.com/user-attachments/assets/825b17b3-f41b-42a4-8067-c132c3627470" /> <br>



### 2.5. Analyse effects of hyperparams variations (10 points)
Analyse the results, by varying a hyperparameter of your choice.
For example `n_points` or `vox_size` or `w_chamfer` or `initial mesh (ico_sphere)` etc.
Try to be unique and conclusive in your analysis.
#### In Processing (Đang dùng wandb :<< và trước đó đã điều chỉnh các thông số trên) 
### 2.6. Interpret your model (15 points)
Simply seeing final predictions and numerical evaluations is not always insightful. Can you create some visualizations that help highlight what your learned model does? Be creative and think of what visualizations would help you gain insights. There is no `right' answer - although reading some papers to get inspiration might give you ideas.
#### 

## 3. Exploring other architectures / datasets. (Choose at least one! More than one is extra credit)

