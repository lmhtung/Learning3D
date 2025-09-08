import torch
from pytorch3d.ops import knn_points 
from pytorch3d.loss import mesh_laplacian_smoothing

# define losses
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

def smoothness_loss(mesh_src):
	loss_laplacian = mesh_laplacian_smoothing(mesh_src)
	# implement laplacian smoothening loss
	return loss_laplacian