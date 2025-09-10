from torchvision import models as torchvision_models
from torchvision import transforms
import time
import torch.nn as nn
import torch.nn.functional as F
import torch
from pytorch3d.utils import ico_sphere
import pytorch3d

#Viết thêm decoder
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
        
# class MeshDecoder(nn.Module):
#     def __init__(self, n_verts ,latent_dim = 512, hidden_dim= 256):
#         super(MeshDecoder, self).__init__()
#         self.n_verts = n_verts
#         self.fc1 = nn.Linear(latent_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim*2)
#         self.fc3 = nn.Linear(hidden_dim*2, self.n_verts *3)
#     def forward(self, z):

#         layer1 = F.relu(self.fc1(z))
#         layer2 = F.relu(self.fc2(layer1))
#         layer3 = torch.tanh(self.fc3(layer2))
#         return layer3.view(-1, self.n_verts, 3)
# class GraphConv(nn.Module):
#     def __init__(self, in_dim, out_dim):
#         super().__init__()
#         self.fc_self = nn.Linear(in_dim, out_dim)
#         self.fc_nb = nn.Linear(in_dim, out_dim)

#     def forward(self, h, edge_index):
#         """
#         h: (N,C)
#         edge_index: (2,E) long tensor
#         """
#         N = h.shape[0]
#         src, dst = edge_index
#         # sum neighbor features
#         nb_sum = torch.zeros_like(h)
#         nb_sum.index_add_(0, dst, h[src])
#         # compute degree
#         deg = torch.zeros(N, device=h.device).index_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
#         deg = deg.clamp(min=1).unsqueeze(-1)
#         nb_mean = nb_sum / deg
#         return F.relu(self.fc_self(h) + self.fc_nb(nb_mean))
    
# class GCNDecoder(nn.Module):
#     def __init__(self, n_verts, latent_dim = 512,hidden_dim = 256, n_layers = 3):
#         super().__init__()
#         self.n_verts = n_verts
#         self.n_layers = n_layers

#         self.fc = nn.Sequential(
#             nn.Linear(latent_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, n_verts)
#         )

#         self.GCN_layers = nn.ModuleList([
#             GraphConv(hidden_dim, hidden_dim) for i in range(n_layers)
#         ])

#         self.pred = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim/2),
#             nn.ReLU(),
#             nn.Linear(hidden_dim/2, 3)
#         )




class MeshModule(nn.Module):
    def __init__(self, shape):
        super(MeshModule, self,).__init__()
        self.shape = shape
        self.layer0 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, self.shape * 3)
        ) 

    def forward(self, feats):
        res = self.layer0(feats)
        res = res.view(-1, self.shape, 3)
        return res
    
class SingleViewto3D(nn.Module):
    def __init__(self, args):
        super(SingleViewto3D, self).__init__()
        self.device = args.device
        if not args.load_feat:
            vision_model = torchvision_models.__dict__[args.arch](pretrained=True)
            self.encoder = torch.nn.Sequential(*(list(vision_model.children())[:-1]))
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])


        # define decoder
        if args.type == "vox":
            # Input: b x 512
            # Output: b x 32 x 32 x 32
            pass
            # TODO:
            self.decoder = MLPDecoder()            
        elif args.type == "point":
            # Input: b x 512
            # Output: b x args.n_points x 3  
            self.n_point = args.n_points
            # TODO:
            self.decoder = PointDecoder(self.n_point)         
        elif args.type == "mesh":
            # Input: b x 512
            # Output: b x mesh_pred.verts_packed().shape[0] x 3  
            # try different mesh initializations
            mesh_pred = ico_sphere(4, self.device)
            print("===================================================")
            self.mesh_pred = pytorch3d.structures.Meshes(mesh_pred.verts_list()*args.batch_size, mesh_pred.faces_list()*args.batch_size)
            print(len(mesh_pred.verts_list()*args.batch_size), len(mesh_pred.faces_list()*args.batch_size))
            # TODO:
            n_verts = mesh_pred.verts_packed().shape[0] 
            self.decoder = MeshModule(shape=n_verts)


    def forward(self, images, args):
        results = dict()

        total_loss = 0.0
        start_time = time.time()

        B = images.shape[0]

        if not args.load_feat:
            images_normalize = self.normalize(images.permute(0,3,1,2))
            encoded_feat = self.encoder(images_normalize).squeeze(-1).squeeze(-1) # b x 512
        else:
            encoded_feat = images # in case of args.load_feat input images are pretrained resnet18 features of b x 512 size

        # call decoder
        if args.type == "vox":
            # TODO:
            voxels_pred = self.decoder(encoded_feat)
            print(voxels_pred.shape)         
            return voxels_pred

        elif args.type == "point":
            # TODO:
            # pointclouds_pred =   
            pointclouds_pred = self.decoder(encoded_feat)        
            return pointclouds_pred

        elif args.type == "mesh":
            # TODO:
            # deform_vertices_pred = 

            


            deform_vertices_pred = self.decoder(encoded_feat)   
            deform_vertices_pred = deform_vertices_pred.view(-1, 3)
            print("mesh verts:", self.mesh_pred.verts_packed().shape)
            mesh_pred = self.mesh_pred.offset_verts(deform_vertices_pred)
            return  mesh_pred   