"""Example taken from kaolin tutorial"""

import json
import os
import glob
import time

from PIL import Image
import torch
import numpy as np
from matplotlib import pyplot as plt

import kaolin as kal

# path to the rendered image (using the data synthesizer)
rendered_path ="../kaolin/examples/samples/rendered_clock/"
# path to the output logs (readable with the training visualizer in the omniverse app)
logs_path = './logs/'

# Hyperparameters
num_epoch = 40
batch_size = 2
laplacian_weight = 0.#0.1
flat_weight = 0.001
image_weight = 0.1
mask_weight = 1.
lr = 5e-2
scheduler_step_size = 15
scheduler_gamma = 0.5

texture_res = 400

# select camera angle for best visualization
test_batch_ids = [2, 5, 10]
test_batch_size = len(test_batch_ids)

num_views = len(glob.glob(os.path.join(rendered_path,'*_rgb.png')))
train_data = []
for i in range(num_views):
    data = kal.io.render.import_synthetic_view(
        rendered_path, i, rgb=True, semantic=True)
    train_data.append(data)

dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                         shuffle=True, pin_memory=True)

mesh = kal.io.obj.import_mesh('../kaolin/examples/samples/sphere.obj', with_materials=True)
# the sphere is usually too small (this is fine-tuned for the clock)
vertices = mesh.vertices.cuda().unsqueeze(0) * 75
vertices.requires_grad = True
vertices_color = torch.ones(vertices.shape, dtype=torch.float, device='cuda', requires_grad=True)
faces = mesh.faces.cuda()
# uvs = mesh.uvs.cuda().unsqueeze(0)
# face_uvs_idx = mesh.face_uvs_idx.cuda()


# face_uvs = kal.ops.mesh.index_vertices_by_faces(uvs, face_uvs_idx).detach()
# face_uvs.requires_grad = False

# texture_map = torch.ones((1, 3, texture_res, texture_res), dtype=torch.float, device='cuda',
#                          requires_grad=True)

## Separate vertices center as a learnable parameter
vertices_init = vertices.detach()
vertices_init.requires_grad = False

# This is the center of the optimized mesh, separating it as a learnable parameter helps the optimization. 
vertice_shift = torch.zeros((1,1,3), dtype=torch.float, device='cuda',
                            requires_grad=True)

def recenter_vertices(vertices, vertice_shift):
    """Recenter vertices on vertice_shift for better optimization"""
    vertices_min = vertices.min(dim=1, keepdim=True)[0]
    vertices_max = vertices.max(dim=1, keepdim=True)[0]
    vertices_mid = (vertices_min + vertices_max) / 2
    vertices = vertices - vertices_mid + vertice_shift
    return vertices


nb_faces = faces.shape[0]
nb_vertices = vertices_init.shape[1]
face_size = 3

## Set up auxiliary connectivity matrix of edges to faces indexes for the flat loss
edges = torch.cat([faces[:,i:i+2] for i in range(face_size - 1)] +
                  [faces[:,[-1,0]]], dim=0)

edges = torch.sort(edges, dim=1)[0]
face_ids = torch.arange(nb_faces, device='cuda', dtype=torch.long).repeat(face_size)
edges, edges_ids = torch.unique(edges, sorted=True, return_inverse=True, dim=0)
nb_edges = edges.shape[0]
# edge to faces
sorted_edges_ids, order_edges_ids = torch.sort(edges_ids)
sorted_faces_ids = face_ids[order_edges_ids]
# indices of first occurences of each key
idx_first = torch.where(
    torch.nn.functional.pad(sorted_edges_ids[1:] != sorted_edges_ids[:-1],
                            (1,0), value=1))[0]
nb_faces_per_edge = idx_first[1:] - idx_first[:-1]
# compute sub_idx (2nd axis indices to store the faces)
offsets = torch.zeros(sorted_edges_ids.shape[0], device='cuda', dtype=torch.long)
offsets[idx_first[1:]] = nb_faces_per_edge
sub_idx = (torch.arange(sorted_edges_ids.shape[0], device='cuda', dtype=torch.long) -
           torch.cumsum(offsets, dim=0))
nb_faces_per_edge = torch.cat([nb_faces_per_edge,
                               sorted_edges_ids.shape[0] - idx_first[-1:]],
                              dim=0)
max_sub_idx = 2
edge2faces = torch.zeros((nb_edges, max_sub_idx), device='cuda', dtype=torch.long)
edge2faces[sorted_edges_ids, sub_idx] = sorted_faces_ids

# ## Set up auxiliary laplacian matrix for the laplacian loss
# vertices_laplacian_matrix = kal.ops.mesh.uniform_laplacian(
#     nb_vertices, faces)

#optim  = torch.optim.Adam(params=[vertices, texture_map, vertice_shift, vertices_color], lr=lr)
optim  = torch.optim.Adam(params=[vertices, vertice_shift, vertices_color], lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=scheduler_step_size,
                                            gamma=scheduler_gamma)

for epoch in range(num_epoch):
    for idx, data in enumerate(dataloader):
        gt_image = data['rgb'].cuda()
        gt_mask = data['semantic'].cuda()
        cam_transform = data['metadata']['cam_transform'].cuda()
        cam_proj = data['metadata']['cam_proj'].cuda()
        
        ### Prepare mesh data with projection regarding to camera ###
        vertices_batch = recenter_vertices(vertices, vertice_shift)

        face_vertices_camera, face_vertices_image, face_normals = \
            kal.render.mesh.prepare_vertices(
                vertices_batch.repeat(batch_size, 1, 1),
                faces, cam_proj, camera_transform=cam_transform
            )

        ### Perform Rasterization ###
        # Construct attributes that DIB-R rasterizer will interpolate.
        # the first is the UVS associated to each face
        # the second will make a hard segmentation mask
        face_vert_colors = kal.ops.mesh.index_vertices_by_faces(vertices_color, faces)
        face_attributes = [
            #face_uvs.repeat(batch_size, 1, 1, 1),
            torch.ones((batch_size, nb_faces, 3, 1), device='cuda'),
            face_vert_colors.repeat(batch_size, 1, 1, 1)
        ]

        image_features, soft_mask, face_idx = kal.render.mesh.dibr_rasterization(
            gt_image.shape[1], gt_image.shape[2], face_vertices_camera[:, :, :, -1],
            face_vertices_image, face_attributes, face_normals[:, :, -1])

        # image_features is a tuple in composed of the interpolated attributes of face_attributes
        #texture_coords, mask, img_color = image_features
        # image = kal.render.mesh.texture_mapping(texture_coords,
        #                                         texture_map.repeat(batch_size, 1, 1, 1), 
        #                                         mode='bilinear')
        mask, img_color = image_features
        image = img_color
        image = torch.clamp(image * mask, 0., 1.)
        
        ### Compute Losses ###
        image_loss = torch.mean(torch.abs(image - gt_image))
        mask_loss = kal.metrics.render.mask_iou(soft_mask, gt_mask.squeeze(-1))

        # laplacian loss
        # vertices_mov = vertices - vertices_init
        # vertices_mov_laplacian = torch.matmul(vertices_laplacian_matrix, vertices_mov)
        # laplacian_loss = torch.mean(vertices_mov_laplacian ** 2) * nb_vertices * 3
        # if laplacian_loss.item() != 0.0:
        #     print("WTF")
        # flat loss
        mesh_normals_e1 = face_normals[:, edge2faces[:, 0]]
        mesh_normals_e2 = face_normals[:, edge2faces[:, 1]]
        faces_cos = torch.sum(mesh_normals_e1 * mesh_normals_e2, dim=2)
        flat_loss = torch.mean((faces_cos - 1) ** 2) * edge2faces.shape[0]

        loss = (
            image_loss * image_weight +
            mask_loss * mask_weight +
            # laplacian_loss * laplacian_weight +
            flat_loss * flat_weight
        )
        ### Update the mesh ###
        optim.zero_grad()
        loss.backward()
        optim.step()

    scheduler.step()
    print(f"Epoch {epoch} - loss: {float(loss)}")

    with torch.no_grad():
        # This is similar to a training iteration (without the loss part)
        data_batch = [train_data[idx] for idx in test_batch_ids]
        cam_transform = torch.stack([data['metadata']['cam_transform'] for data in data_batch], dim=0).cuda()
        cam_proj = torch.stack([data['metadata']['cam_proj'] for data in data_batch], dim=0).cuda()
        orig_image = torch.stack([data['rgb'] for data in data_batch], dim=0).cuda()

        vertices_batch = recenter_vertices(vertices, vertice_shift)

        face_vertices_camera, face_vertices_image, face_normals = \
            kal.render.mesh.prepare_vertices(
                vertices_batch.repeat(test_batch_size, 1, 1),
                faces, cam_proj, camera_transform=cam_transform
            )
        face_vert_colors = kal.ops.mesh.index_vertices_by_faces(vertices_color, faces)
        face_attributes = [
            #face_uvs.repeat(test_batch_size, 1, 1, 1),
            torch.ones((test_batch_size, nb_faces, 3, 1), device='cuda'),
            face_vert_colors.repeat(test_batch_size, 1, 1, 1)
        ]

        image_features, soft_mask, face_idx = kal.render.mesh.dibr_rasterization(
            256, 256, face_vertices_camera[:, :, :, -1],
            face_vertices_image, face_attributes, face_normals[:, :, -1])

        #texture_coords, mask, img_color = image_features
        # image = kal.render.mesh.texture_mapping(texture_coords,
        #                                         texture_map.repeat(test_batch_size, 1, 1, 1), 
        #                                         mode='bilinear')
        mask, img_color = image_features
        image = img_color
        image = torch.clamp(image * mask, 0., 1.)
        
        ## Display the rendered images
        f, axarr = plt.subplots(2, test_batch_size, figsize=(10, 10))
        f.tight_layout()
        plt.subplots_adjust(top=0.88)
        f.suptitle('DIB-R rendering', fontsize=30)
        for i in range(test_batch_size):
            axarr[0][i].imshow(image[i].cpu().detach())
            axarr[1][i].imshow(orig_image[i].cpu().detach() )
        plt.savefig(f"{logs_path}/dibr-{epoch}.png")
        plt.close('all')