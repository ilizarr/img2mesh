from matplotlib import pyplot as plt

import torch
import kaolin as kal

from model import create_model
from dataset import load_kaolin_data, vgg16_preprocess

# Use GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

def recenter_vertices(vertices, mesh_shift):
    """Recenter vertices by mesh_shift"""
    vertices_min, _ = vertices.min(dim=1, keepdim=True)
    vertices_max, _ = vertices.max(dim=1, keepdim=True)
    vertices_mid = (vertices_min + vertices_max) / 2
    vertices = vertices - vertices_mid + mesh_shift
    return vertices

def render(height, width, vertices, mesh_shift, vertex_colors, faces, cam_transform, cam_proj):
    """ Differentiable render call using DIB-R
    vertices      - learned mesh vertices           (B x Nv x 3)
    mesh_shift    - learned mesh offset             (B x 1 x 3)
    vertex_colors - learned vertex colors           (B x Nv x 3)
    faces         - (static) mesh faces             (Nf x 3)
    cam_transform - (batched) camera transform      (B x 4 x 3)
    cam_proj      - (batched) camera projection     (B x 3 x 1)  opengl frustrum [n/f, n/t, -1]
    """
    batch_size = cam_transform.shape[0]
    nb_faces = faces.shape[0]
    # Recenter mesh
    centered_vertices = recenter_vertices(vertices, mesh_shift)
    
    # Project faces to camera and image space
    face_vertices_camera, face_vertices_image, face_normals = \
        kal.render.mesh.prepare_vertices(
            centered_vertices, faces, cam_proj, camera_transform=cam_transform)

    ### Perform Rasterization ###
    # Construct attributes that DIB-R rasterizer will interpolate.
    # the first will render into an image without the need for texture mapping
    # the second will make a hard segmentation mask
    #vertex_colors = torch.clamp(vertex_colors, 0., 1.)
    face_vert_colors = kal.ops.mesh.index_vertices_by_faces(vertex_colors, faces)
    face_attributes = [
        face_vert_colors,
        torch.ones((batch_size, nb_faces, 3, 1), device=device)
    ]

    image_features, soft_mask, rendered_face_per_pix = kal.render.mesh.dibr_rasterization(
        height, width, face_vertices_camera[:, :, :, -1],
        face_vertices_image, face_attributes, face_normals[:, :, -1])

    # image_features is a tuple in composed of the interpolated attributes of face_attributes
    image_color, mask = image_features
    image = torch.clamp(image_color, 0., 1.)# * mask, 0., 1.)
    # Convert (B x H x W x C) to (B x C x H x W)
    image = image.permute(0, 3, 1, 2) 

    return image, mask, soft_mask, rendered_face_per_pix, face_normals

def save_plots(dirname, epoch_num, data_set, elem_ids, model, vertices_init, faces):
    """Save epoch progress. Renders current predicted mesh versus truth images from different view points"""
    with torch.no_grad():
        batch_size = len(elem_ids)

        # Get input data in a batch. Each input has a different camera
        samples = [data_set[idx] for idx in elem_ids]
        cam_transform = torch.stack([data['cam_transform'] for data in samples], dim=0).cuda()
        cam_proj = torch.stack([data['cam_proj'] for data in samples], dim=0).cuda()
        in_images = torch.stack([data['rgb'].cpu().detach() for data in samples], dim=0).cuda()
        orig_image = torch.stack([data['rgb_orig'].cpu().detach() for data in samples], dim=0).cuda()
        height, width = orig_image.shape[-2], orig_image.shape[-1]

        # Run through model. 
        # Output shapes:
        #   y_vertices (1 x Vertices x 3)
        #   y_color    (1 x Vertices x 3)
        #   y_pos      (1 x 1 x 3) 
        x = in_images.unsqueeze(0)
        y_vertices, y_color, y_pos = model(x)

        # Defrom the base mesh
        y_vertices = vertices_init + y_vertices

        # Expand out to full batch
        vertices = y_vertices.repeat(batch_size, 1, 1)
        vertex_colors = y_color.repeat(batch_size, 1, 1)
        mesh_offset = y_pos.repeat(batch_size, 1, 1)

        # Render from all views at once
        image, mask, soft_mask, rendered_face_per_pix, face_normals = \
            render(height, width, vertices, mesh_offset, vertex_colors, faces, cam_transform, cam_proj)

        # Display the rendered images
        f, axarr = plt.subplots(2, batch_size, figsize=(10, 10))
        f.tight_layout()
        plt.subplots_adjust(top=0.88)
        f.suptitle(f'Epoch {epoch_num}: Model (top) vs Truth (bottom)', fontsize=30)
        for i in range(batch_size):
            axarr[0][i].imshow(image[i].cpu().detach().permute(1,2,0) )
            axarr[1][i].imshow(orig_image[i].cpu().detach().permute(1,2,0) )
        plt.savefig(f"{dirname}/pred-vs-truth-epoch-{epoch_num}.png")
        plt.close('all')

def save_plots_no_model(dirname, epoch_num, data_set, elem_ids, vertices_alt, vertices_color_alt, vertice_shift_alt, faces):
    """Save epoch progress. Renders current predicted mesh versus truth images from different view points"""
    with torch.no_grad():
        batch_size = len(elem_ids)

        # Get input data in a batch. Each input has a different camera
        samples = [data_set[idx] for idx in elem_ids]
        cam_transform = torch.stack([data['cam_transform'] for data in samples], dim=0).cuda()
        cam_proj = torch.stack([data['cam_proj'] for data in samples], dim=0).cuda()
        orig_image = torch.stack([data['rgb_orig'] for data in samples], dim=0).cuda()
        height, width = orig_image.shape[-2], orig_image.shape[-1]

        vertices = vertices_alt.repeat(batch_size, 1, 1)
        vertex_colors = vertices_color_alt.repeat(batch_size, 1, 1)
        mesh_offset = vertice_shift_alt.repeat(batch_size, 1, 1)

        # Render from all views at once
        image, mask, soft_mask, rendered_face_per_pix, face_normals = \
            render(height, width, vertices, mesh_offset, vertex_colors, faces, cam_transform, cam_proj)

        # Display the rendered images
        f, axarr = plt.subplots(2, batch_size, figsize=(10, 10))
        f.tight_layout()
        plt.subplots_adjust(top=0.88)
        f.suptitle(f'Epoch {epoch_num}: Model (top) vs Truth (bottom)', fontsize=30)
        for i in range(batch_size):
            axarr[0][i].imshow(image[i].cpu().detach().permute(1,2,0) )
            axarr[1][i].imshow(orig_image[i].cpu().detach().permute(1,2,0) )
        plt.savefig(f"{dirname}/pred-vs-truth-epoch-{epoch_num}.png")
        plt.close('all')

# path to the output logs
logs_path = './logs/'

# Hyperparameters
skip_model = False
num_epoch = 30
flat_weight = 1e-3 if skip_model else 1e-3
edge_weight = 1e-8 if skip_model else 1e-8
image_weight = 0.1 if skip_model else 0.1
mask_weight = 1.   if skip_model else 10.
lr = 5e-2 if skip_model else 1e-5
scheduler_step_size = 15
scheduler_gamma = 0.5

# Load data
batch_size = 1 if skip_model else 1
views_per_batch = 2
data_transform = vgg16_preprocess if not skip_model else None
full_dataset, dataloader_train, dataloader_val = load_kaolin_data("../kaolin/examples/samples/rendered_clock/", train_split=0.8, batch_size=batch_size, \
    views_per_batch = views_per_batch, transform=data_transform)

# Load base mesh that the model will deform
mesh = kal.io.obj.import_mesh('../kaolin/examples/samples/sphere.obj', with_materials=True)
# vertices_alt = mesh.vertices.cuda().unsqueeze(0) * 75
# vertices_alt.requires_grad = True
vertices_init = mesh.vertices.cuda().unsqueeze(0) * 75 # 75 - default sphere is too small...(TODO)
vertices_init.requires_grad = False
faces = mesh.faces.cuda()                            

nb_faces = faces.shape[0]
nb_vertices =  mesh.vertices.shape[0]
face_size = 3

# Create the model
if not skip_model:
    model = create_model(num_views=views_per_batch, num_vertices=nb_vertices, att_heads=2, att_layers=1)
    model.to(device)
    model.train()

## Set up auxiliary connectivity matrix of edges to faces indexes for the flat loss
edges = torch.cat([faces[:,i:i+2] for i in range(face_size - 1)] + [faces[:,[-1,0]]], dim=0)
edges = torch.sort(edges, dim=1)[0]
face_ids = torch.arange(nb_faces, device=device, dtype=torch.long).repeat(face_size)
edges, edges_ids = torch.unique(edges, sorted=True, return_inverse=True, dim=0)
nb_edges = edges.shape[0]
# edge to faces
sorted_edges_ids, order_edges_ids = torch.sort(edges_ids)
sorted_faces_ids = face_ids[order_edges_ids]
# indices of first occurences of each key
idx_first = torch.where(torch.nn.functional.pad(sorted_edges_ids[1:] != sorted_edges_ids[:-1], (1,0), value=1))[0]
nb_faces_per_edge = idx_first[1:] - idx_first[:-1]
# compute sub_idx (2nd axis indices to store the faces)
offsets = torch.zeros(sorted_edges_ids.shape[0], device=device, dtype=torch.long)
offsets[idx_first[1:]] = nb_faces_per_edge
sub_idx = (torch.arange(sorted_edges_ids.shape[0], device=device, dtype=torch.long) - torch.cumsum(offsets, dim=0))
nb_faces_per_edge = torch.cat([nb_faces_per_edge, sorted_edges_ids.shape[0] - idx_first[-1:]], dim=0)
max_sub_idx = 2
edge2faces = torch.zeros((nb_edges, max_sub_idx), device=device, dtype=torch.long)
edge2faces[sorted_edges_ids, sub_idx] = sorted_faces_ids

## TRAIN
if not skip_model:
    optim  = torch.optim.Adam(model.parameters(), lr=lr)
else:
    vertices_alt = vertices_init.detach().clone().cuda()
    vertices_alt.requires_grad = True
    vertice_shift_alt = torch.zeros((1,1,3), dtype=torch.float, device='cuda', requires_grad=True)
    vertices_color_alt = torch.ones(vertices_alt.shape, dtype=torch.float, device='cuda', requires_grad=True)
    optim  = torch.optim.Adam(params=[vertices_alt, vertices_color_alt, vertice_shift_alt], lr=lr)

scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=scheduler_step_size, gamma=scheduler_gamma)
for epoch in range(num_epoch):
    for b_idx, data in enumerate(dataloader_train):
    
        # Load data 
        gt_mask = data['semantic'].cuda() # (N*V x H x W x 1)
        orig_image = data['rgb_orig'].cuda()  # (N*V x C x H x W) 
        cam_transform = data['cam_transform'].cuda() # (N*V x 4 x 3)
        cam_proj = data['cam_proj'].cuda() # (N*V x 3 x 1)
        image_in = data['rgb'].cuda() # (N*V x C x H' x W')  Different height/width due to model preprocessing

        if not skip_model:
            # Split by the number of views processed in parallel
            x = image_in.view(batch_size, views_per_batch, image_in.shape[-3], image_in.shape[-2], image_in.shape[-1])

            # Run through model. 
            # Output shapes:
            #   y_vertices (N x Vertices x 3)
            #   y_color    (N x Vertices x 3)
            #   y_pos      (N x 1 x 3) 
            y_vertices, y_color, y_pos = model(x)

            # Defrom the base mesh
            y_vertices = vertices_init + y_vertices

            # Expand out to full batch
            vertices = torch.repeat_interleave(y_vertices, views_per_batch, dim=0) # (N*V x Vertices x 3)
            vertex_colors = torch.repeat_interleave(y_color, views_per_batch, dim=0) # (N*V x Vertices x 3)
            mesh_offset = torch.repeat_interleave(y_pos, views_per_batch, dim=0)   # (N*V x 1 x 3) 
        else:
            vertices = vertices_alt.repeat(batch_size*views_per_batch, 1, 1)
            vertex_colors = vertices_color_alt.repeat(batch_size*views_per_batch, 1, 1)
            mesh_offset = vertice_shift_alt.repeat(batch_size*views_per_batch, 1, 1)

        # Render to original height/width (not the same as model input) to match mask label
        height, width = orig_image.shape[-2], orig_image.shape[-1]

        # Render all batches and views at once
        image, mask, soft_mask, rendered_face_per_pix, face_normals = \
            render(height, width, vertices, mesh_offset, vertex_colors, faces, cam_transform, cam_proj)

        # Display the rendered images
        # f, axarr = plt.subplots(2, batch_size*views_per_batch, figsize=(10, 10))
        # f.tight_layout()
        # plt.subplots_adjust(top=0.88)
        # f.suptitle(f'{epoch}-{b_idx}: Model (top) vs Truth (bottom)', fontsize=30)
        # for i in range(batch_size*views_per_batch):
        #     axarr[0][i].imshow(image[i].detach().clone().cpu().permute(1,2,0) )
        #     axarr[1][i].imshow(orig_image[i].detach().clone().cpu().permute(1,2,0) )
        # plt.savefig(f"./logs/out-{epoch}.png")
        # plt.close('all')

        ### Compute Losses ###
        image_loss = torch.mean(torch.abs(image - orig_image))
        mask_loss = kal.metrics.render.mask_iou(soft_mask, gt_mask.squeeze(-1))
        # flat loss
        mesh_normals_e1 = face_normals[:, edge2faces[:, 0]]
        mesh_normals_e2 = face_normals[:, edge2faces[:, 1]]
        faces_cos = torch.sum(mesh_normals_e1 * mesh_normals_e2, dim=2)
        flat_loss = torch.mean((faces_cos - 1) ** 2) * edge2faces.shape[0]
        # Edge length loss
        # edges: nb_edges x 2
        edges_p1 = vertices[:, edges[:,0]] # (N*V x nb_edges x 3)
        edges_p2 = vertices[:, edges[:,1]] # (N*V x nb_edges x 3)
        edges_vec = edges_p2 - edges_p1
        edge_loss_per_batch = torch.sum(torch.sum(edges_vec * edges_vec, axis=-1), axis=-1)
        edge_loss = torch.mean(edge_loss_per_batch)

        loss = (
            image_loss * image_weight +
            mask_loss * mask_weight +
            flat_loss * flat_weight +
            edge_loss * edge_weight
        )

        ### Update the mesh ###
        optim.zero_grad()
        loss.backward()
        optim.step()

    scheduler.step()
    print(f"Epoch {epoch} - loss: {float(loss)}")
   
    # Save epoch progress using plots (5)
    if not skip_model:
        model.eval()
        save_plots(logs_path, epoch, full_dataset, [2, 10], model, vertices_init, faces)
        model.train()
    else:
        save_plots_no_model(logs_path, epoch, full_dataset, [2, 10], vertices_alt, vertices_color_alt, vertice_shift_alt, faces)
# for each epoch
