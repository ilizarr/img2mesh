import os, json, math
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import kaolin as kal

def import_synthetic_view(root_dir, idx, rgb=True, semantic=False ):
    output = {}
    aspect_ratio = None

    def _import_npy(cat):
        path = os.path.join(root_dir, f'{idx}_{cat}.npy')
        if os.path.exists(path):
            output[cat] = torch.from_numpy(np.load(path))
        else:
            output[cat] = None

    def _import_png(cat):
        path = os.path.join(root_dir, f'{idx}_{cat}.png')
        if os.path.exists(path):
            output[cat] = Image.open(path).convert('RGB')
        else:
            output[cat] = None

    if rgb:
        _import_png('rgb')

    if semantic:
        _import_npy('semantic')

    with open(os.path.join(root_dir, f'{idx}_metadata.json'), 'r') as f:
        fmetadata = json.load(f)
        asset_transforms = torch.FloatTensor(fmetadata['asset_transforms'][0][1])
        cam_transform = torch.FloatTensor(fmetadata['camera_properties']['tf_mat'])
        aspect_ratio = (fmetadata['camera_properties']['resolution']['width'] /
                        fmetadata['camera_properties']['resolution']['height'])
        focal_length = fmetadata['camera_properties']['focal_length']
        horizontal_aperture = fmetadata['camera_properties']['horizontal_aperture']
        fov = 2 * math.atan(horizontal_aperture / (2 * focal_length))
        output['metadata'] = {
            'cam_transform': cam_transform[:, :3],
            'asset_transforms': asset_transforms,
            'cam_proj': kal.render.camera.generate_perspective_projection(fov, aspect_ratio),
            'clipping_range': fmetadata['camera_properties']['clipping_range']
        }
    return output

class KaolinData(Dataset):
    """Reads data generated by Nvidia's Kaolin framework.
        Uses function kal.io.render.import_synthetic_view() to load data from the file system:
            {index}_metadata.json
            {index}_rgb.png
            {index}_semantic.npy
    """

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on the RGB images.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.num_samples = len(glob.glob(os.path.join(self.root_dir,'*_rgb.png')))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):

        data = import_synthetic_view(self.root_dir, idx, rgb=True, semantic=True)
        sample = {
            'semantic': data['semantic'],
            'cam_transform': data['metadata']['cam_transform'],
            'cam_proj': data['metadata']['cam_proj'],
            'rgb_orig': torch.from_numpy(
                np.array(data['rgb'])
            )[:, :, :3].float().permute(2,0,1)  / 255.
        }
        # Convert (H x W x C) to (C x H x W)
        if self.transform is not None:
            sample['rgb'] = self.transform(data['rgb'])
        else:
            sample['rgb'] = torch.from_numpy(
                np.array(data['rgb'])
            )[:, :, :3].float().permute(2,0,1)  / 255.
        return sample

# Preprocess input image for VGG16 encoder
vgg16_preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_kaolin_data(data_dir, train_split, batch_size, views_per_batch, transform=None, seed=42):
    full_dataset = KaolinData(data_dir, transform=transform)

    num_samples = len(full_dataset)
    train_size = int(train_split * num_samples)
    val_size = num_samples - train_size
    effective_batch_size = batch_size * views_per_batch

    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))

    dataloader_train = DataLoader(train_dataset, batch_size=effective_batch_size, shuffle=True, pin_memory=True, drop_last=True)
    dataloader_val = DataLoader(val_dataset, batch_size=views_per_batch, shuffle=False, pin_memory=True, drop_last=True)
    return full_dataset, dataloader_train, dataloader_val