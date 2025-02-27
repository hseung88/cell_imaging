import os
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset
from custom_resnet import ResNetPatchCNN
from PIL import Image


class CellImageDataset(Dataset):
    """
    Dataset for high-resolution cell images.
    Each image is split into fixed-size patches with recorded spatial coordinates.
    """

    def __init__(self, image_dir, label_file, patch_size=224, transform=None):
        self.image_dir = image_dir
        self.patch_size = patch_size
        self.transform = transform
        self.image_labels = []  # List of tuples: (filename, label)
        with open(label_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    filename, label = line.split(',')
                    self.image_labels.append((filename, int(label)))

    def __len__(self):
        return len(self.image_labels)

    def extract_patches(self, image):
        """
        Divide the image into fixed-size patches and record normalized (x, y) coordinates.
        """
        width, height = image.size
        patch_size = self.patch_size
        patches = []
        coords = []
        for y in range(0, height, patch_size):
            for x in range(0, width, patch_size):
                if x + patch_size > width or y + patch_size > height:
                    continue
                patch = image.crop((x, y, x + patch_size, y + patch_size))
                patches.append(patch)
                # Normalize coordinates to [0, 1]
                coords.append([x / width, y / height])
        return patches, coords

    def __getitem__(self, idx):
        filename, label = self.image_labels[idx]
        img_path = os.path.join(self.image_dir, filename)
        image = Image.open(img_path).convert('RGB')
        patches, coords = self.extract_patches(image)

        # Apply transforms to each patch; if not provided, convert to tensor.
        if self.transform is not None:
            patches = [self.transform(p) for p in patches]
        else:
            patches = [transforms.ToTensor()(p) for p in patches]

        # Stack patches to create a tensor of shape (num_patches, C, H, W)
        patches = torch.stack(patches)
        # Convert coordinates to tensor of shape (num_patches, 2)
        coords = torch.tensor(coords, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.float)
        return patches, coords, label


class AttentionMIL(nn.Module):
    """
    Attention module for MIL aggregation.
    Computes an attention weight for each patch feature.
    """

    def __init__(self, feature_dim, hidden_dim=64):
        super(AttentionMIL, self).__init__()
        self.attention_fc = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, h):
        # h: (num_instances, feature_dim)
        a = self.attention_fc(h)  # (num_instances, 1)
        a = torch.softmax(a, dim=0)  # Softmax over all instances in the bag
        M = torch.sum(a * h, dim=0)  # Aggregated feature: (feature_dim,)
        return M, a


class MILModel(nn.Module):
    """
    Overall MIL model.
    Processes patches through ResNetPatchCNN in mini-batches, concatenates spatial coordinates,
    aggregates features using an attention module, and classifies the image.

    The bag_batch_size parameter controls mini-batching within a bag (i.e., within a single image).
    """
    def __init__(self, patch_feature_dim=128, include_coords=True, coord_dim=2, bag_batch_size=128):
        super(MILModel, self).__init__()
        self.include_coords = include_coords
        self.bag_batch_size = bag_batch_size
        self.patch_cnn = ResNetPatchCNN(num_classes=patch_feature_dim)
        agg_input_dim = patch_feature_dim + (coord_dim if include_coords else 0)
        self.attention = AttentionMIL(feature_dim=agg_input_dim, hidden_dim=64)
        self.classifier = nn.Sequential(
            nn.Linear(agg_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, patches, coords):
        num_patches = patches.size(0)
        patch_features = []
        for i in range(0, num_patches, self.bag_batch_size):
            batch_patches = patches[i:i + self.bag_batch_size]
            features_batch = self.patch_cnn(batch_patches)
            patch_features.append(features_batch)
        patch_features = torch.cat(patch_features, dim=0)

        if self.include_coords:
            x = torch.cat([patch_features, coords], dim=1)
        else:
            x = patch_features

        # MIL aggregation via attention.
        M, attn_weights = self.attention(x)
        # Add a batch dimension so that M has shape (1, agg_input_dim)
        M = M.unsqueeze(0)
        y_pred = self.classifier(M)
        return y_pred, attn_weights
