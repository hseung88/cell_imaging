import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset
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


# --- Positional Encoding Module ---
class PositionalEncoding(nn.Module):
    """
    A simple MLP-based positional encoding module.
    Maps 2D normalized coordinates to a higher-dimensional embedding.
    """

    def __init__(self, pos_dim):
        super(PositionalEncoding, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, pos_dim),
            nn.ReLU(),
            nn.Linear(pos_dim, pos_dim)
        )

    def forward(self, coords):
        # coords: (num_patches, 2)
        # Returns: (num_patches, pos_dim)
        return self.mlp(coords)


# --- BasicBlock and ResNetPatchCNN remain unchanged ---
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        shortcut = self.shortcut(x)
        out += shortcut
        return F.relu(out)


class ResNetPatchCNN(nn.Module):
    """
    A ResNet-style CNN for patch-level feature extraction.
    """

    def __init__(self, block=BasicBlock, num_blocks=[2, 2, 2], num_classes=128):
        super(ResNetPatchCNN, self).__init__()
        self.in_channels = 64

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# --- AttentionMIL Module ---
class AttentionMIL(nn.Module):
    """
    Attention module for MIL aggregation.
    Computes an attention score for each patch feature.
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
        a = torch.softmax(a, dim=0)  # Softmax over instances
        M = torch.sum(a * h, dim=0)  # Aggregated feature: (feature_dim,)
        return M, a


# --- MIL Model with Advanced Positional Encoding ---
class MILModel(nn.Module):
    """
    MIL model that extracts patch-level features using a ResNet-based CNN,
    integrates spatial information via a learned positional encoding,
    aggregates patch features with an attention module, and performs image-level classification.

    Parameters:
      - patch_feature_dim: Dimension of patch features from ResNetPatchCNN.
      - include_coords: Boolean flag to include spatial information.
      - pos_dim: Dimension of the positional encoding.
      - bag_batch_size: Mini-batch size for processing patches within a bag.
    """

    def __init__(self, patch_feature_dim=128, include_coords=True, pos_dim=32, bag_batch_size=128):
        super(MILModel, self).__init__()
        self.include_coords = include_coords
        self.bag_batch_size = bag_batch_size
        self.patch_cnn = ResNetPatchCNN(num_classes=patch_feature_dim)

        if include_coords:
            self.positional_encoder = PositionalEncoding(pos_dim)
            agg_input_dim = patch_feature_dim + pos_dim
        else:
            agg_input_dim = patch_feature_dim

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
        # Process patches in mini-batches to manage memory.
        for i in range(0, num_patches, self.bag_batch_size):
            batch_patches = patches[i:i + self.bag_batch_size]
            features_batch = self.patch_cnn(batch_patches)
            patch_features.append(features_batch)
        patch_features = torch.cat(patch_features, dim=0)  # (num_patches, patch_feature_dim)

        if self.include_coords:
            # Use the advanced positional encoder instead of raw coordinates.
            pos_encoding = self.positional_encoder(coords)  # (num_patches, pos_dim)
            x = torch.cat([patch_features, pos_encoding], dim=1)
        else:
            x = patch_features

        # Aggregate patch features using the attention module.
        M, attn_weights = self.attention(x)
        M = M.unsqueeze(0)  # Add batch dimension: (1, agg_input_dim)
        y_pred = self.classifier(M)
        return y_pred, attn_weights

