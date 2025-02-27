import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from mil_model import MILModel, CellImageDataset


def visualize_attention(image_path, coords, attn_weights, patch_size):
    """
    Overlays attention weights on the original image.

    Args:
        image_path (str): Path to the original high-resolution image.
        coords (Tensor or ndarray): Normalized (x, y) coordinates for each patch (shape: [num_patches, 2]).
        attn_weights (Tensor or ndarray): Attention weights for each patch (shape: [num_patches, 1]).
        patch_size (int): The size of each patch in pixels.
    """
    # Load the original image.
    img = Image.open(image_path).convert('RGB')
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(img)
    width, height = img.size

    # Convert tensors to numpy arrays if needed.
    if torch.is_tensor(coords):
        coords = coords.cpu().detach().numpy()
    if torch.is_tensor(attn_weights):
        attn_weights = attn_weights.cpu().detach().numpy()
    attn_weights = attn_weights.squeeze()  # Now shape: (num_patches,)

    # Overlay each patch with a rectangle whose edge opacity reflects the attention weight.
    for (x_norm, y_norm), weight in zip(coords, attn_weights):
        # Convert normalized coordinates to pixel coordinates.
        x = x_norm * width
        y = y_norm * height
        # Draw a rectangle at the patch location.
        rect = patches.Rectangle(
            (x, y), patch_size, patch_size,
            linewidth=2,
            edgecolor=(1, 0, 0, weight),  # Red color with transparency proportional to weight.
            facecolor='none'
        )
        ax.add_patch(rect)

    plt.title("Attention Weights Overlay")
    plt.axis('off')
    plt.show()


# ---- Main Visualization Script ----

# Parameters
image_dir = "synthetic_images_marker"  # Your synthetic data directory with marker images.
label_file = f"{image_dir}/labels.csv"
patch_size = 16

# Define transforms for each patch.
transform = transforms.Compose([
    transforms.Resize((patch_size, patch_size)),
    transforms.ToTensor()
])

# Load dataset and pick one sample.
dataset = CellImageDataset(image_dir=image_dir, label_file=label_file, patch_size=patch_size, transform=transform)
patches, coords, label = dataset[0]  # For example, choose the first image.
print(f"Image label: {label.item()}")

# Initialize the model.
model = MILModel(patch_feature_dim=128, include_coords=True, bag_batch_size=64)
# Load the saved model checkpoint.
checkpoint_path = "./checkpoints/final_model.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
model.eval()

# Run a forward pass to get attention weights.
with torch.no_grad():
    y_pred, attn_weights = model(patches, coords)
    print(f"Model prediction: {y_pred.item()}")

# Visualize the attention.
# Construct the path to the image file from the dataset.
image_filename = dataset.image_labels[0][0]
image_path = f"{image_dir}/{image_filename}"
visualize_attention(image_path, coords, attn_weights, patch_size)
