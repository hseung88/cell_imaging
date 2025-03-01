import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from mil_model import MILModel, CellImageDataset


def visualize_attention(image_path, coords, attn_weights, patch_size):
    """
    Overlays attention weights on the original image
    """
    img = Image.open(image_path).convert('RGB')
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(img)
    width, height = img.size

    if torch.is_tensor(coords):
        coords = coords.cpu().detach().numpy()
    if torch.is_tensor(attn_weights):
        attn_weights = attn_weights.cpu().detach().numpy()
    attn_weights = attn_weights.squeeze()  # (num_patches,)
    print("Attention weights:", attn_weights)

    for (x_norm, y_norm), weight in zip(coords, attn_weights):
        # convert normalized coordinates to pixel coordinates
        x = x_norm * width
        y = y_norm * height
        # draw a rectangle at the patch location
        rect = mpatches.Rectangle(
            (x, y), patch_size, patch_size,
            linewidth=10,
            edgecolor=(0, 0, 1, weight),
            facecolor='none'
        )
        ax.add_patch(rect)

    plt.title("Attention Weights Overlay")
    plt.axis('off')
    save_path = "attention_overlay.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Plot saved to {save_path}")
    plt.show()


def main():
    image_dir = "synthetic_images"
    label_file = f"{image_dir}/labels.csv"
    patch_size = 224

    transform = transforms.Compose([
        transforms.Resize((patch_size, patch_size)),
        transforms.ToTensor()
    ])

    dataset = CellImageDataset(image_dir=image_dir, label_file=label_file,
                               patch_size=patch_size, transform=transform)
    patches, coords, label = dataset[8]  # Load an image
    print(f"Image label: {label.item()}")

    # initialize the MIL model
    model = MILModel(patch_feature_dim=128, include_coords=True, bag_batch_size=64)

    # load the saved model checkpoint
    checkpoint_path = "./checkpoints/final_model.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    # run a forward pass to obtain the attention weights
    with torch.no_grad():
        y_pred, attn_weights = model(patches, coords)
        print(f"Model prediction: {y_pred.item()}")

    # construct the full image path for visualization
    image_filename = dataset.image_labels[0][0]
    image_path = f"{image_dir}/{image_filename}"

    # visualize the attention weights
    visualize_attention(image_path, coords, attn_weights, patch_size)


if __name__ == "__main__":
    main()
