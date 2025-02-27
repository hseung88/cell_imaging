import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from mil_model import MILModel, CellImageDataset


def train_model(model, dataloader, num_epochs, lr, device, checkpoint_dir, checkpoint_interval):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    criterion = torch.nn.BCELoss()
    start_epoch = 0

    # Check for an existing checkpoint to resume training.
    checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Loaded checkpoint from epoch {start_epoch}")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0

        # Process one bag (image) per iteration.
        for patches, coords, label in dataloader:
            # Each sample is a bag with variable number of patches.
            # Squeeze batch dimension since batch_size=1.
            patches = patches.squeeze(0).to(device)  # (num_patches, C, H, W)
            coords = coords.squeeze(0).to(device)  # (num_patches, 2)
            label = label.to(device).unsqueeze(0)  # (1)

            optimizer.zero_grad()
            y_pred, attn_weights = model(patches, coords)
            loss = criterion(y_pred, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

        # Save a checkpoint at specified intervals.
        if (epoch + 1) % checkpoint_interval == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss
            }
            checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth")
            torch.save(checkpoint, checkpoint_file)
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint for epoch {epoch + 1}")

    # Save the final model.
    os.makedirs(checkpoint_dir, exist_ok=True)
    final_model_path = os.path.join(checkpoint_dir, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print("Training complete. Final model saved.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True, help="Directory with high-res images")
    parser.add_argument("--label_file", type=str, required=True, help="CSV file with image labels (filename,label)")
    parser.add_argument("--patch_size", type=int, default=224, help="Patch size to extract from images")
    parser.add_argument("--bag_batch_size", type=int, default=128, help="Mini-batch size within each bag")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to store checkpoints")
    parser.add_argument("--checkpoint_interval", type=int, default=5, help="Interval (in epochs) to save checkpoints")
    parser.add_argument("--device", type=str, default="cuda", help="Device to train on (cuda or cpu)")
    args = parser.parse_args()

    # Define transforms for each patch.
    transform = transforms.Compose([
        transforms.Resize((args.patch_size, args.patch_size)),
        transforms.ToTensor()
    ])

    # Create the dataset and dataloader.
    dataset = CellImageDataset(image_dir=args.image_dir, label_file=args.label_file,
                               patch_size=args.patch_size, transform=transform)
    # Each sample is a bag (one image with many patches); use batch_size=1.
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Initialize the MIL model.
    model = MILModel(patch_feature_dim=128, include_coords=True, bag_batch_size=args.bag_batch_size)

    # Train the model.
    train_model(model, dataloader, num_epochs=args.num_epochs, lr=args.lr,
                device=args.device, checkpoint_dir=args.checkpoint_dir,
                checkpoint_interval=args.checkpoint_interval)


if __name__ == "__main__":
    main()