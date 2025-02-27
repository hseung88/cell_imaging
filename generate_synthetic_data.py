import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import csv
import random


def generate_synthetic_data(num_images=10,
                                        image_size=(4096, 4096),
                                        output_dir="synthetic_images",
                                        csv_filename="labels.csv",
                                        marker_min_size=50,
                                        marker_max_size=150,
                                        max_markers=5,
                                        blur_radius=15):
    """
    Generate synthetic high-resolution images with a more realistic tissue-like background and
    diverse markers. For images with label==1, randomly generate between 1 and max_markers markers.
    Markers can be circles, ellipses, or rectangles with random sizes, positions, orientations,
    and colors.

    Args:
        num_images (int): Number of images to generate.
        image_size (tuple): Size of each image (width, height). Default is (4096, 4096).
        output_dir (str): Directory to store images and CSV.
        csv_filename (str): CSV file name to store image filenames and labels.
        marker_min_size (int): Minimum size (width/height or diameter) for a marker.
        marker_max_size (int): Maximum size for a marker.
        max_markers (int): Maximum number of markers in a positive image.
        blur_radius (int): Radius for Gaussian blur on the background.
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, csv_filename)

    with open(csv_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        for i in range(num_images):
            # Step 1: Generate a random noise background
            noise = np.random.rand(image_size[1], image_size[0], 3) * 255  # float values 0-255
            noise = noise.astype(np.uint8)
            img = Image.fromarray(noise)
            # Apply Gaussian blur to create smoother, tissue-like texture.
            img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

            # Optional: Vary brightness across the image.
            brightness_factor = random.uniform(0.8, 1.2)
            img = Image.fromarray((np.array(img) * brightness_factor).clip(0, 255).astype(np.uint8))

            # Step 2: Randomly assign a binary label.
            label = np.random.randint(0, 2)

            # Step 3: For positive images, add random markers.
            if label == 1:
                draw = ImageDraw.Draw(img)
                num_markers = random.randint(1, max_markers)
                for _ in range(num_markers):
                    # Randomly choose a marker shape.
                    shape = random.choice(["circle", "ellipse", "rectangle"])

                    # Random size: use marker_min_size and marker_max_size
                    size = random.randint(marker_min_size, marker_max_size)

                    # Ensure marker fits within the image.
                    max_x = image_size[0] - size
                    max_y = image_size[1] - size
                    x = random.randint(0, max_x)
                    y = random.randint(0, max_y)

                    # Random color for marker outline (e.g., red, blue, green, etc.)
                    color = random.choice(["red", "blue", "green", "yellow"])

                    # Randomly set a bold outline width.
                    width_outline = random.randint(3, 8)

                    if shape == "circle":
                        # For circle, define bounding box with equal width and height.
                        bbox = [x, y, x + size, y + size]
                        draw.ellipse(bbox, outline=color, width=width_outline)
                    elif shape == "ellipse":
                        # For ellipse, allow different width and height.
                        w = random.randint(marker_min_size, marker_max_size)
                        h = random.randint(marker_min_size, marker_max_size)
                        # Ensure they fit within the image.
                        max_x_ellipse = image_size[0] - w
                        max_y_ellipse = image_size[1] - h
                        x_ellipse = random.randint(0, max_x_ellipse)
                        y_ellipse = random.randint(0, max_y_ellipse)
                        bbox = [x_ellipse, y_ellipse, x_ellipse + w, y_ellipse + h]
                        draw.ellipse(bbox, outline=color, width=width_outline)
                    elif shape == "rectangle":
                        bbox = [x, y, x + size, y + size]
                        draw.rectangle(bbox, outline=color, width=width_outline)

            # Save image and write to CSV.
            filename = f"synthetic_{i}.jpg"
            img.save(os.path.join(output_dir, filename))
            csvwriter.writerow([filename, label])

    print(f"Synthetic data generated in '{output_dir}' with CSV file '{csv_path}'.")


if __name__ == "__main__":
    generate_synthetic_data(num_images=100, image_size=(4096, 4096))
