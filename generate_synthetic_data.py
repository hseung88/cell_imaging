import os
import numpy as np
from PIL import Image, ImageDraw
import csv
import random


def generate_synthetic_data_with_random_circles(num_images=10,
                                                image_size=(4096, 4096),  # Updated image size
                                                output_dir="synthetic_images_random",
                                                csv_filename="labels.csv",
                                                marker_min_radius=25,
                                                marker_max_radius=75,
                                                max_markers=5):
    """
    Generate synthetic high-resolution images (default size: 4096x4096). For images with label==1,
    randomly generate multiple bold circle markers at random positions.

    Args:
        num_images (int): Number of images to generate.
        image_size (tuple): Size of each image (width, height). Default is (4096, 4096).
        output_dir (str): Directory to store images and CSV.
        csv_filename (str): Name of the CSV file to store image filenames and labels.
        marker_min_radius (int): Minimum radius for a circle marker.
        marker_max_radius (int): Maximum radius for a circle marker.
        max_markers (int): Maximum number of markers in a positive image.
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, csv_filename)

    with open(csv_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        for i in range(num_images):
            # Create a random background image.
            img_array = np.random.randint(0, 256, (image_size[1], image_size[0], 3), dtype=np.uint8)
            img = Image.fromarray(img_array)

            # Randomly assign a binary label.
            label = np.random.randint(0, 2)

            # For positive images, add a random number of bold circle markers.
            if label == 1:
                draw = ImageDraw.Draw(img)
                num_markers = random.randint(1, max_markers)
                for _ in range(num_markers):
                    # Randomly determine marker radius.
                    radius = random.randint(marker_min_radius, marker_max_radius)
                    # Ensure the circle fits within the image boundaries:
                    center_x = random.randint(radius, image_size[0] - radius)
                    center_y = random.randint(radius, image_size[1] - radius)
                    # Define bounding box for the circle: (left, top, right, bottom)
                    bbox = [center_x - radius, center_y - radius, center_x + radius, center_y + radius]
                    # Draw a bold circle marker with a red outline.
                    draw.ellipse(bbox, outline="red", width=5)

            filename = f"synthetic_{i}.jpg"
            img.save(os.path.join(output_dir, filename))
            csvwriter.writerow([filename, label])

    print(f"Synthetic data generated in '{output_dir}' with CSV file '{csv_path}'.")


if __name__ == "__main__":
    generate_synthetic_data_with_random_circles(num_images=100, image_size=(4096, 4096))
