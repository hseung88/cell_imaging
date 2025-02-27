import os
import numpy as np
from PIL import Image, ImageDraw
import csv


def generate_synthetic_data_with_marker(num_images=10, image_size=(4096, 4096),
                                        output_dir="synthetic_images_marker",
                                        csv_filename="labels.csv",
                                        marker_min_size=50,
                                        marker_max_size=150,
                                        max_markers=5):
    """
    Generate synthetic high-resolution images. For images with label==1,
    randomly generate multiple markers at random positions.

    Args:
        num_images (int): Number of images to generate.
        image_size (tuple): Size of each image (width, height).
        output_dir (str): Directory to store images and CSV.
        csv_filename (str): Name of the CSV file to store image filenames and labels.
        marker_min_size (int): Minimum size for a marker.
        marker_max_size (int): Maximum size for a marker.
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

            # For positive images, add a random number of markers.
            if label == 1:
                draw = ImageDraw.Draw(img)
                num_markers = random.randint(1, max_markers)
                for _ in range(num_markers):
                    # Randomly determine marker size.
                    marker_size = random.randint(marker_min_size, marker_max_size)
                    # Ensure marker fits within the image.
                    max_x = image_size[0] - marker_size
                    max_y = image_size[1] - marker_size
                    x = random.randint(0, max_x)
                    y = random.randint(0, max_y)
                    # Draw a rectangle marker with a red outline.
                    draw.rectangle([(x, y), (x + marker_size, y + marker_size)], outline="red", width=5)

            filename = f"synthetic_{i}.jpg"
            img.save(os.path.join(output_dir, filename))
            csvwriter.writerow([filename, label])

    print(f"Synthetic data generated in '{output_dir}' with CSV file '{csv_path}'.")

if __name__ == "__main__":
    generate_synthetic_data_with_random_markers(num_images=10, image_size=(1024, 1024))