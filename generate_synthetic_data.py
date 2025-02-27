import os
import numpy as np
from PIL import Image, ImageDraw
import csv


def generate_synthetic_data_with_marker(num_images=20, image_size=(4096, 4096),
                                        output_dir="synthetic_images_marker",
                                        csv_filename="labels.csv", marker_size=100):
    """
    Generate synthetic high-resolution images. If label==1, add a red square marker in the top-left corner.

    Args:
        num_images (int): Number of images to generate.
        image_size (tuple): (width, height) of each image.
        output_dir (str): Directory to store images and CSV.
        csv_filename (str): CSV filename to store image names and labels.
        marker_size (int): Size of the marker square.
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

            # If label is 1, add a red square marker at the top-left.
            if label == 1:
                draw = ImageDraw.Draw(img)
                draw.rectangle([(0, 0), (marker_size, marker_size)], outline="red", width=5)

            filename = f"synthetic_{i}.jpg"
            img.save(os.path.join(output_dir, filename))
            csvwriter.writerow([filename, label])

    print(f"Synthetic images generated in '{output_dir}' with CSV file '{csv_path}'.")


if __name__ == "__main__":
    generate_synthetic_data_with_marker(num_images=10, image_size=(1024, 1024))
