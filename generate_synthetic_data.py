import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import csv
import random


def generate_synthetic_data(num_images=10,
                                        image_size=(4096, 4096),
                                        output_dir="synthetic_images",
                                        csv_filename="labels.csv",
                                        marker_min_size=30,
                                        marker_max_size=50,
                                        max_markers=10,
                                        blur_radius=15):
    """
    Generate synthetic high-resolution images with diverse markers.
    For images with label==1, randomly generate between 1 and max_markers markers.
    Markers can be circles, ellipses, or rectangles with random sizes, positions, orientations, and colors.
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, csv_filename)

    with open(csv_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        for i in range(num_images):
            # generate a random noise background
            noise = np.random.rand(image_size[1], image_size[0], 3) * 255
            noise = noise.astype(np.uint8)
            img = Image.fromarray(noise)
            # apply gaussian blur to create smoother, tissue-like texture
            img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

            # vary brightness across the image
            brightness_factor = random.uniform(0.8, 1.2)
            img = Image.fromarray((np.array(img) * brightness_factor).clip(0, 255).astype(np.uint8))

            # randomly assign a binary label
            label = np.random.randint(0, 2)

            # add random markers
            if label == 1:
                draw = ImageDraw.Draw(img)
                num_markers = random.randint(1, max_markers)
                for _ in range(num_markers):
                    shape = random.choice(["circle", "ellipse", "rectangle"])
                    size = random.randint(marker_min_size, marker_max_size)

                    max_x = image_size[0] - size
                    max_y = image_size[1] - size
                    x = random.randint(0, max_x)
                    y = random.randint(0, max_y)

                    color = random.choice(["red", "blue", "green", "yellow"])

                    width_outline = random.randint(50, 100)

                    if shape == "circle":
                        bbox = [x, y, x + size, y + size]
                        draw.ellipse(bbox, outline=color, width=width_outline)
                    elif shape == "ellipse":
                        w = random.randint(marker_min_size, marker_max_size)
                        h = random.randint(marker_min_size, marker_max_size)
                        max_x_ellipse = image_size[0] - w
                        max_y_ellipse = image_size[1] - h
                        x_ellipse = random.randint(0, max_x_ellipse)
                        y_ellipse = random.randint(0, max_y_ellipse)
                        bbox = [x_ellipse, y_ellipse, x_ellipse + w, y_ellipse + h]
                        draw.ellipse(bbox, outline=color, width=width_outline)
                    elif shape == "rectangle":
                        bbox = [x, y, x + size, y + size]
                        draw.rectangle(bbox, outline=color, width=width_outline)

            # save images
            filename = f"synthetic_{i}.jpg"
            img.save(os.path.join(output_dir, filename))
            csvwriter.writerow([filename, label])

    print(f"Synthetic data generated in '{output_dir}' with CSV file '{csv_path}'.")


if __name__ == "__main__":
    generate_synthetic_data(num_images=100, image_size=(4096, 4096))
