import torch
import matplotlib.pyplot as plt
import cv2
import os

def plot_images(image_tensor, phosphenes_tensor, reconstruction_tensor, figure_title, subtitles, save_name):
    # Convert tensors to numpy and handle channels
    image_np = image_tensor[5].cpu().detach().numpy().squeeze()
    phosphenes_np = phosphenes_tensor[5].cpu().detach().numpy().squeeze()
    reconstruction_np = reconstruction_tensor[5].cpu().detach().numpy().squeeze()

    # Resize phosphenes image to (128, 128)
    phosphenes_np = cv2.resize(phosphenes_np, (128, 128))

    # Create a figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Adjust layout to make space for the general title
    plt.subplots_adjust(top=0.85)

    # Plotting each image with its subtitle
    images = [image_np, phosphenes_np, reconstruction_np]
    for ax, img, subtitle in zip(axes, images, subtitles):
        ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
        ax.set_title(subtitle)
        ax.axis('off')
        ax.set_xticks([])  # Remove x-axis ticks
        ax.set_yticks([])  # Remove y-axis ticks

    # Check if the directory exists, and if not, create it
    save_dir = '/scratch/big/home/carsan/Internship/PyCharm_projects/habitat_2.3/habitat-phosphenes/reconstructionEvolutionImg'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save the figure
    save_path = os.path.join(save_dir, f'{save_name}.png')
    # Set the general title for the figure
    fig.suptitle(figure_title, fontsize=16)
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

# Example usage
# Assuming 'image', 'phosphenes', and 'reconstruction' are your tensors
# and they are already on the CPU and detached from gradients
plot_images(image, phosphenes, reconstruction["rgb"],
            'Reconstruction after 1 update',
            ['Gray Scaled Image', 'Phosphenes', 'Reconstruction'],
            'MSE_1update')
