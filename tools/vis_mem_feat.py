import matplotlib.pyplot as plt
import os
import torch

# Example input: 
# Step 1: Reshape your tensor
x = current_vision_feats[-1].cpu().to(torch.float32)  # Your input tensor: torch.randn(1024, 1, 64)
x = x.view(32, 32, 256)      # Now shape is (H, W, C) = (32, 32, 64)  (128, 128, 32)  (32, 32, 256)  (64, 64, 64)  
x = x.permute(2, 0, 1)        # Convert to (C, H, W) = (64, 32, 32)

#  Step 2: Visualize one or multiple feature maps
# Output directory
os.makedirs("feature_maps", exist_ok=True)

# Plot and save each feature map
for i in range(x.shape[0]):
    fig, ax = plt.subplots(figsize=(1, 1))  # Increase figure size for higher resolution
    ax.imshow(x[i].cpu().numpy(), cmap='viridis', interpolation='nearest')
    ax.axis('off')
    plt.tight_layout()
    # plt.savefig(f"feature_maps/feature_map_{i:02d}.png", dpi=300, bbox_inches='tight', pad_inches=0)
    # plt.close(fig)
    plt.show()  # Display the feature map
    break
print("check")



#########
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

def save_numpy_as_rgb_image(array, save_path, cmap='viridis'):
    """
    Normalize a 2D NumPy array, apply a colormap, and save as RGB PNG.

    Args:
        array (np.ndarray): 2D input array.
        save_path (str): Output image path (e.g., "feature_map_rgb.png").
        cmap (str): Matplotlib colormap name.
    """
    assert array.ndim == 2, "Only 2D arrays are supported"

    # Normalize to [0, 1]
    norm_array = (array - np.min(array)) / (np.max(array) - np.min(array) + 1e-8)

    # Apply colormap to get RGB (float range [0,1], shape HxWx4)
    colormap = cm.get_cmap(cmap)
    colored_array = colormap(norm_array)[:, :, :3]  # Drop alpha channel

    # Convert to uint8 for saving
    rgb_uint8 = (colored_array * 255).astype(np.uint8)

    # Save using PIL
    img = Image.fromarray(rgb_uint8)
    img.save(save_path)

# Example usage
# feature_map = np.random.randn(32, 32)  # Replace with your data
save_numpy_as_rgb_image(feature_map, "feature_map_rgb.png", cmap='magma')


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from scipy.ndimage import zoom

def save_numpy_as_rgb_image(array, save_path, cmap='gray'):
    """
    Normalize a 2D NumPy array, apply a colormap, and save as RGB PNG.

    Args:
        array (np.ndarray): 2D input array.
        save_path (str): Output image path (e.g., "feature_map_rgb.png").
        cmap (str): Matplotlib colormap name. viridis
    """
    assert array.ndim == 2, "Only 2D arrays are supported"


    # Interpolate (resize)
    zoom_factors = (1024.0  / array.shape[0], 1280.0 / array.shape[1])
    array = zoom(array, zoom_factors, order=3)  # bicubic interpolation (order=3)

    # Normalize to [0, 1]
    norm_array = (array - np.min(array)) / (np.max(array) - np.min(array) + 1e-8)

    # Apply colormap to get RGB (float range [0,1], shape HxWx4)
    colormap = cm.get_cmap(cmap)
    colored_array = colormap(norm_array)[:, :, :3]  # Drop alpha channel

    # Convert to uint8 for saving
    rgb_uint8 = (colored_array * 255).astype(np.uint8)

    # Save using PIL
    img = Image.fromarray(rgb_uint8)
    img.save(save_path)

# Example usage
# feature_map = np.random.randn(32, 32)  # Replace with your data
x = current_vision_feats[0].cpu().to(torch.float32)  # Your input tensor: torch.randn(1024, 1, 64)
x = x.view(128, 128, 32)     # Now shape is (H, W, C) = (32, 32, 64)  (128, 128, 32)  (32, 32, 256)  (64, 64, 64)  
x = x.permute(2, 0, 1)        # Convert to (C, H, W) = (64, 32, 32)
sel_idx = [0, 1, 2, 3, 4, 5] # [0,1,2,3] # [0,1] # [0] # [0]
for i in sel_idx:
    feature_map = x[i, :, :].cpu().numpy()  # Select the i-th feature map
    # Save the feature map as an RGB image
    save_numpy_as_rgb_image(feature_map, "seq_227_feature_map_ch"+ str(i)+".png", cmap='gray')
