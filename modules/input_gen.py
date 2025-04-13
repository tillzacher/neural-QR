import qrcode
import numpy as np
from PIL import Image
import os

# def generate_qr_code(
#     data,
#     filename=None,
#     version=None,
#     error_correction=qrcode.constants.ERROR_CORRECT_H,
#     box_size=20,
#     border=4,
#     inverted=False,
#     add_logo=False,
#     logo_path='logo.png',
#     logo_scale=1.5,
#     central_mask_shape='circle',
#     central_mask_size=70,
#     column_width=None,
#     border_softness=0
# ):
#     """
#     Generates a QR code with optional masking and logo overlay.

#     Parameters:
#     - data (str): Data to encode in the QR code.
#     - filename (str): Optional filename to save the QR code image.
#     - version (int): Version of the QR code (controls size).
#     - error_correction: Error correction level from qrcode.constants.
#     - box_size (int): Size of each QR code box.
#     - border (int): Border size around the QR code.
#     - inverted (bool): If True, invert QR colors.
#     - add_logo (bool): If True, add a logo to the center of the QR code.
#     - logo_path (str): Path to the logo image file.
#     - logo_scale (float): Scaling factor for the logo size relative to central_mask_size.
#     - central_mask_shape (str): Shape of the central mask ('circle', 'square', 'diamond').
#     - central_mask_size (int): Size parameter controlling the size of the central mask.
#     - column_width (int): Width of the columns to blank out.
#     - border_softness (int): Controls the blur of the transition.

#     Returns:
#     - PIL.Image: The generated QR code image.
#     - dict: A dictionary containing masks for different regions:
#         - 'padding_mask': Mask for the padding area.
#         - 'central_mask': Mask for the central masked area (including columns).
#         - 'qr_module_mask': Mask for the QR code modules.
#     """
#     # Initialize QR Code
#     qr = qrcode.QRCode(
#         version=version,
#         error_correction=error_correction,
#         box_size=box_size,
#         border=border,
#     )
#     qr.add_data(data)
#     qr.make(fit=True)

#     # Generate the QR code image
#     if inverted:
#         fill_color = "white"
#         back_color = "black"
#     else:
#         fill_color = "black"
#         back_color = "white"
#     img = qr.make_image(fill_color=fill_color, back_color=back_color).convert("RGB")

#     img_array = np.array(img, dtype=np.uint8)
#     height, width = img_array.shape[:2]
#     center_y = height // 2
#     center_x = width // 2

#     # Create grayscale version and binary mask of QR code modules
#     img_gray = np.array(img.convert('L'))
#     threshold = 128
#     if inverted:
#         qr_module_mask = img_gray >= threshold  # QR code modules
#     else:
#         qr_module_mask = img_gray <= threshold  # QR code modules

#     padding_mask = ~qr_module_mask  # Padding area (background)

#     # Get QR code module boundaries
#     coords = np.column_stack(np.where(qr_module_mask))
#     top_left = coords.min(axis=0)
#     bottom_right = coords.max(axis=0)
#     qr_top, qr_left = top_left
#     qr_bottom, qr_right = bottom_right

#     # Create coordinate grid using np.meshgrid
#     Y, X = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

#     # Initialize central mask
#     mask_total = np.zeros((height, width), dtype=float)

#     # Ensure border_softness is a positive value to avoid division by zero
#     if border_softness is None or border_softness <= 0:
#         border_softness = 1e-6  # Small value to prevent division by zero

#     # Create central mask
#     if central_mask_shape == 'circle':
#         distance = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
#         mask_central = np.clip((central_mask_size + border_softness - distance) / border_softness, 0, 1)
#     elif central_mask_shape == 'square':
#         distance_x = np.abs(X - center_x) - central_mask_size / 2
#         distance_y = np.abs(Y - center_y) - central_mask_size / 2
#         distance = np.maximum(distance_x, distance_y)
#         mask_central = np.clip((border_softness - distance) / border_softness, 0, 1)
#     elif central_mask_shape == 'diamond':
#         distance = np.abs(X - center_x) + np.abs(Y - center_y) - central_mask_size
#         mask_central = np.clip((border_softness - distance) / border_softness, 0, 1)
#     else:
#         raise ValueError(f"Invalid central_mask_shape: {central_mask_shape}. Should be 'circle', 'square', or 'diamond'.")

#     mask_total = np.clip(mask_total + mask_central, 0, 1)

#     # Create column masks if column_width is set
#     if column_width is not None and column_width > 0:
#         column_half_width = column_width / 2

#         # Limit columns to be within QR code module boundaries minus 1 pixel
#         vertical_start = qr_top + 2*box_size
#         vertical_end = qr_bottom - 2*box_size
#         horizontal_start = qr_left + 2*box_size
#         horizontal_end = qr_right - 2*box_size

#         # Vertical columns (left and right)
#         mask_vertical = np.zeros_like(mask_total)
#         within_vertical_range = (Y >= vertical_start) & (Y <= vertical_end)
#         distance_x = np.abs(X - center_x) - column_half_width
#         mask_vertical = np.where(
#             within_vertical_range,
#             np.clip((border_softness - distance_x) / border_softness, 0, 1),
#             0
#         )

#         # Horizontal columns (top and bottom)
#         mask_horizontal = np.zeros_like(mask_total)
#         within_horizontal_range = (X >= horizontal_start) & (X <= horizontal_end)
#         distance_y = np.abs(Y - center_y) - column_half_width
#         mask_horizontal = np.where(
#             within_horizontal_range,
#             np.clip((border_softness - distance_y) / border_softness, 0, 1),
#             0
#         )

#         # Combine vertical and horizontal masks
#         mask_columns = np.clip(mask_vertical + mask_horizontal, 0, 1)
#         mask_total = np.clip(mask_total + mask_columns, 0, 1)

#     central_mask = mask_total > 0

#     # Now, adjust the target color for the masked areas
#     # The central masked area should be the same color as the padding
#     # Determine the padding color
#     if inverted:
#         padding_color = np.array([0, 0, 0])  # Black
#     else:
#         padding_color = np.array([255, 255, 255])  # White

#     # Apply the mask to blend the image and padding color
#     img_array = img_array.astype(np.float32)
#     img_array = img_array * (1 - mask_total[:, :, np.newaxis]) + padding_color * mask_total[:, :, np.newaxis]
#     img_array = img_array.astype(np.uint8)

#     # Convert to PIL Image after masking
#     img_pil = Image.fromarray(img_array)

#     # Add logo if requested
#     if add_logo:
#         # Check if the logo file exists
#         if not os.path.isfile(logo_path):
#             raise FileNotFoundError(f"Logo file not found at path: {logo_path}")

#         # Load the logo image (ensure it's in RGBA)
#         logo_img = Image.open(logo_path).convert("RGBA")

#         # Recolor the logo based on the 'inverted' parameter
#         # Replace all non-transparent pixels with target color
#         logo_data = np.array(logo_img)

#         # Define target color based on inversion
#         if inverted:
#             target_color_logo = np.array([255, 255, 255, 255])  # White with full alpha
#         else:
#             target_color_logo = np.array([0, 0, 0, 255])        # Black with full alpha

#         # Create a mask where alpha is not zero (non-transparent pixels)
#         alpha_channel = logo_data[:, :, 3]
#         non_transparent_mask = alpha_channel > 0

#         # Replace non-transparent pixels with target color using a 2D mask
#         logo_data[non_transparent_mask] = target_color_logo

#         # Convert back to an image
#         logo_img = Image.fromarray(logo_data, mode='RGBA')

#         # Resize the logo to fit the center area, maintaining aspect ratio
#         # Calculate logo size based on central_mask_size and logo_scale
#         logo_size = int(central_mask_size * logo_scale)
#         logo_img.thumbnail((logo_size, logo_size), Image.LANCZOS)

#         # Calculate the position to place the logo
#         logo_width, logo_height = logo_img.size
#         x = center_x - logo_width // 2
#         y = center_y - logo_height // 2

#         # Paste the logo onto the QR code image with transparency mask
#         img_pil.paste(logo_img, (x, y), logo_img)

#     # Ensure the output directory exists
#     if filename:
#         output_dir = 'output_images/raw_qrs'
#         os.makedirs(output_dir, exist_ok=True)
#         img_pil.save(os.path.join(output_dir, f'{filename}.png'))

#     # Prepare masks to return
#     masks = {
#         'padding_mask': padding_mask,
#         'central_mask': central_mask,
#         'qr_module_mask': qr_module_mask
#     }

#     return img_pil, masks





# import numpy as np
# from PIL import Image
# import os

# import numpy as np
# from PIL import Image
# import os

# def add_noise_to_qr_code(
#     qr_code_image,
#     masks,
#     qr_noise_level=0.2,
#     padding_noise_level=1.0,
#     central_noise_level=1.0,
#     noise_type='uniform',
#     filename=None,
#     inverted=False
# ):
#     """
#     Adds noise to a QR code image based on specified noise levels and masks.

#     Parameters:
#     - qr_code_image (PIL.Image): The QR code image to which noise will be added.
#     - masks (dict): A dictionary containing masks from generate_qr_code function.
#         - 'padding_mask': Mask for the padding area.
#         - 'central_mask': Mask for the central masked area (including columns).
#         - 'qr_module_mask': Mask for the QR code modules.
#     - qr_noise_level (float): Noise level for the QR code modules (0 to 1).
#     - padding_noise_level (float): Noise level for the padding area (0 to 1).
#     - central_noise_level (float): Noise level for the central masked area (0 to 1).
#     - noise_type (str): Type of noise to apply ('uniform', 'gaussian', 'salt_pepper', 'speckle').
#     - filename (str): Optional filename to save the noisy QR code image.
#     - inverted (bool): If True, invert QR colors.

#     Returns:
#     - PIL.Image: The noisy QR code image.
#     """
#     qr_array = np.array(qr_code_image.convert('RGB'), dtype=np.float32)
#     height, width, channels = qr_array.shape

#     # Get the masks
#     padding_mask = masks['padding_mask']
#     central_mask = masks['central_mask']
#     qr_module_mask = masks['qr_module_mask']

#     # Remaining QR code area (QR modules excluding central masked area)
#     qr_code_area_mask = qr_module_mask & (~central_mask)

#     # Function to apply noise
#     def apply_noise(array, noise_level, mask):
#         if noise_level <= 0:
#             return array  # No noise applied

#         if noise_type == 'uniform':
#             # Generate uniform random noise
#             random_noise = np.random.randint(0, 256, size=array.shape).astype(np.float32)
#         elif noise_type == 'gaussian':
#             # Generate Gaussian noise
#             mean = 0
#             sigma = 255 * noise_level
#             random_noise = array + np.random.normal(mean, sigma, array.shape)
#             random_noise = np.clip(random_noise, 0, 255)
#         elif noise_type == 'salt_pepper':
#             # Generate salt and pepper noise
#             random_noise = array.copy()
#             num_salt = np.ceil(noise_level * mask.sum() * 0.5)
#             num_pepper = np.ceil(noise_level * mask.sum() * 0.5)

#             # Add salt (white pixels)
#             coords = [np.random.randint(0, i - 1, int(num_salt)) for i in array.shape[:2]]
#             random_noise[coords[0], coords[1], :] = 255

#             # Add pepper (black pixels)
#             coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in array.shape[:2]]
#             random_noise[coords[0], coords[1], :] = 0
#         elif noise_type == 'speckle':
#             # Generate speckle noise
#             random_noise = array + array * np.random.randn(*array.shape) * noise_level
#             random_noise = np.clip(random_noise, 0, 255)
#         else:
#             raise ValueError(f"Invalid noise_type: {noise_type}")

#         # Blend the original array with the noise based on noise_level
#         noisy_array = (1 - noise_level) * array + noise_level * random_noise
#         return np.clip(noisy_array, 0, 255)

#     # Apply noise to the QR code modules
#     qr_array[qr_code_area_mask] = apply_noise(
#         qr_array[qr_code_area_mask], qr_noise_level, qr_code_area_mask
#     )

#     # Apply noise to the central masked area
#     qr_array[central_mask] = apply_noise(
#         qr_array[central_mask], central_noise_level, central_mask
#     )

#     # Apply noise to the padding area
#     qr_array[padding_mask] = apply_noise(
#         qr_array[padding_mask], padding_noise_level, padding_mask
#     )

#     qr_array = np.clip(qr_array, 0, 255).astype(np.uint8)
#     noisy_qr_code = Image.fromarray(qr_array, 'RGB')

#     if filename:
#         output_dir = 'output_images/raw_qrs'
#         os.makedirs(output_dir, exist_ok=True)
#         noisy_qr_code.save(os.path.join(output_dir, f'{filename}.png'))

#     return noisy_qr_code

def generate_qr_code(
    data,
    filename=None,
    version=None,
    error_correction=qrcode.constants.ERROR_CORRECT_H,
    box_size=20,
    border=4,
    mask_logo=70,
    inverted=False,
    add_logo=False,
    logo_path='logo.png',  # Optional: Allow specifying logo path
    logo_scale=1.5      # Scaling factor for the logo relative to mask_logo
):
    """
    Generates a QR code with optional circular masking and logo overlay.

    Parameters:
    - data (str): Data to encode in the QR code.
    - filename (str): Optional filename to save the QR code image.
    - version (int): Version of the QR code (controls size).
    - error_correction: Error correction level from qrcode.constants.
    - box_size (int): Size of each QR code box.
    - border (int): Border size around the QR code.
    - mask_logo (int): Radius of the circular mask in pixels.
    - inverted (bool): If True, invert QR colors.
    - add_logo (bool): If True, add a logo to the center of the QR code.
    - logo_path (str): Path to the logo image file.
    - logo_scale (float): Scaling factor for the logo size relative to mask_logo.

    Returns:
    - PIL.Image: The generated QR code image.
    """
    
    # Initialize QR Code
    qr = qrcode.QRCode(
        version=version,
        error_correction=error_correction,
        box_size=box_size,
        border=border,
    )
    qr.add_data(data)
    qr.make(fit=True)
    
    # Generate the QR code image
    if inverted:
        img = qr.make_image(fill_color="white", back_color="black").convert("RGB")
    else:
        img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    
    img_array = np.array(img, dtype=np.uint8)
    height, width = img_array.shape[:2]
    center_y = height // 2
    center_x = width // 2
    
    # Apply Circular Mask if mask_logo is set
    if mask_logo:
        # Create coordinate grid
        Y, X = np.ogrid[:height, :width]
        
        # Calculate Euclidean distance from the center
        distance = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        
        # Define mask radius and smoothness
        mask_radius = mask_logo
        smooth_width = mask_logo / 1  # Adjust for smoother edges
        
        # Create smooth circular mask using a linear gradient
        mask = np.clip((mask_radius + smooth_width - distance) / smooth_width, 0, 1)
        
        # Define target color based on inversion
        if inverted:
            target_color = np.array([0, 0, 0])  # Black
        else:
            target_color = np.array([255, 255, 255])  # White
        
        # Apply the smooth mask to blend the image and target color
        img_array = img_array.astype(np.float32)
        img_array = img_array * (1 - mask[:, :, np.newaxis]) + target_color * mask[:, :, np.newaxis]
        img_array = img_array.astype(np.uint8)
    
    # Convert to PIL Image after masking
    img_pil = Image.fromarray(img_array)
    
    # Add logo if requested
    if add_logo:
        # Check if the logo file exists
        if not os.path.isfile(logo_path):
            raise FileNotFoundError(f"Logo file not found at path: {logo_path}")
        
        # Load the logo image (ensure it's in RGBA)
        logo_img = Image.open(logo_path).convert("RGBA")
        
        # Recolor the logo based on the 'inverted' parameter
        # Replace all non-transparent pixels with target color
        logo_data = np.array(logo_img)
        
        # Define target color based on inversion
        if inverted:
            target_color_logo = np.array([255, 255, 255, 255])  # White with full alpha
        else:
            target_color_logo = np.array([0, 0, 0, 255])        # Black with full alpha
        
        # Create a mask where alpha is not zero (non-transparent pixels)
        alpha_channel = logo_data[:, :, 3]
        non_transparent_mask = alpha_channel > 0
        
        # Replace non-transparent pixels with target color using a 2D mask
        logo_data[non_transparent_mask] = target_color_logo
        
        # Convert back to an image
        logo_img = Image.fromarray(logo_data, mode='RGBA')
        
        # Resize the logo to fit the center area, maintaining aspect ratio
        # Calculate logo size based on mask_logo and logo_scale
        logo_size = int(mask_logo * logo_scale)
        logo_img.thumbnail((logo_size, logo_size), Image.LANCZOS)
        
        # Calculate the position to place the logo
        logo_width, logo_height = logo_img.size
        x = center_x - logo_width // 2
        y = center_y - logo_height // 2
        
        # Paste the logo onto the QR code image with transparency mask
        img_pil.paste(logo_img, (x, y), logo_img)
    
    # Ensure the output directory exists
    if filename:
        output_dir = 'output_images/raw_qrs'
        os.makedirs(output_dir, exist_ok=True)
        img_pil.save(os.path.join(output_dir, f'{filename}.png'))
    
    return img_pil


def add_noise_to_qr_code(
    qr_code_image,
    noise_level=0.2,
    border_noise_level=1.0,
    center_noise_level=1.0,
    mask_logo=None,
    filename=None,
    inverted=False
):
    qr_array = np.array(qr_code_image.convert('RGB'), dtype=np.float32)
    height, width, channels = qr_array.shape
    qr_gray = np.array(qr_code_image.convert('L'))
    
    threshold = 128
    if inverted:
        binary_mask = qr_gray >= threshold
    else:
        binary_mask = qr_gray <= threshold
    
    coords = np.column_stack(np.where(binary_mask))
    if coords.size == 0:
        raise ValueError("QR code modules not found in the image.")
    
    top_left = coords.min(axis=0)
    bottom_right = coords.max(axis=0)
    
    inner_mask = np.zeros((height, width), dtype=bool)
    inner_mask[
        top_left[0]:bottom_right[0] + 1, top_left[1]:bottom_right[1] + 1
    ] = True
    border_mask = ~inner_mask
    
    if mask_logo is not None:
        mask_logo = int(mask_logo)
        center_y = height // 2
        center_x = width // 2

        # Create coordinate grid
        X, Y = np.meshgrid(np.arange(width), np.arange(height))

        # Compute distance from center in diamond shape
        distance = np.abs(X - center_x) + np.abs(Y - center_y)

        # Boundary layer width (one-third of mask_logo)
        boundary_width = mask_logo / 2

        # Create smooth mask for logo area
        logo_mask = np.clip((mask_logo - distance) / boundary_width, 0, 1)

        # Remove logo area from inner_mask
        inner_mask[logo_mask > 0] = False
    else:
        center_noise_level = 0.0
        logo_mask = np.zeros((height, width))  # No logo mask applied
    
    if border_noise_level > 0:
        border_noise = np.random.randint(
            0, 256, size=(height, width, channels)
        ).astype(np.float32)
        qr_array[border_mask] = (
            (1 - border_noise_level) * qr_array[border_mask]
            + border_noise_level * border_noise[border_mask]
        )
    
    if noise_level > 0:
        inner_noise = np.random.randint(
            0, 256, size=(height, width, channels)
        ).astype(np.float32)
        qr_array[inner_mask] = (
            (1 - noise_level) * qr_array[inner_mask]
            + noise_level * inner_noise[inner_mask]
        )
    
    if center_noise_level > 0 and mask_logo is not None:
        logo_noise = np.random.randint(
            0, 256, size=(height, width, channels)
        ).astype(np.float32)
        # Apply smooth mask to blend noise and original image
        for c in range(channels):
            qr_array[:, :, c] = (
                qr_array[:, :, c] * (1 - logo_mask * center_noise_level)
                + logo_noise[:, :, c] * logo_mask * center_noise_level
            )
    
    qr_array = np.clip(qr_array, 0, 255).astype(np.uint8)
    noisy_qr_code = Image.fromarray(qr_array, 'RGB')
    
    if filename:
        noisy_qr_code.save(f'output_images/raw_qrs/{filename}.png')
    
    return noisy_qr_code


import random
import string

def generate_prompt():
    verbs = [
        'integrated',
        'blended',
        'embedded',
        'merged',
        'incorporated',
        'interwoven',
        'woven',
        'fused',
        'engraved',
        'imprinted',
        'crafted',
        'designed',
        'infused',
        'surrounded',
        'encased'
    ]
    
    adjectives = [
        'futuristic',
        'ancient',
        'mystical',
        'vibrant',
        'serene',
        'abstract',
        'surreal',
        'majestic',
        'ethereal',
        'dynamic',
        'colorful',
        'monochrome',
        'minimalist',
        'intricate',
        'ornate',
        'gleaming',
        'shadowy',
        'luminous',
        'rustic',
        'industrial'
    ]
    
    subjects = [
        'cityscape',
        'forest',
        'mountain landscape',
        'underwater scene',
        'space nebula',
        'desert',
        'ocean',
        'galaxy',
        'garden',
        'skyline',
        'countryside',
        'rainforest',
        'ice cave',
        'ancient temple',
        'futuristic metropolis',
        'market scene',
        'shopping mall',
        'palace',
        'abstract vector art',
        'robotic assembly line',
        'cyberpunk alley',
        'medieval village',
        'steampunk laboratory',
        'floating island',
        'holographic display',
        'fantasy castle',
        'art deco building',
        'modern art gallery',
        'galactic battleground',
        'enchanted forest',
        'urban street'
    ]
    
    artstyles = [
        'digital art',
        'oil painting',
        'watercolor',
        'pencil sketch',
        'cyberpunk style',
        'steampunk aesthetic',
        'fantasy art',
        'minimalist design',
        'photorealistic rendering',
        'pop art',
        'impressionist painting',
        'surrealism',
        'abstract expressionism',
        'low-poly art',
        'graffiti style',
        'vector illustration',
        'pixel art',
        'vector art',
        '3D render',
        'line art'
    ]

    # Optional: Add more categories for further diversity
    themes = [
        'horror',
        'romantic',
        'sci-fi',
        'noir',
        'vintage',
        'retro',
        'modern',
        'classic',
        'baroque',
        'gothic',
        'abstract',
        'conceptual',
        'biotech',
        'eco-friendly',
        'stealth',
        'neon-lit',
        'transparent',
        'glowing',
        'metallic',
        'crystal-like'
    ]
    
    # Select random elements from each category
    verb = random.choice(verbs)
    adjective = random.choice(adjectives)
    subject = random.choice(subjects)
    artstyle = random.choice(artstyles)
    theme = random.choice(themes)
    
    # Construct the prompt
    prompt = f"A QR code, cleverly {verb} into a {adjective} {subject}, {theme} theme, {artstyle}, 3D rendered"
    
    # Generate a unique and concise filename based on selected elements
    # Replace spaces with underscores and remove special characters for filesystem compatibility
    safe_adjective = ''.join(e if e.isalnum() else '_' for e in adjective)
    safe_subject = ''.join(e if e.isalnum() else '_' for e in subject)
    safe_artstyle = ''.join(e if e.isalnum() else '_' for e in artstyle)
    safe_theme = ''.join(e if e.isalnum() else '_' for e in theme)
    
    # Generate a short random string to ensure uniqueness
    random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    
    filename = f"{safe_adjective}_{safe_subject}_{safe_artstyle}_{safe_theme}_{random_suffix}"
    
    return prompt, filename

def generate_wifi_qr_string(ssid, password):
    def escape(s):
        return s.replace('\\', '\\\\').replace(';', '\\;').replace(',', '\\,').replace(':', '\\:')
    ssid_escaped = escape(ssid)
    password_escaped = escape(password)
    qr_string = f'WIFI:S:{ssid_escaped};T:WPA;P:{password_escaped};;'
    return qr_string