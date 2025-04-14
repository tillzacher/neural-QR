import qrcode
import numpy as np
from PIL import Image
import os

if not os.path.exists("output_images"):
    os.makedirs("output_images")
if not os.path.exists("output_images/temp"):
    os.makedirs("output_images/temp")

def generate_qr_code(data, filename=None, version=None,
                     error_correction=qrcode.constants.ERROR_CORRECT_H,
                     border=4, mask_logo=4, box_size=20):
    """
    Generates a QR code with a white 'blank' center for a logo area.
    """
    qr = qrcode.QRCode(
        version=version,
        error_correction=error_correction,
        box_size=box_size,
        border=border,
    )
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image().convert("RGB")  # ensure RGB

    if mask_logo is not None:
        # 'blank out' the center region
        len_to_block = box_size * mask_logo
        img_array = np.array(img)
        height, width, _ = img_array.shape

        # define center region
        top = height // 2 - len_to_block
        bottom = height // 2 + len_to_block
        left = width // 2 - len_to_block
        right = width // 2 + len_to_block

        # fill center region with white (255, 255, 255)
        img_array[top:bottom, left:right] = (255, 255, 255)

        img = Image.fromarray(img_array, "RGB")

    if filename is not None:
        img.save(filename)

    # Save the latest QR code image
    img.save("output_images/temp/latest_clean.png")

    return img


def add_noise_to_qr_code(
    qr_code_image,
    noise_level=0.2,
    border_noise_level=1.0,
    center_noise_level=0.5,
    mask_logo=4,
    box_size=20
):
    """
    Adds three tiers of noise:
      1) border_noise_level -> outside the black module area
      2) noise_level        -> inside the black module area (excluding center)
      3) center_noise_level -> the blanked-out center region (logo area)
    """

    qr_array = np.array(qr_code_image.convert('RGB'), dtype=np.float32)
    height, width, channels = qr_array.shape

    # Convert to grayscale for thresholding black modules
    qr_gray = np.array(qr_code_image.convert('L'))
    threshold = 128
    # True where pixel > 128 (white), False where <= 128 (black)
    binary_mask = qr_gray > threshold

    # Locate bounding box of all black modules
    coords = np.column_stack(np.where(binary_mask == False))  # black = False
    if coords.size == 0:
        raise ValueError("No black modules found in the QR code.")

    top_left = coords.min(axis=0)      # (min_row, min_col)
    bottom_right = coords.max(axis=0)  # (max_row, max_col)

    # Make masks
    border_mask = np.ones((height, width), dtype=bool)
    inner_mask = np.zeros((height, width), dtype=bool)

    # The "inner_mask" is the bounding box of the black modules
    inner_mask[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1] = True
    border_mask[inner_mask] = False  # border is outside the black module area

    # Define the center/blank region if we used 'mask_logo'
    center_mask = np.zeros((height, width), dtype=bool)
    if mask_logo is not None and mask_logo > 0:
        len_to_block = box_size * mask_logo
        ctop = height // 2 - len_to_block
        cbottom = height // 2 + len_to_block
        cleft = width // 2 - len_to_block
        cright = width // 2 + len_to_block
        center_mask[ctop:cbottom, cleft:cright] = True

    # 1) Apply noise to border area
    if border_noise_level > 0:
        border_noise = np.random.randint(0, 256, size=(height, width, channels), dtype=np.uint8)
        border_noise = border_noise.astype(np.float32)
        qr_array[border_mask] = ((1 - border_noise_level) * qr_array[border_mask]
                                + border_noise_level * border_noise[border_mask])

    # 2) Apply noise to inner/black modules **excluding** center
    #    i.e. (inner_mask) AND (NOT center_mask)
    inner_mask_no_center = np.logical_and(inner_mask, np.logical_not(center_mask))

    if noise_level > 0:
        inner_noise = np.random.randint(0, 256, size=(height, width, channels), dtype=np.uint8)
        inner_noise = inner_noise.astype(np.float32)
        qr_array[inner_mask_no_center] = (
            (1 - noise_level) * qr_array[inner_mask_no_center]
            + noise_level * inner_noise[inner_mask_no_center]
        )

    # 3) Apply separate noise to the center region
    if center_noise_level > 0 and center_mask.any():
        center_noise = np.random.randint(0, 256, size=(height, width, channels), dtype=np.uint8)
        center_noise = center_noise.astype(np.float32)
        qr_array[center_mask] = (
            (1 - center_noise_level) * qr_array[center_mask]
            + center_noise_level * center_noise[center_mask]
        )

    # Convert back to uint8 and save
    qr_array = np.clip(qr_array, 0, 255).astype(np.uint8)
    noisy_qr_code = Image.fromarray(qr_array, 'RGB')
    noisy_qr_code.save("output_images/temp/latest_noisy.png")

    return noisy_qr_code