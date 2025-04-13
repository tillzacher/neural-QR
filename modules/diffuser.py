import torch
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, DDIMScheduler
from diffusers.utils import load_image
from PIL import Image
import os
from pathlib import Path
import warnings
import logging

# Suppress all Python warnings
warnings.filterwarnings("ignore")

# Configure the root logger to suppress unwanted logs
logging.basicConfig(level=logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("diffusers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("PIL").setLevel(logging.ERROR)

def resize_for_condition_image(input_image: Image.Image, resolution: int):
    """
    Resizes the input image to the desired resolution while maintaining aspect ratio
    and ensuring dimensions are multiples of 64.

    Parameters:
    - input_image (Image.Image): The image to resize.
    - resolution (int): The target resolution.

    Returns:
    - Image.Image: The resized image.
    """
    input_image = input_image.convert("RGB")
    W, H = input_image.size
    k = float(resolution) / min(H, W)
    H = int(H * k)
    W = int(W * k)
    H = (H // 64) * 64
    W = (W // 64) * 64
    img = input_image.resize((W, H), resample=Image.LANCZOS)
    return img

from PIL import ImageOps
from modules.input_gen import add_noise_to_qr_code  # Ensure this module is correctly imported

# def run_diffusion_on_qr_code(
#     image_of_qr: Image.Image,
#     prompt: str,
#     output_filename: str = 'diffusion_output',
#     model_id: str = 'models/animerge_v23',
#     controlnet_model_id: str = 'DionTimmer/controlnet_qrcode-control_v1p_sd15',
#     strength: float = 0.9,
#     guidance_scale: float = 20.0,
#     controlnet_conditioning_scale: float = 1.5,
#     num_inference_steps: int = 150,
#     resolution: int = 768,
#     seed: int = 11111,
#     noise_params = None,
#     verbose: bool = False,
#     external_output_dir: str = r"C:\Users\Napi\iCloudDrive\generated_qr_codes"  # Absolute external path
# ):
#     """
#     Generates a diffusion-enhanced QR code image and saves it to both internal and external directories.

#     Parameters:
#     - image_of_qr (Image.Image): The original QR code image.
#     - prompt (str): The text prompt for the diffusion model.
#     - output_filename (str): The base name for the output image files. Can include subdirectories (e.g., 'hd/my_image').
#     - model_id (str): The identifier for the diffusion model.
#     - controlnet_model_id (str): The identifier for the ControlNet model.
#     - strength (float): The strength of the initial image influence.
#     - guidance_scale (float): The scale for classifier-free guidance.
#     - controlnet_conditioning_scale (float): The scale for ControlNet conditioning.
#     - num_inference_steps (int): The number of inference steps for the diffusion process.
#     - resolution (int): The target resolution for the images.
#     - seed (int): The random seed for reproducibility.
#     - noise_level (float): The noise level to add to the QR code.
#     - border_noise_level (float): The noise level for the borders.
#     - center_noise_level (float): The noise level for the center.
#     - mask_logo (int): The size of the mask/logo area.
#     - inverted (bool): Whether to invert the QR code colors.
#     - verbose (bool): Whether to print detailed logs.
#     - external_output_dir (str): The absolute path to the external output directory.

#     Returns:
#     - None
#     """
    
#     # Define internal and external output paths using pathlib for robust path handling
#     internal_output_path = Path('output_images')
#     external_output_path = Path(external_output_dir)
    
#     # Create internal output directory if it doesn't exist
#     try:
#         internal_output_path.mkdir(parents=True, exist_ok=True)
#         if verbose:
#             print(f"Ensured internal directory exists: {internal_output_path.resolve()}")
#     except Exception as e:
#         print(f"Error creating internal output directory '{internal_output_path}': {e}")
#         return  # Exit the function if directory creation fails
    
#     # Prepare the full internal output filenames
#     internal_output_filename = internal_output_path / f"{output_filename}.png"
#     latest_internal_output = internal_output_path / "latest_diffusion_output.png"
    
#     # Prepare the full external output filenames
#     external_output_filename = external_output_path / f"{output_filename}.png"
#     latest_external_output = external_output_path / "latest_diffusion_output.png"
    
#     # Ensure that the parent directories for external output exist
#     try:
#         external_output_filename.parent.mkdir(parents=True, exist_ok=True)
#         if verbose:
#             print(f"Ensured external directory exists: {external_output_filename.parent.resolve()}")
#     except Exception as e:
#         print(f"Error creating external output directory '{external_output_filename.parent}': {e}")
#         return  # Exit the function if directory creation fails
    
#     if verbose:
#         print(f'Running QR code diffusion with prompt: "{prompt}"')
    
#     # Check for available devices
#     if torch.cuda.is_available():
#         device = torch.device("cuda")
#         torch_dtype = torch.float16  # CUDA supports float16 for performance
#         if verbose:
#             print("Using CUDA device")
#     elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
#         device = torch.device("mps")
#         torch_dtype = torch.float32  # MPS supports float32
#         if verbose:
#             print("Using MPS device")
#     else:
#         device = torch.device("cpu")
#         torch_dtype = torch.float32  # CPU uses float32
#         if verbose:
#             print("Using CPU device")

#     # Load ControlNet model
#     if verbose:
#         print("Loading ControlNet model...")
#     try:
#         controlnet = ControlNetModel.from_pretrained(
#             controlnet_model_id,
#             torch_dtype=torch_dtype
#         ).to(device)
#     except Exception as e:
#         print(f"Error loading ControlNet model '{controlnet_model_id}': {e}")
#         return

#     # Load the pipeline
#     if verbose:
#         print("Loading Stable Diffusion ControlNet Img2Img Pipeline...")
#     try:
#         pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
#             model_id,
#             controlnet=controlnet,
#             safety_checker=None,
#             torch_dtype=torch_dtype
#         ).to(device)
#     except Exception as e:
#         print(f"Error loading pipeline with model '{model_id}': {e}")
#         return

#     # Set the scheduler
#     if device.type == 'cuda':
#         # Disable xFormers due to incompatibility
#         if verbose:
#             print("xFormers not compatible with this GPU; enabling attention slicing instead.")
#         pipe.enable_attention_slicing()
#     else:
#         pipe.enable_attention_slicing()

#     # Optional: Enable other memory optimizations
#     pipe.enable_model_cpu_offload()
    
#     # Set the scheduler to DDIM
#     pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

#     # Resize the QR code image (condition image)
#     condition_image = resize_for_condition_image(image_of_qr, resolution)
#     if verbose:
#         print(f"Condition image resized to: {condition_image.size}")

#     # Prepare the initial image with noise
#     if verbose:
#         print("Adding noise to the QR code image...")
#     init_image = image_of_qr.copy()
#     init_image = add_noise_to_qr_code(qr_code_image=init_image, **noise_params)
    
#     init_image = resize_for_condition_image(init_image, resolution)
#     if verbose:
#         print(f"Initial image resized to: {init_image.size}")

#     # Set a random seed for reproducibility
#     generator = torch.Generator(device=device).manual_seed(seed)

#     if verbose:
#         print("Starting the diffusion process...")
    
#     # Generate the image
#     try:
#         result = pipe(
#             prompt=prompt,
#             negative_prompt="ugly, disfigured, low quality, blurry",
#             image=init_image,
#             control_image=condition_image,
#             width=condition_image.width,
#             height=condition_image.height,
#             guidance_scale=guidance_scale,
#             controlnet_conditioning_scale=controlnet_conditioning_scale,
#             generator=generator,
#             strength=strength,
#             num_inference_steps=num_inference_steps,
#         )
#     except Exception as e:
#         print(f"Error during the diffusion process: {e}")
#         return
    
#     # Save the result to internal path
#     try:
#         result.images[0].save(internal_output_filename)
#         result.images[0].save(latest_internal_output)
#         if verbose:
#             print(f"Image saved to internal path: {internal_output_filename.resolve()}")
#             print(f"Latest internal image updated at: {latest_internal_output.resolve()}")
#     except Exception as e:
#         print(f"Error saving image to internal path '{internal_output_filename}': {e}")
    
#     # Save the result to external path
#     try:
#         result.images[0].save(external_output_filename)
#         result.images[0].save(latest_external_output)
#         if verbose:
#             print(f"Image saved to external path: {external_output_filename.resolve()}")
#             print(f"Latest external image updated at: {latest_external_output.resolve()}")
#     except Exception as e:
#         print(f"Error saving image to external path '{external_output_filename}': {e}")

def run_diffusion_on_qr_code(
    image_of_qr: Image.Image,
    prompt: str,
    output_filename: str = 'diffusion_output',
    model_id: str = 'models/animerge_v23',
    controlnet_model_id: str = 'DionTimmer/controlnet_qrcode-control_v1p_sd15',
    strength: float = 0.9,
    guidance_scale: float = 20.0,
    controlnet_conditioning_scale: float = 1.5,
    num_inference_steps: int = 150,
    resolution: int = 768,
    noise_level: float = 0.1,
    border_noise_level: float = 0.1,
    center_noise_level: float = 0.1,
    mask_logo: int = 0,
    inverted: bool = False,
    seed: int = 11111,
    verbose: bool = False,
    external_output_dir: str = r"C:\Users\Napi\iCloudDrive\generated_qr_codes"  # Absolute external path
):
    """
    Generates a diffusion-enhanced QR code image and saves it to both internal and external directories.

    Parameters:
    - image_of_qr (Image.Image): The original QR code image.
    - prompt (str): The text prompt for the diffusion model.
    - output_filename (str): The base name for the output image files. Can include subdirectories (e.g., 'hd/my_image').
    - model_id (str): The identifier for the diffusion model.
    - controlnet_model_id (str): The identifier for the ControlNet model.
    - strength (float): The strength of the initial image influence.
    - guidance_scale (float): The scale for classifier-free guidance.
    - controlnet_conditioning_scale (float): The scale for ControlNet conditioning.
    - num_inference_steps (int): The number of inference steps for the diffusion process.
    - resolution (int): The target resolution for the images.
    - seed (int): The random seed for reproducibility.
    - noise_level (float): The noise level to add to the QR code.
    - border_noise_level (float): The noise level for the borders.
    - center_noise_level (float): The noise level for the center.
    - mask_logo (int): The size of the mask/logo area.
    - inverted (bool): Whether to invert the QR code colors.
    - verbose (bool): Whether to print detailed logs.
    - external_output_dir (str): The absolute path to the external output directory.

    Returns:
    - None
    """
    
    # Define internal and external output paths using pathlib for robust path handling
    internal_output_path = Path('output_images')
    external_output_path = Path(external_output_dir)
    
    # Create internal output directory if it doesn't exist
    try:
        internal_output_path.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"Ensured internal directory exists: {internal_output_path.resolve()}")
    except Exception as e:
        print(f"Error creating internal output directory '{internal_output_path}': {e}")
        return  # Exit the function if directory creation fails
    
    # Prepare the full internal output filenames
    internal_output_filename = internal_output_path / f"{output_filename}.png"
    latest_internal_output = internal_output_path / "latest_diffusion_output.png"
    
    # Prepare the full external output filenames
    external_output_filename = external_output_path / f"{output_filename}.png"
    latest_external_output = external_output_path / "latest_diffusion_output.png"
    
    # Ensure that the parent directories for external output exist
    try:
        external_output_filename.parent.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"Ensured external directory exists: {external_output_filename.parent.resolve()}")
    except Exception as e:
        print(f"Error creating external output directory '{external_output_filename.parent}': {e}")
        return  # Exit the function if directory creation fails
    
    if verbose:
        print(f'Running QR code diffusion with prompt: "{prompt}"')
    
    # Check for available devices
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch_dtype = torch.float16  # CUDA supports float16 for performance
        if verbose:
            print("Using CUDA device")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        torch_dtype = torch.float32  # MPS supports float32
        if verbose:
            print("Using MPS device")
    else:
        device = torch.device("cpu")
        torch_dtype = torch.float32  # CPU uses float32
        if verbose:
            print("Using CPU device")

    # Load ControlNet model
    if verbose:
        print("Loading ControlNet model...")
    try:
        controlnet = ControlNetModel.from_pretrained(
            controlnet_model_id,
            torch_dtype=torch_dtype
        ).to(device)
    except Exception as e:
        print(f"Error loading ControlNet model '{controlnet_model_id}': {e}")
        return

    # Load the pipeline
    if verbose:
        print("Loading Stable Diffusion ControlNet Img2Img Pipeline...")
    try:
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            model_id,
            controlnet=controlnet,
            safety_checker=None,
            torch_dtype=torch_dtype
        ).to(device)
    except Exception as e:
        print(f"Error loading pipeline with model '{model_id}': {e}")
        return

    # Set the scheduler
    if device.type == 'cuda':
        # Disable xFormers due to incompatibility
        if verbose:
            print("xFormers not compatible with this GPU; enabling attention slicing instead.")
        pipe.enable_attention_slicing()
    else:
        pipe.enable_attention_slicing()

    # Optional: Enable other memory optimizations
    pipe.enable_model_cpu_offload()
    
    # Set the scheduler to DDIM
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # Resize the QR code image (condition image)
    condition_image = resize_for_condition_image(image_of_qr, resolution)
    if verbose:
        print(f"Condition image resized to: {condition_image.size}")

    # Prepare the initial image with noise
    if verbose:
        print("Adding noise to the QR code image...")
    init_image = image_of_qr.copy()
    init_image = add_noise_to_qr_code(image_of_qr, noise_level, border_noise_level, center_noise_level, mask_logo, inverted=inverted, filename='latest_noisy_qr')
    
    init_image = resize_for_condition_image(init_image, resolution)
    if verbose:
        print(f"Initial image resized to: {init_image.size}")

    # Set a random seed for reproducibility
    generator = torch.Generator(device=device).manual_seed(seed)

    if verbose:
        print("Starting the diffusion process...")
    
    # Generate the image
    try:
        result = pipe(
            prompt=prompt,
            negative_prompt="ugly, disfigured, low quality, blurry",
            image=init_image,
            control_image=condition_image,
            width=condition_image.width,
            height=condition_image.height,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            generator=generator,
            strength=strength,
            num_inference_steps=num_inference_steps,
        )
    except Exception as e:
        print(f"Error during the diffusion process: {e}")
        return
    
    # Save the result to internal path
    try:
        result.images[0].save(internal_output_filename)
        result.images[0].save(latest_internal_output)
        if verbose:
            print(f"Image saved to internal path: {internal_output_filename.resolve()}")
            print(f"Latest internal image updated at: {latest_internal_output.resolve()}")
    except Exception as e:
        print(f"Error saving image to internal path '{internal_output_filename}': {e}")
    
    # Save the result to external path
    try:
        result.images[0].save(external_output_filename)
        result.images[0].save(latest_external_output)
        if verbose:
            print(f"Image saved to external path: {external_output_filename.resolve()}")
            print(f"Latest external image updated at: {latest_external_output.resolve()}")
    except Exception as e:
        print(f"Error saving image to external path '{external_output_filename}': {e}")