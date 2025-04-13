from PIL import Image
import os
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, DDIMScheduler
import torch
import datetime
from PIL import ImageOps

from modules.qr_code_gen import generate_qr_code, add_noise_to_qr_code

def resize_for_condition_image(input_image: Image.Image, resolution: int):
    input_image = input_image.convert("RGB")
    W, H = input_image.size
    k = float(resolution) / min(H, W)
    H = int(H * k)
    W = int(W * k)
    H = (H // 64) * 64
    W = (W // 64) * 64
    img = input_image.resize((W, H), resample=Image.LANCZOS)
    return img

def run_diffusion_on_qr_code(
    message='Hello, world!',
    prompt='A QR code, 3D rendered',
    filename='qr_code.png',
    border=10,
    mask_logo=4,

    center_noise_level=0.5,
    noise_level=0.5,
    border_noise_level=1.0,

    model_id='models/animerge_v23',
    controlnet_model_id='DionTimmer/controlnet_qrcode-control_v1p_sd15',
    resolution=768,
    seed=11111,

    invert_colors=False,
    guidance_scale=20,
    controlnet_conditioning_scale=1.5,
    strength=0.9,
    num_inference_steps=150,

    verbose=False,
):
    
    # Set the output filename
    now = datetime.datetime.now()
    # format date as YY_MM_DD
    date_str = now.strftime("%y_%m_%d")

    # make new directory if it doesn't exist
    if not os.path.exists(f"output_images/{date_str}"):
        os.makedirs(f"output_images/{date_str}")
    
    output_filename = f"output_images/{date_str}/{filename}"

    # Generate the QR code
    clean_qr = generate_qr_code(message,
                                border=border,
                                mask_logo=mask_logo)

    # Add Noise to QR code
    noisy_qr = add_noise_to_qr_code(clean_qr,
                                    noise_level=noise_level, border_noise_level=border_noise_level, center_noise_level=center_noise_level, mask_logo=mask_logo)


    if verbose:
        print(f'Running qr code diffusion with prompt: {prompt}')
        
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
    controlnet = ControlNetModel.from_pretrained(
        controlnet_model_id,
        torch_dtype=torch_dtype
    ).to(device)

    # Load the pipeline
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        model_id,
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=torch_dtype
    ).to(device)

    # Memory optimization
    if device.type == 'cuda':
        # Disable xFormers due to incompatibility
        if verbose:
            print("xFormers not compatible with this GPU; enabling attention slicing instead.")
        pipe.enable_attention_slicing()
    elif device.type == 'mps':
        pipe.enable_attention_slicing()
    else:
        pipe.enable_attention_slicing()

    # Optional: Enable other memory optimizations
    pipe.enable_model_cpu_offload()

    #pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)

    # Resize the QR code image (condition image)
    init_image = resize_for_condition_image(noisy_qr, resolution)
    condition_image = resize_for_condition_image(clean_qr, resolution)

    if invert_colors:
        init_image = init_image.convert("RGB")
        init_image = ImageOps.invert(init_image)
        # save the inverted image
        init_image.save("output_images/temp/latest_noisy.png")
        condition_image = condition_image.convert("RGB")
        condition_image = ImageOps.invert(condition_image)
        # save the inverted image
        condition_image.save("output_images/temp/latest_clean.png")
        if verbose:
            print("Inverted colors of the images")

    # Set a random seed for reproducibility
    #generator = torch.Generator(device=device).manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)

    # Generate the image
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

    final_image = result.images[0]

    final_image.save("output_images/temp/latest_diffused.png")
    final_image.save(output_filename)

    if verbose:
        print(f"Image saved to {output_filename}")