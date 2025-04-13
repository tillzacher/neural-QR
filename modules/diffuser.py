from PIL import Image
import os
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, DDIMScheduler
import torch
import datetime
from PIL import ImageOps
import warnings
from modules.input_gen import generate_prompt
from modules.qr_code_gen import generate_qr_code, add_noise_to_qr_code

# Ignore specific warnings
warnings.filterwarnings(
    "ignore",
    message=r"Defaulting to unsafe serialization\. Pass `allow_pickle=False` to raise an error instead\."
)
warnings.filterwarnings(
    "ignore",
    message=r"An error occurred while trying to fetch models/animerge_v23/unet:"
)
warnings.filterwarnings(
    "ignore",
    message=r"An error occurred while trying to fetch models/animerge_v23/vae:"
)
warnings.filterwarnings(
    "ignore",
    message=r"You have disabled the safety checker for"
)

def resize_for_condition_image(input_image: Image.Image, resolution: int):
    input_image = input_image.convert("RGB")
    W, H = input_image.size
    k = float(resolution) / min(H, W)
    H = int(H * k)
    W = int(W * k)
    H = (H // 64) * 64
    W = (W // 64) * 64
    return input_image.resize((W, H), resample=Image.LANCZOS)

def run_diffusion_on_qr_code(
    message='Hello, world!',
    prompt='A QR code, 3D rendered',
    filename='qr_code',
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
    continuous=False
):
    # Create a date folder for outputs (updates once per function call)
    now = datetime.datetime.now()
    date_str = now.strftime("%y_%m_%d")
    output_dir = f"output_images/{date_str}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Device selection and type assignment.
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch_dtype = torch.float16
        if verbose:
            print("Using CUDA device")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        torch_dtype = torch.float32
        if verbose:
            print("Using MPS device")
    else:
        device = torch.device("cpu")
        torch_dtype = torch.float32
        if verbose:
            print("Using CPU device")

    # Load ControlNet model and pipeline (only once).
    controlnet = ControlNetModel.from_pretrained(
        controlnet_model_id,
        torch_dtype=torch_dtype
    ).to(device)
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        model_id,
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=torch_dtype
    ).to(device)

    # Enable attention slicing.
    pipe.enable_attention_slicing()
    if device.type != "mps":
        pipe.enable_model_cpu_offload()

    # Setup scheduler and move models to device.
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    if device.type == "mps":
        pipe.text_encoder.to("cpu")
    else:
        pipe.text_encoder.to(device)
    pipe.vae.to(device)
    pipe.unet.to(device)

    # For MPS, monkey-patch encode_prompt so that text encoding runs on CPU then moves results to MPS.
    if device.type == "mps":
        original_encode_prompt = pipe.encode_prompt
        def new_encode_prompt(prompt, _device, num_images_per_prompt, do_classifier_free_guidance,
                              negative_prompt=None, prompt_embeds=None, negative_prompt_embeds=None,
                              lora_scale=None, clip_skip=None):
            # Force encoding on CPU.
            prompt_embeds, negative_prompt_embeds = original_encode_prompt(
                prompt,
                "cpu",
                num_images_per_prompt,
                do_classifier_free_guidance,
                negative_prompt=negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                lora_scale=lora_scale,
                clip_skip=clip_skip,
            )
            # Move embeddings to MPS.
            prompt_embeds = prompt_embeds.to("mps")
            if negative_prompt_embeds is not None:
                negative_prompt_embeds = negative_prompt_embeds.to("mps")
            return prompt_embeds, negative_prompt_embeds
        pipe.encode_prompt = new_encode_prompt

    def generate_and_save():
        # If continuous is set, generate a new prompt and filename.
        cur_prompt = prompt
        cur_filename = filename
        if continuous:
            cur_prompt, cur_filename = generate_prompt()
        out_filename = f"{output_dir}/{cur_filename}.png"

        # Generate the QR code and add noise.
        clean_qr = generate_qr_code(message, border=border, mask_logo=mask_logo)
        noisy_qr = add_noise_to_qr_code(
            clean_qr,
            noise_level=noise_level,
            border_noise_level=border_noise_level,
            center_noise_level=center_noise_level,
            mask_logo=mask_logo
        )

        # Resize images for conditioning.
        init_image = resize_for_condition_image(noisy_qr, resolution)
        condition_image = resize_for_condition_image(clean_qr, resolution)
        if invert_colors:
            init_image = ImageOps.invert(init_image.convert("RGB"))
            condition_image = ImageOps.invert(condition_image.convert("RGB"))

        # Set a random seed (could update per iteration if desired).
        if device.type == "mps":
            gen = torch.Generator().manual_seed(seed)
        else:
            gen = torch.Generator(device=device).manual_seed(seed)

        # Run the pipeline.
        result = pipe(
            prompt=cur_prompt,
            negative_prompt="ugly, disfigured, low quality, blurry",
            image=init_image,
            control_image=condition_image,
            width=condition_image.width,
            height=condition_image.height,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            generator=gen,
            strength=strength,
            num_inference_steps=num_inference_steps,
        )
        final_image = result.images[0]
        # Save outputs.
        final_image.save("output_images/temp/latest_diffused.png")
        final_image.save(out_filename)
        if verbose:
            print(f"Image saved to {out_filename}")

    # If continuous is not set, generate a single image.
    if not continuous:
        generate_and_save()
    else:
        # Loop forever generating images with new prompt/filename.
        while True:
            generate_and_save()
            # (Optional) add a delay here if desired, e.g.,
            # time.sleep(5)


import itertools

def run_parameter_sweep(sweep_params: dict, **kwargs):
    """
    Runs the diffusion generation repeatedly for each combination of parameters in sweep_params.
    
    Parameters:
        sweep_params (dict): A dictionary mapping parameter names (as strings) to lists of values.
            For example: {"guidance_scale": [1, 5, 10, 15, 20]}
        **kwargs: Other keyword arguments to pass to run_diffusion_on_qr_code.
                 These will be used as the constant parameters.
                 
    For each combination from sweep_params, run_diffusion_on_qr_code will be called and the filename
    will be augmented with a suffix indicating the parameter values.
    """
    # Get the parameter names and lists of values.
    keys = list(sweep_params.keys())
    value_lists = list(sweep_params.values())
    
    for combination in itertools.product(*value_lists):
        # Make a copy of kwargs for this iteration.
        current_kwargs = dict(kwargs)
        suffix = ""
        # Update current_kwargs with the current combination values and build filename suffix.
        for k, val in zip(keys, combination):
            current_kwargs[k] = val
            suffix += f"_{k}{val}"
        # Update the filename in current_kwargs with the suffix.
        if "filename" in current_kwargs:
            current_kwargs["filename"] = current_kwargs["filename"] + suffix
        else:
            current_kwargs["filename"] = "output" + suffix
            
        print("Running with parameters:", ", ".join(f"{k}={v}" for k, v in zip(keys, combination)))
        # Call your diffusion function.
        run_diffusion_on_qr_code(**current_kwargs)