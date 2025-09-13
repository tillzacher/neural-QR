from PIL import Image
import os
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, DDIMScheduler
import torch
import datetime
from PIL import ImageOps
import warnings
from modules.input_gen import generate_prompt
from modules.qr_code_gen import generate_qr_code, add_noise_to_qr_code
import platform
import subprocess

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
    model_id='models/photon_v1',
    controlnet_model_id='monster-labs/control_v1p_sd15_qrcode_monster',
    resolution=768,
    seed=11111,
    invert_colors=False,
    guidance_scale=20,
    controlnet_conditioning_scale=1.5,
    strength=0.9,
    num_inference_steps=150,
    verbose=False,
    continuous=False,
    make_gif=True,
    final_static_frames=30
):
    # Create a date folder for outputs (updates once per function call)
    now = datetime.datetime.now()
    date_str = now.strftime("%y_%m_%d")
    output_dir = f"output_images/{date_str}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # ------------------------------------------------------------
    # Device & dtype selection – CUDA ➜ MPS ➜ CPU (fallback)
    # ------------------------------------------------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch_dtype = torch.float16          # fastest & smallest on GPU
        if verbose:
            print("Using CUDA device")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        torch_dtype = torch.float32          # float16 isn’t fully supported on MPS
        if verbose:
            print("Using MPS device")
    else:
        device = torch.device("cpu")         # 🚥 graceful fallback
        torch_dtype = torch.float32
        if verbose:
            print("Using CPU device")

    # ------------------------------------------------------------
    # Load ControlNet model and pipeline (only once per call)
    # ------------------------------------------------------------
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

    # Memory-/speed tweaks
    pipe.enable_attention_slicing()

    # Off-load large sub-modules to CPU *only* when we’re running on a GPU.
    if device.type == "cuda":
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

        print(f'Using prompt: {cur_prompt}')

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
            # save the inverted image
            init_image.save("output_images/temp/latest_noisy.png")
            condition_image = ImageOps.invert(condition_image.convert("RGB"))
            # save the inverted image
            condition_image.save("output_images/temp/latest_clean.png")

        # Set a random seed (could update per iteration if desired).
        if device.type == "mps":
            gen = torch.Generator().manual_seed(seed)
        else:
            gen = torch.Generator(device=device).manual_seed(seed)
        
        # make sure the directory exists
        anim_parent = "output_images/temp/anim"
        os.makedirs(anim_parent, exist_ok=True)
        # Find all subfolders that consist only of digits.
        existing_folders = [
            d for d in os.listdir(anim_parent)
            if os.path.isdir(os.path.join(anim_parent, d)) and d.isdigit()
        ]
        if existing_folders:
            next_folder_num = max(int(d) for d in existing_folders) + 1
        else:
            next_folder_num = 0
        current_anim_folder = os.path.join(anim_parent, f"{next_folder_num:03d}")
        os.makedirs(current_anim_folder, exist_ok=True)

        def save_intermediate(step: int, timestep: int, latents: torch.Tensor):
            if num_inference_steps >= 100:
                # save only every 4nd step
                if step % 4 != 0:
                    return
            elif num_inference_steps >= 50:
                # save only every 2nd step
                if step % 2 != 0:
                    return
            if verbose:
                print(f"Step {step}, latent mean: {latents.mean().item()}, std: {latents.std().item()}")
            intermediate_imgs = pipe.decode_latents(latents)
            pil_img = pipe.numpy_to_pil(intermediate_imgs)[0]
            # Optionally, overwrite a generic intermediate image.
            pil_img.save("output_images/temp/latest_intermediate.png")
            # Format the step number with leading zeros.
            step_str = str(step).zfill(3)
            # Construct the full path for the intermediate image in the new folder.
            out_path = os.path.join(current_anim_folder, f"latest_intermediate_{step_str}.png")
            pil_img.save(out_path)
            if verbose:
                print(f"Intermediate image saved at step {step_str} in folder {current_anim_folder}")

        # Run the pipeline with the callback.
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
            callback=save_intermediate,  # Call this function every step.
            callback_steps=1,  # Call after each diffusion step.
        )
        final_image = result.images[0]
        # Save outputs.
        final_image.save("output_images/temp/latest_diffused.png")
        final_image.save(out_filename)
        if verbose:
            print(f"Image saved to {out_filename}")
        

        if platform.system() == "Windows":
            icloud_target = r"C:\Users\Schweini\iCloudDrive\qrs"
            # Using robocopy to copy all output images recursively, excluding older files.
            cmd = f'robocopy "output_images" "{icloud_target}" /E /XO'
            proc = subprocess.run(["powershell", "-Command", cmd], capture_output=True, text=True)
            # Robocopy exit codes:
            # 0: No files were copied.
            # 1: Some files were copied.
            # 2: Extra files or mismatches were detected.
            # 3: Files were copied and extra files were detected.
            # Exit codes less than 8 are success.
            if proc.returncode >= 8:
                print("Robocopy reported an error. Return code:", proc.returncode)
                print("stdout:", proc.stdout)
                print("stderr:", proc.stderr)

        # If the gif toggle is enabled, compose all intermediate images into an animated GIF.
        if make_gif:
            import glob
            from PIL import Image
            # Use glob to find all intermediate images saved in current_anim_folder.
            pattern = os.path.join(current_anim_folder, "latest_intermediate_*.png")
            intermediate_files = sorted(glob.glob(pattern))
            if intermediate_files:
                # Remove the last intermediate file if there is more than one,
                # because that one is likely the final image (already in final_image).
                if len(intermediate_files) > 1:
                    intermediate_files = intermediate_files[:-1]
                frames = []
                for img_file in intermediate_files:
                    frames.append(Image.open(img_file))
                # Append the final image repeatedly for final_static_frames times.
                for _ in range(final_static_frames):
                    frames.append(final_image)
                
                # remove the last 4 characters from the output filename
                gif_out = out_filename[:-4] + ".gif"
                # Save the frames as an animated GIF (duration in milliseconds, loop=0 means infinite loop).
                frames[0].save(gif_out, save_all=True, append_images=frames[1:], loop=0, duration=100)
                if verbose:
                    print(f"Animated GIF saved to {gif_out}")
                # Finally delete the intermediate images.
                for img_file in intermediate_files:
                    os.remove(img_file)

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
                 Any keys that are also provided in sweep_params will be removed from kwargs.
                 
    For each combination from sweep_params, run_diffusion_on_qr_code is called and the filename
    is augmented with a suffix indicating the parameter values.
    """
    # Remove keys from kwargs that are also in sweep_params so that only sweep values are used.
    for key in sweep_params.keys():
        kwargs.pop(key, None)

    keys = list(sweep_params.keys())
    value_lists = list(sweep_params.values())
    
    for combination in itertools.product(*value_lists):
        current_kwargs = dict(kwargs)
        suffix = ""
        # Update current_kwargs with each sweep key's value and build filename suffix.
        for k, val in zip(keys, combination):
            current_kwargs[k] = val
            suffix += f"_{k}{val}"
        # Append suffix to filename.
        if "filename" in current_kwargs:
            current_kwargs["filename"] = current_kwargs["filename"] + suffix
        else:
            current_kwargs["filename"] = "output" + suffix
            
        print("Running with parameters:", ", ".join(f"{k}={v}" for k, v in zip(keys, combination)))
        run_diffusion_on_qr_code(**current_kwargs)