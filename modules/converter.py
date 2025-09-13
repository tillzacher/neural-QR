"""
Simplified conversion utility to convert a single Stable Diffusion
`.safetensors` checkpoint into a Diffusers pipeline directory.

This is a minimal wrapper around `diffusers.pipelines.stable_diffusion
.convert_from_ckpt.download_from_original_stable_diffusion_ckpt` and is
adapted from the original Hugging Face conversion script.

Authors: The Hugging Face Inc. team
License: Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

Notes:
- Always assumes the input is a `.safetensors` checkpoint.
- Saves the converted pipeline under the repository's `models/` directory using
  the filename (without extension) as the output folder name.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
    download_from_original_stable_diffusion_ckpt,
)


def convert_safetensors_to_diffusers(checkpoint_path):
    ckpt = Path(checkpoint_path).expanduser().resolve()
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    if ckpt.suffix.lower() != ".safetensors":
        raise ValueError(
            f"Expected a .safetensors file, got '{ckpt.suffix}' for: {ckpt}"
        )

    # Repo root is the parent of the `modules/` directory where this file lives.
    repo_root = Path(__file__).resolve().parent.parent
    models_dir = repo_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_name = ckpt.stem  # filename without extension
    out_dir = models_dir / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Minimal, opinionated defaults matching the typical CLI usage in this repo.
    pipe = download_from_original_stable_diffusion_ckpt(
        checkpoint_path_or_dict=str(ckpt),
        original_config_file=None,
        config_files=None,
        image_size=None,
        prediction_type=None,
        model_type=None,
        extract_ema=True,
        scheduler_type="pndm",
        num_in_channels=None,
        upcast_attention=False,
        from_safetensors=True,
        device="cpu",
        stable_unclip=None,
        stable_unclip_prior=None,
        clip_stats_path=None,
        controlnet=False,
        vae_path=None,
        pipeline_class=None,
    )

    # Do not force half precision here; keep weights as returned.
    pipe.save_pretrained(str(out_dir), safe_serialization=True)

    return str(out_dir)

