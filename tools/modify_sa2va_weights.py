import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    import json
    import os
    import shutil
    from pathlib import Path
    return Path, json, os, safe_open, save_file, torch


@app.cell
def _(Path):
    hf_dir = Path("/alexey/work_dirs/sa2va_4b_6_hf")
    safetensors_json = hf_dir / "model.safetensors.index.json"
    sam2_checkpoint_path = "/alexey/saved/sam2/sam2_hiera_large.pt"
    return hf_dir, safetensors_json, sam2_checkpoint_path


@app.cell
def _(Path, json):
    def update_keys_in_weight_map(input_filename, output_filename="model.safetensors.index.json.updated"):
        with open(input_filename, 'r') as f:
            data = json.load(f)

        if 'weight_map' in data and isinstance(data['weight_map'], dict):
            weight_map = data['weight_map']
            new_weight_map = {}

            sam_mask_decoder_safetensor_chunk = list()
            for key, value in weight_map.items():
                segments = key.split('.')
                updated_segments = [seg if seg != "sam_mask_decoder" else "sam_mask_decoder_modified" for seg in segments]
                new_key = '.'.join(updated_segments)
                new_weight_map[key] = value
                if new_key != key:
                    new_weight_map[new_key] = value
                    sam_mask_decoder_safetensor_chunk.append(value)

            safetensor_chunk = set(sam_mask_decoder_safetensor_chunk)
            if len(safetensor_chunk) > 1:
                raise FileNotFoundError("Weights should be in the same chunk.")
            safetensor_chunk = list(safetensor_chunk)[0]

            data['weight_map'] = new_weight_map

        with open(Path(input_filename).parent / output_filename, 'w') as f:
            json.dump(data, f, indent=2)

        return safetensor_chunk
    return (update_keys_in_weight_map,)


@app.cell
def _(safetensors_json, update_keys_in_weight_map):
    safetensor_chunk = update_keys_in_weight_map(safetensors_json)
    return (safetensor_chunk,)


@app.cell
def _(os, safe_open, save_file, torch):
    def append_mask_decoder_keys(
        safetensors_path: str,
        checkpoint_path: str,
        output_path: str = None
    ):
        """
        Appends 'mask_decoder' keys from PyTorch checkpoint to SafeTensors file.
        Renames existing keys with '_modified' suffix in the 'mask_decoder' substring.
        Args:
            safetensors_path: Path to input SafeTensors file
            checkpoint_path: Path to PyTorch checkpoint file
            output_path: Output path (default: appends '_modified' to input filename)
        """
        # Set default output path if not provided
        if output_path is None:
            base, ext = os.path.splitext(safetensors_path)
            output_path = f"{base}_modified{ext}"

        # 1. Load existing SafeTensors data
        existing_tensors = {}
        mask_decoder_keys = list()
        prefix = "grounding_encoder.sam2_model."
        with safe_open(safetensors_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if "mask_decoder" in key:
                    mask_decoder_keys.append(key)
                existing_tensors[key] = f.get_tensor(key)
            metadata = f.metadata()

        initial_number_of_keys = len(existing_tensors)

        # 2. Load PyTorch checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']

        # 3. Process mask_decoder keys
        for sam2_key, sam2_tensor in checkpoint["model"].items():
            if "mask_decoder" not in sam2_key:
                continue

            # Handle existing keys
            key = prefix + sam2_key
            if key in mask_decoder_keys:
                new_key = key.replace("sam_mask_decoder", "sam_mask_decoder_modified")

                # Rename existing tensor
                existing_tensors[new_key] = existing_tensors.pop(key)
                print(f"♻️ Renamed existing key: {key} -> {new_key}")

            # Add new tensor (either new key or replacing renamed key)
            existing_tensors[key] = sam2_tensor
            print(f"✅ Added key: {key}")

        # 4. Save updated tensors
        save_file(existing_tensors, output_path, metadata=metadata)
        print(f"\n💾 Saved updated SafeTensors to: {output_path}")
        print(f"Total keys: {len(existing_tensors)}, mask decoder keys {len(mask_decoder_keys)}, inital num keys: {initial_number_of_keys}")

    return (append_mask_decoder_keys,)


@app.cell
def _(
    append_mask_decoder_keys,
    hf_dir,
    safetensor_chunk,
    sam2_checkpoint_path,
):
    safetensor_chunk_path = hf_dir / safetensor_chunk
    append_mask_decoder_keys(
        safetensors_path=safetensor_chunk_path,
        checkpoint_path=sam2_checkpoint_path,
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
