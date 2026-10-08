"""
Encode ImageNet-1k (train) into the two files train.py reads:

    <out>/latents.npy   (N, 8, 32, 32) float32   SD-VAE posterior mean + logvar per image
    <out>/labels.npy    (N,)           int64     class index 0..999

Source: the parquet shards of the Hugging Face dataset ILSVRC/imagenet-1k
(data/train-00000-of-00294.parquet ... 294 shards of ~500MB, 1,281,167 images).
Shards are handled one at a time, so the full 147GB never has to sit on disk:

    # download each shard, encode it, delete it  (needs internet where this runs)
    python encode_latents.py --out /media/ltnghia33/sit --src /media/ltnghia33/imagenet_parquet \
        --hf-download --delete-shards

    # shards are already (or are being) downloaded by something else, e.g. download_shards.sh
    python encode_latents.py --out /media/ltnghia33/sit --src /media/ltnghia33/imagenet_parquet \
        --wait --delete-shards

Safe to interrupt: finished shards are recorded in <out>/encode_progress.json and skipped on the
next start. The output is called latents.npy.partial until every shard is done.

Preprocessing is the one SiT/DiT use: ADM center crop to 256, scale to [-1, 1], no flip.
"""
import argparse
import io
import json
import os
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

IMAGENET_TRAIN_SIZE = 1_281_167


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]


class BytesImages(Dataset):
    """Decodes a list of encoded image bytes (one parquet row group) in DataLoader workers."""
    def __init__(self, blobs, image_size):
        self.blobs = blobs
        self.image_size = image_size

    def __len__(self):
        return len(self.blobs)

    def __getitem__(self, i):
        img = Image.open(io.BytesIO(self.blobs[i])).convert("RGB")   # handles the grayscale/CMYK/PNG files
        arr = center_crop_arr(img, self.image_size)                  # (H, W, 3) uint8
        return torch.from_numpy(arr).permute(2, 0, 1)                # uint8, normalised on the GPU


def shard_name(i, n):
    return f"train-{i:05d}-of-{n:05d}.parquet"


def locate(src, name):
    for p in (os.path.join(src, name), os.path.join(src, "data", name)):
        if os.path.isfile(p):
            return p
    return None


def open_parquet(path):
    """Returns a ParquetFile, or None while the file is still being written."""
    import pyarrow.parquet as pq
    try:
        return pq.ParquetFile(path)
    except Exception:
        return None


def get_shard(args, name):
    """Local file if present; otherwise download it (--hf-download) or wait for it (--wait)."""
    waited = 0
    while True:
        path = locate(args.src, name)
        if path is not None and open_parquet(path) is not None:
            return path
        if args.hf_download:
            from huggingface_hub import hf_hub_download
            return hf_hub_download(args.hf_repo, f"data/{name}", repo_type="dataset", local_dir=args.src)
        if not args.wait:
            raise FileNotFoundError(f"{name} not found in {args.src} (or {args.src}/data). "
                                    f"Use --hf-download to fetch it, or --wait if a downloader is running.")
        if waited >= args.wait_timeout:
            raise TimeoutError(f"waited {waited}s for {name}; is the downloader still running?")
        if waited % 300 == 0:
            print(f"waiting for {name} ...", flush=True)
        time.sleep(15)
        waited += 15


def main(args):
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.src, exist_ok=True)
    final_lat = os.path.join(args.out, "latents.npy")
    part_lat = final_lat + ".partial"
    lab_path = os.path.join(args.out, "labels.npy")
    prog_path = os.path.join(args.out, "encode_progress.json")
    assert not os.path.exists(final_lat), f"{final_lat} already exists; nothing to do (delete it to re-encode)."

    latent_size = args.image_size // 8
    shape = (args.num_samples, 8, latent_size, latent_size)
    config = {"num_samples": args.num_samples, "num_shards": args.num_shards, "image_size": args.image_size,
              "vae": args.vae}

    # Resume state: shards are always taken in index order, so "next shard" + "rows written" is enough.
    if os.path.exists(prog_path) and os.path.exists(part_lat):
        prog = json.load(open(prog_path))
        assert prog["config"] == config, f"settings differ from the interrupted run: {prog['config']} vs {config}"
        lat = np.load(part_lat, mmap_mode="r+")
        labels = np.load(lab_path + ".partial.npy")
        print(f"Resuming: {prog['next_shard']}/{args.num_shards} shards done, {prog['written']:,} images.", flush=True)
    else:
        prog = {"config": config, "next_shard": 0, "written": 0}
        lat = np.lib.format.open_memmap(part_lat, mode="w+", dtype=np.float32, shape=shape)
        labels = np.full((args.num_samples,), -1, dtype=np.int64)
        print(f"Allocated {part_lat}: {shape} float32 = {np.prod(shape) * 4 / 1e9:.1f} GB", flush=True)
    assert lat.shape == shape

    # VAE (encoder only is used). ft-ema and ft-mse share the same encoder.
    from diffusers.models import AutoencoderKL
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = AutoencoderKL.from_pretrained(args.vae).to(device).eval()
    vae.requires_grad_(False)
    use_fp16 = args.fp16 and device == "cuda"
    print(f"VAE {args.vae} on {device}, {'fp16' if use_fp16 else 'fp32'}", flush=True)

    t_start = time.time()
    done_now = 0
    for si in range(prog["next_shard"], args.num_shards):
        name = shard_name(si, args.num_shards)
        path = get_shard(args, name)
        pf = open_parquet(path)
        rows = pf.metadata.num_rows
        w0 = prog["written"]
        assert w0 + rows <= args.num_samples, \
            f"{name} would bring the total to {w0 + rows:,}, more than --num-samples {args.num_samples:,}"

        w = w0
        for rg in range(pf.num_row_groups):
            tbl = pf.read_row_group(rg, columns=["image", "label"])
            blobs = [d["bytes"] for d in tbl.column("image").to_pylist()]
            lab = np.asarray(tbl.column("label").to_pylist(), dtype=np.int64)
            assert lab.min() >= 0 and lab.max() < args.num_classes, f"unexpected labels in {name}: {lab.min()}..{lab.max()}"
            loader = DataLoader(BytesImages(blobs, args.image_size), batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=device == "cuda")
            k = w
            with torch.no_grad():
                for x in loader:
                    x = x.to(device, non_blocking=True).float().div_(127.5).sub_(1.0)   # [-1, 1]
                    with torch.autocast(device, dtype=torch.float16, enabled=use_fp16):
                        moments = vae.encode(x).latent_dist.parameters                  # (B, 8, h, w): mean, logvar
                    moments = moments.float()
                    assert torch.isfinite(moments).all(), f"non-finite latents in {name}"
                    lat[k:k + x.shape[0]] = moments.cpu().numpy()
                    k += x.shape[0]
            assert k - w == len(blobs)
            labels[w:k] = lab
            w = k
            del blobs, tbl, loader

        # Commit the shard: data first, then the progress record that makes it count.
        lat.flush()
        np.save(lab_path + ".partial.npy", labels)
        prog["next_shard"], prog["written"] = si + 1, w
        tmp = prog_path + ".tmp"
        json.dump(prog, open(tmp, "w"))
        os.replace(tmp, prog_path)
        if args.delete_shards:
            os.remove(path)

        done_now += rows
        rate = done_now / (time.time() - t_start)
        eta_h = (args.num_samples - w) / max(rate, 1e-9) / 3600
        print(f"[{si + 1}/{args.num_shards}] {name}: {rows} images | total {w:,}/{args.num_samples:,} "
              f"| {rate:.0f} img/s | about {eta_h:.1f} h left", flush=True)

    # All shards done: the count must match exactly, otherwise the tail of the file is empty.
    assert prog["written"] == args.num_samples, \
        (f"encoded {prog['written']:,} images but --num-samples is {args.num_samples:,}. "
         f"Files are left as .partial; rerun with the right --num-samples/--num-shards.")
    assert (labels >= 0).all()
    del lat
    np.save(lab_path, labels)
    os.replace(part_lat, final_lat)
    os.remove(lab_path + ".partial.npy")
    os.remove(prog_path)
    counts = np.bincount(labels, minlength=args.num_classes)
    print(f"DONE: {final_lat} {shape}, {lab_path}. Images per class: min {counts.min()}, max {counts.max()}.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, required=True, help="Folder for latents.npy and labels.npy (= train.py --data-path).")
    parser.add_argument("--src", type=str, required=True, help="Folder holding (or receiving) the parquet shards.")
    parser.add_argument("--hf-download", action="store_true", help="Download each missing shard from Hugging Face.")
    parser.add_argument("--hf-repo", type=str, default="ILSVRC/imagenet-1k")
    parser.add_argument("--wait", action="store_true", help="Wait for missing shards to appear (another process downloads them).")
    parser.add_argument("--wait-timeout", type=int, default=3600, help="Seconds to wait for one shard before giving up.")
    parser.add_argument("--delete-shards", action="store_true", help="Delete each parquet shard once it is encoded.")
    parser.add_argument("--num-shards", type=int, default=294)
    parser.add_argument("--num-samples", type=int, default=IMAGENET_TRAIN_SIZE)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--vae", type=str, default="stabilityai/sd-vae-ft-ema", help="Hub id or local folder.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--fp16", action="store_true", help="Encode in fp16 (faster on V100; default fp32 is exact).")
    main(parser.parse_args())
