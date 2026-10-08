"""
Downloads the ImageNet-1k train parquet shards one by one, never keeping more than --max-on-disk
of them at a time. Run it on a node WITH internet (the login node) while encode_latents.py runs
with --wait --delete-shards on a GPU node: this script feeds shards, the encoder consumes them.

    python download_shards.py --src /media/ltnghia33/imagenet_parquet

Needs a Hugging Face token with access to ILSVRC/imagenet-1k (accept the terms on the dataset
page, then `huggingface-cli login`). Safe to interrupt and restart.
"""
import argparse
import glob
import os
import time


def fetch(repo, filename, local_dir):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, filename, repo_type="dataset", local_dir=local_dir)


def main(args):
    os.makedirs(args.src, exist_ok=True)
    state = os.path.join(args.src, "downloaded.txt")   # shard indices fully downloaded so far
    done = set(int(x) for x in open(state).read().split()) if os.path.exists(state) else set()
    for i in range(args.num_shards):
        if i in done:
            continue
        name = f"train-{i:05d}-of-{args.num_shards:05d}.parquet"
        while len(glob.glob(os.path.join(args.src, "data", "*.parquet"))) >= args.max_on_disk:
            time.sleep(10)   # the encoder deletes shards as it finishes them
        for attempt in range(1, 6):
            try:
                fetch(args.hf_repo, f"data/{name}", args.src)
                break
            except Exception as e:   # network hiccups: retry a few times, then give up loudly
                print(f"{name}: attempt {attempt} failed ({type(e).__name__}: {e})", flush=True)
                if attempt == 5:
                    raise
                time.sleep(30 * attempt)
        with open(state, "a") as f:
            f.write(f"{i}\n")
        print(f"[{i + 1}/{args.num_shards}] {name}", flush=True)
    print("All shards downloaded.", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=str, required=True, help="Same folder as encode_latents.py --src.")
    p.add_argument("--hf-repo", type=str, default="ILSVRC/imagenet-1k")
    p.add_argument("--num-shards", type=int, default=294)
    p.add_argument("--max-on-disk", type=int, default=8, help="About 0.5GB per shard.")
    main(p.parse_args())
