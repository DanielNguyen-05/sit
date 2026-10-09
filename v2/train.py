# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for SiT using PyTorch DDP.

Changes from the previous version of this file:
  * resume works: --exp-name gives the run a stable folder and training continues from
    <results-dir>/<exp-name>/checkpoints/latest.pt whenever that file exists
  * --amp picks the autocast dtype (auto = bf16 on Ampere and newer, fp16 + GradScaler on
    older GPUs such as V100)
  * checkpoints: one rolling full checkpoint (latest.pt) plus small EMA-only milestones
  * --no-compress --no-detailer trains the plain SiT baseline in this same codebase
  * --max-steps ends the run at an exact step count
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch._dynamo
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import math
import os
from models import SiT_models
from dataset import CustomDataset, MemmapLatentDataset
from transport import create_transport, Sampler
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from train_utils import parse_transport_args, parse_model_args, model_kwargs_from_args, MODEL_ARG_KEYS
import wandb_utils


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def pick_amp_dtype(choice, use_cuda):
    """auto: bf16 where the GPU has native support (compute capability >= 8.0, i.e. Ampere and
    newer), fp16 otherwise. V100 is 7.0: bf16 runs there but without tensor-core speedup."""
    if not use_cuda or choice == "none":
        return None
    if choice == "auto":
        choice = "bf16" if torch.cuda.get_device_capability()[0] >= 8 else "fp16"
    return {"bf16": torch.bfloat16, "fp16": torch.float16}[choice]


def save_checkpoint(path, *, ema, args, train_steps, epoch, model=None, opt=None, scaler=None):
    """Full checkpoint when model/opt are given (for resuming), EMA-only otherwise (for FID).
    Written to a temp file first so a job killed mid-write never leaves a corrupt latest.pt."""
    checkpoint = {"ema": ema.state_dict(), "args": args, "train_steps": train_steps, "epoch": epoch}
    if model is not None:
        checkpoint["model"] = model.state_dict()
        checkpoint["opt"] = opt.state_dict()
        checkpoint["scaler"] = scaler.state_dict()
    tmp = f"{path}.tmp"
    torch.save(checkpoint, tmp)
    os.replace(tmp, path)


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new SiT model.
    """
    use_cuda = torch.cuda.is_available()
    assert use_cuda or args.allow_cpu, "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl" if use_cuda else "gloo")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count() if use_cuda else "cpu"
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    local_batch_size = int(args.global_batch_size // dist.get_world_size())

    # Setup an experiment folder. With --exp-name the folder is stable across job submissions,
    # which is what makes automatic resume possible.
    model_string_name = args.model.replace("/", "-")  # e.g., SiT-XL/2 --> SiT-XL-2 (for naming folders)
    if args.exp_name is not None:
        experiment_name = args.exp_name
    elif rank == 0:
        experiment_index = len(glob(f"{args.results_dir}/*"))
        experiment_name = f"{experiment_index:03d}-{model_string_name}-" \
                        f"{args.path_type}-{args.prediction}-{args.loss_weight}-{int(time())}"
    else:
        experiment_name = None
    experiment_dir = f"{args.results_dir}/{experiment_name}"  # Create an experiment folder
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
    latest_path = f"{checkpoint_dir}/latest.pt"
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory: {experiment_dir}")
        if args.wandb:
            wandb_utils.initialize(args, os.environ["ENTITY"], experiment_name, os.environ["PROJECT"])
    else:
        logger = create_logger(None)

    # Which checkpoint (if any) to continue from:
    resume_path = None
    if args.resume == "auto":
        if args.exp_name is not None and os.path.isfile(latest_path):
            resume_path = latest_path
    elif args.resume not in (None, "none"):
        assert os.path.isfile(args.resume), f"--resume: no checkpoint at {args.resume}"
        resume_path = args.resume

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = SiT_models[args.model](
        aux_recon=args.aux_recon,   # A = PE-only (default off); B = --aux-recon (bottleneck-AE loss)
        learn_sigma=False,
        **model_kwargs_from_args(args),
    )
    # Note that parameter initialization is done within the SiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training

    train_steps = 0
    start_epoch = 0
    resume_state = None
    if resume_path is not None:
        resume_state = torch.load(resume_path, map_location="cpu", weights_only=False)
        assert "model" in resume_state and "opt" in resume_state, \
            f"{resume_path} is an EMA-only milestone; resume from checkpoints/latest.pt instead."
        saved = resume_state["args"]
        for k in MODEL_ARG_KEYS + ["aux_recon", "global_batch_size", "path_type", "prediction"]:
            if hasattr(saved, k) and getattr(saved, k) != getattr(args, k):
                raise ValueError(f"Resume mismatch on --{k.replace('_', '-')}: checkpoint has "
                                 f"{getattr(saved, k)!r}, this run has {getattr(args, k)!r}.")
        model.load_state_dict(resume_state["model"])
        ema.load_state_dict(resume_state["ema"])
        train_steps = resume_state["train_steps"]
        # Start on a fresh epoch permutation rather than replaying the interrupted one.
        start_epoch = resume_state["epoch"] + 1
        # ...and do not replay the noise/timestep stream of the first submission either.
        torch.manual_seed(seed + train_steps)
        logger.info(f"Resumed from {resume_path} at step {train_steps}.")

    raw_model = model.to(device)  # uncompiled module: EMA source and what gets checkpointed
    requires_grad(ema, False)
    if args.compile and use_cuda:
        # If the compiler toolchain is broken on a node (e.g. Triton cannot link libcuda),
        # run that graph eagerly instead of crashing; the warning shows up in the .err log.
        torch._dynamo.config.suppress_errors = True
        model = torch.compile(raw_model)
    model = DDP(model, device_ids=[device]) if use_cuda else DDP(model)
    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps
    )  # default: velocity;
    transport_sampler = Sampler(transport)
    logger.info(f"SiT Parameters: {sum(p.numel() for p in raw_model.parameters()):,} "
                f"(compress={args.compress}, detailer={args.detailer}, legacy_residual={args.legacy_residual})")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # Mixed precision. fp16 needs loss scaling; bf16 and fp32 do not (the scaler is then a no-op).
    amp_dtype = pick_amp_dtype(args.amp, use_cuda)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype is torch.float16)
    logger.info(f"Autocast dtype: {amp_dtype} (GradScaler {'on' if scaler.is_enabled() else 'off'})")

    if resume_state is not None:
        opt.load_state_dict(resume_state["opt"])
        if resume_state.get("scaler") and scaler.is_enabled():
            scaler.load_state_dict(resume_state["scaler"])
        del resume_state

    # Setup data:
    # Use the fast uncompressed memmap if it exists (see repack_latents.py); else
    # fall back to the per-file .npz tree.
    if os.path.exists(os.path.join(args.data_path, "latents.npy")):
        dataset = MemmapLatentDataset(args.data_path)
        logger.info("Using MemmapLatentDataset (uncompressed memmap).")
    else:
        dataset = CustomDataset(args.data_path)
        logger.info("Using CustomDataset (.npz tree) — consider repack_latents.py for speed.")
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=local_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
        drop_last=True,
        collate_fn=dataset.collate_fn,
        persistent_workers=args.num_workers > 0,   # don't respawn workers every epoch
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Prepare models for training:
    if train_steps == 0:
        update_ema(ema, raw_model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    running_aux = 0
    bad_steps = 0
    start_time = time()

    # Labels to condition the model with (feel free to change):
    n = 32
    ys = torch.randint(args.num_classes, size=(n,), device=device)
    use_cfg = args.cfg_scale > 1.0
    # Create sampling noise:
    zs = torch.randn(n, 4, latent_size, latent_size, device=device)
    # Setup classifier-free guidance:
    if use_cfg:
        zs = torch.cat([zs, zs], 0)
        y_null = torch.tensor([args.num_classes] * n, device=device)
        ys = torch.cat([ys, y_null], 0)
        sample_model_kwargs = dict(y=ys, cfg_scale=args.cfg_scale)
        model_fn = ema.forward_with_cfg
    else:
        sample_model_kwargs = dict(y=ys)
        model_fn = ema.forward
    vae = None  # only loaded if/when preview samples are actually generated

    def checkpoint(step, epoch):
        """Rolling full checkpoint every --ckpt-every steps; EMA-only milestone every --keep-every."""
        if rank == 0:
            save_checkpoint(latest_path, ema=ema, args=args, train_steps=step, epoch=epoch,
                            model=raw_model, opt=opt, scaler=scaler)
            logger.info(f"Saved checkpoint to {latest_path} (step {step})")
            if args.keep_every > 0 and step % args.keep_every == 0:
                milestone = f"{checkpoint_dir}/{step:07d}.pt"
                save_checkpoint(milestone, ema=ema, args=args, train_steps=step, epoch=epoch)
                logger.info(f"Saved EMA milestone to {milestone}")
        dist.barrier()

    done = args.max_steps is not None and train_steps >= args.max_steps
    logger.info(f"Training for {args.epochs} epochs" + (f" or {args.max_steps} steps" if args.max_steps else "") + "...")
    epoch = start_epoch
    for epoch in range(start_epoch, args.epochs):
        if done:
            break
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            opt.zero_grad(set_to_none=True)

            x = x.to(device)
            y = y.to(device)
            x = DiagonalGaussianDistribution(x).sample().mul_(0.18215)
            model_kwargs = dict(y=y)
            # Mixed-precision forward; master weights stay fp32 in AdamW.
            with torch.autocast("cuda" if use_cuda else "cpu", dtype=amp_dtype, enabled=amp_dtype is not None):
                loss_dict = transport.training_losses(model, x, model_kwargs)
            loss = loss_dict["loss"]
            aux_recon = loss_dict.get("aux_recon_loss")

            loss_value = loss.item()
            # backward always runs, so every DDP rank stays in step even when one rank's loss is bad
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            # total grad norm; with --grad-clip 0 the huge max_norm makes this a pure measurement
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=args.grad_clip if args.grad_clip > 0 else 1e9).item()

            if not math.isfinite(norm):
                # Non-finite gradients (fp16 overflow, or a NaN loss) must never reach the weights
                # or the EMA. The norm is computed on the all-reduced gradients, so every rank
                # takes this branch together. Give up if it keeps happening.
                if scaler.is_enabled():
                    scaler.step(opt)   # no-op on the weights; lets the scaler back off its scale
                    scaler.update()
                bad_steps += 1
                logger.info(f"(step={train_steps:07d}) non-finite gradients (loss={loss_value}), "
                            f"update skipped ({bad_steps} in a row)")
                if bad_steps >= 20:
                    raise RuntimeError("20 consecutive non-finite updates; stopping. Resume from latest.pt "
                                       "with --amp none (or --grad-clip 1.0) to check whether fp16 is the cause.")
                continue
            bad_steps = 0

            scaler.step(opt)
            scaler.update()
            update_ema(ema, raw_model)

            # Log loss values:
            running_loss += loss_value
            running_aux += aux_recon.item() if aux_recon is not None else 0.
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                if use_cuda:
                    torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                avg_aux = torch.tensor(running_aux / log_steps, device=device)
                dist.all_reduce(avg_aux, op=dist.ReduceOp.SUM)
                avg_aux = avg_aux.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Aux Recon: {avg_aux:.4f}, "
                            f"Grad Norm: {norm:.3f}, Train Steps/Sec: {steps_per_sec:.2f}")
                if args.wandb:
                    wandb_utils.log(
                        { "train loss": avg_loss, "aux recon loss": avg_aux, "train steps/sec": steps_per_sec, "norm": norm },
                        step=train_steps
                    )
                # Reset monitoring variables:
                running_loss = 0
                running_aux = 0
                log_steps = 0
                start_time = time()

            done = args.max_steps is not None and train_steps >= args.max_steps

            # Save SiT checkpoint:
            if train_steps % args.ckpt_every == 0 or done:
                checkpoint(train_steps, epoch)

            if args.sample_every > 0 and train_steps % args.sample_every == 0:
                logger.info("Generating EMA samples...")
                if vae is None:
                    from diffusers.models import AutoencoderKL
                    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
                    vae.requires_grad_(False)
                with torch.no_grad():
                    sample_fn = transport_sampler.sample_ode() # default to ode sampling
                    samples = sample_fn(zs, model_fn, **sample_model_kwargs)[-1]
                    dist.barrier()

                    if use_cfg: #remove null samples
                        samples, _ = samples.chunk(2, dim=0)

                    # [vram] decode in sub-batches: the 256px SD-VAE decoder is the memory
                    # hotspot, so a single decode of all samples is a large one-off spike.
                    samples = samples / 0.18215
                    _dec_chunk = 8
                    decoded = [vae.decode(samples[i:i + _dec_chunk]).sample.float()
                               for i in range(0, samples.shape[0], _dec_chunk)]
                    samples = torch.cat(decoded, dim=0)
                    del decoded
                    out_samples = torch.zeros((32 * dist.get_world_size(), 3, args.image_size, args.image_size), device=device)
                    dist.all_gather_into_tensor(out_samples, samples)

                if args.wandb:
                    wandb_utils.log_image(out_samples, train_steps)
                # [vram] release the sampling transients so the caching allocator doesn't hold
                # the peak reserved for the rest of training (the 8.5->15GB plateau).
                del samples, out_samples
                if use_cuda:
                    torch.cuda.empty_cache()
                logger.info("Generating EMA samples done.")

            if done:
                break

    if not done and train_steps % args.ckpt_every != 0:
        checkpoint(train_steps, epoch)   # ran out of epochs: keep the final weights too

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info(f"Done! Finished at step {train_steps}.")
    cleanup()


if __name__ == "__main__":
    # Default args here will train SiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--exp-name", type=str, default=None,
                        help="Stable run folder name under --results-dir. Required for automatic resume.")
    parser.add_argument("--resume", type=str, default="auto",
                        help="auto = continue from <exp>/checkpoints/latest.pt if it exists (needs --exp-name); "
                             "none = always start fresh; or a path to a full checkpoint.")
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Stop after this many optimizer steps (e.g. 400000), whichever of this and --epochs comes first.")
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=5_000,
                        help="Overwrite checkpoints/latest.pt (model+EMA+optimizer) this often.")
    parser.add_argument("--keep-every", type=int, default=50_000,
                        help="Also keep an EMA-only milestone this often (0 = never). Must be a multiple of --ckpt-every.")
    parser.add_argument("--sample-every", type=int, default=100_100, help="Preview samples (0 = never).")
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--amp", type=str, default="auto", choices=["auto", "bf16", "fp16", "none"],
                        help="Autocast dtype. auto = bf16 on Ampere and newer, fp16 on older GPUs (V100).")
    parser.add_argument("--grad-clip", type=float, default=0.0, help="Max grad norm (0 = measure only, no clipping).")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-cpu", action="store_true", help="Smoke tests only.")
    parser.add_argument("--aux-recon", action="store_true",
                        help="Enable bottleneck-AE reconstruction loss (run B). Off = run A, PE-only.")
    parser.add_argument("--wandb", action="store_true")

    parse_model_args(parser)
    parse_transport_args(parser)
    args = parser.parse_args()
    assert args.keep_every % args.ckpt_every == 0, "--keep-every must be a multiple of --ckpt-every"
    main(args)
