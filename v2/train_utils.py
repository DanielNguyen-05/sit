def none_or_str(value):
    if value == 'None':
        return None
    return value

def parse_transport_args(parser):
    group = parser.add_argument_group("Transport arguments")
    group.add_argument("--path-type", type=str, default="Linear", choices=["Linear", "GVP", "VP"])
    group.add_argument("--prediction", type=str, default="velocity", choices=["velocity", "score", "noise"])
    group.add_argument("--loss-weight", type=none_or_str, default=None, choices=[None, "velocity", "likelihood"])
    group.add_argument("--sample-eps", type=float)
    group.add_argument("--train-eps", type=float)

def parse_ode_args(parser):
    group = parser.add_argument_group("ODE arguments")
    group.add_argument("--sampling-method", type=str, default="dopri5", help="blackbox ODE solver methods; for full list check https://github.com/rtqichen/torchdiffeq")
    group.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance")
    group.add_argument("--rtol", type=float, default=1e-5, help="Relative tolerance")
    group.add_argument("--reverse", action="store_true")
    group.add_argument("--likelihood", action="store_true")

def parse_sde_args(parser):
    group = parser.add_argument_group("SDE arguments")
    group.add_argument("--sampling-method", type=str, default="Euler", choices=["Euler", "Heun"])
    group.add_argument("--diffusion-form", type=str, default="sigma", \
                        choices=["constant", "SBDM", "sigma", "linear", "decreasing", "increasing-decreasing"],\
                        help="form of diffusion coefficient in the SDE")
    group.add_argument("--diffusion-norm", type=float, default=1.0)
    group.add_argument("--last-step", type=none_or_str, default="Mean", choices=[None, "Mean", "Tweedie", "Euler"],\
                        help="form of last step taken in the SDE")
    group.add_argument("--last-step-size", type=float, default=0.04, \
                        help="size of the last step taken")

# ---------------------------------------------------------------------------------------
# Model arguments shared by train.py / sample.py / sample_ddp.py, so a checkpoint is always
# rebuilt with the architecture it was trained with.
# ---------------------------------------------------------------------------------------
MODEL_ARG_KEYS = ["model", "image_size", "num_classes", "query_len", "compressor_version",
                  "keep_ratio", "compress", "detailer", "legacy_residual", "zero_init_final"]

def parse_model_args(parser):
    import argparse
    group = parser.add_argument_group("Architecture arguments")
    group.add_argument("--query-len", type=int, default=32)
    group.add_argument("--compressor-version", type=int, default=1, choices=[1, 2, 3],
                       help="1 = fixed pool+xattn (default); 2 = soft content-adaptive; 3 = hard top-k routing.")
    group.add_argument("--keep-ratio", type=float, default=0.5, help="V3 only.")
    group.add_argument("--compress", action=argparse.BooleanOptionalAction, default=True,
                       help="--no-compress = plain SiT trunk (the in-codebase baseline).")
    group.add_argument("--detailer", action=argparse.BooleanOptionalAction, default=True,
                       help="--no-detailer = drop the pooled-input skip before the final layer.")
    group.add_argument("--legacy-residual", action="store_true",
                       help="Reproduce the pre-fix V1 compressor/decompressor forward (old checkpoints only).")
    group.add_argument("--zero-init-final", action="store_true",
                       help="Zero-init the output layer like upstream SiT.")

def model_kwargs_from_args(args):
    """kwargs for SiT_models[args.model](...). getattr defaults keep old arg namespaces working."""
    return dict(
        input_size=args.image_size // 8,
        num_classes=args.num_classes,
        query_seqlen=getattr(args, "query_len", 32),
        compressor_version=getattr(args, "compressor_version", 1),
        keep_ratio=getattr(args, "keep_ratio", 0.5),
        compress=getattr(args, "compress", True),
        detailer=getattr(args, "detailer", True),
        # checkpoints written before the residual fix have no such field -> they are legacy
        legacy_residual=getattr(args, "legacy_residual", True),
        zero_init_final=getattr(args, "zero_init_final", False),
    )

def apply_checkpoint_args(args, ckpt_args, log=print):
    """Overwrite the architecture fields of `args` with the ones stored in a checkpoint."""
    if ckpt_args is None:
        return args
    for k in MODEL_ARG_KEYS:
        default = True if k == "legacy_residual" else None
        v = getattr(ckpt_args, k, default)
        if v is None:
            continue
        if getattr(args, k, v) != v:
            log(f"[ckpt] using {k}={v} from the checkpoint (command line had {getattr(args, k)})")
        setattr(args, k, v)
    return args
