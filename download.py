# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Functions for downloading pre-trained SiT models
"""
from torchvision.datasets.utils import download_url
import torch
import os
import paths


pretrained_models = {'SiT-XL-2-256x256.pt'}


def find_model(model_name):
    """
    Finds a pre-trained SiT model, downloading it if necessary. Alternatively, loads a model from a local path.
    """
    if model_name in pretrained_models:  
        return download_model(model_name)
    else:  
        assert os.path.isfile(model_name), f'Could not find SiT checkpoint at {model_name}'
        checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage, weights_only=False)
        if "ema" in checkpoint:  # supports checkpoints from train.py
            checkpoint = checkpoint["ema"]
        return checkpoint


def find_model_and_args(model_name):
    """Like find_model, but also returns the training args stored in a train.py checkpoint
    (None for the released weights), so samplers can rebuild the exact architecture."""
    if model_name in pretrained_models:
        return download_model(model_name), None
    assert os.path.isfile(model_name), f'Could not find SiT checkpoint at {model_name}'
    checkpoint = torch.load(model_name, map_location="cpu", weights_only=False)
    if "ema" in checkpoint:
        return checkpoint["ema"], checkpoint.get("args")
    return checkpoint, None


def download_model(model_name):
    """
    Downloads a pre-trained SiT model from the web.
    """
    assert model_name in pretrained_models
    local_path = os.path.join(paths.PRETRAINED_DIR, model_name)
    legacy = os.path.join(paths.PRETRAINED_DIR, model_name.replace("256x256", "256"))  # name used by older downloads
    if not os.path.isfile(local_path) and os.path.isfile(legacy):
        local_path = legacy
    if not os.path.isfile(local_path):
        os.makedirs(paths.PRETRAINED_DIR, exist_ok=True)
        web_path = f'https://www.dl.dropboxusercontent.com/scl/fi/as9oeomcbub47de5g4be0/SiT-XL-2-256.pt?rlkey=uxzxmpicu46coq3msb17b9ofa&dl=0'
        download_url(web_path, paths.PRETRAINED_DIR, filename=model_name)
    model = torch.load(local_path, map_location=lambda storage, loc: storage, weights_only=False)
    return model
