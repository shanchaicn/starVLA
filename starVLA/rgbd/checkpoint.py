"""Compact checkpoints: immutable base weights are referenced, not duplicated."""
import hashlib, json, os, random
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def experiment_identity(config):
    exp=config.rgbd_experiment
    return {k:sha256(getattr(exp,k)) for k in ("selected_tasks","normalization","contract")}

def learned_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()
            if not k.startswith(("qwen_vl_interface.", "depth_encoder.model."))}

def save_checkpoint(path,model,optimizer,step,config,sampler_generator):
    path=Path(path)
    payload={"learned":learned_state(model),"optimizer":optimizer.state_dict(),"step":step,
             "config":OmegaConf.to_container(config,resolve=True),"identity":experiment_identity(config),
             "torch_rng":torch.get_rng_state(),"numpy_rng":np.random.get_state(),"python_rng":random.getstate(),
             "cuda_rng":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
             "sampler_rng":sampler_generator.get_state()}
    temp=path.with_suffix(".tmp")
    torch.save(payload,temp);os.replace(temp,path)

def restore_checkpoint(path,model,config,optimizer=None,generator=None):
    # Only load checkpoints created locally by this experiment.
    ck=torch.load(path,map_location="cpu",weights_only=False)
    if ck["identity"] != experiment_identity(config): raise ValueError("Dataset/preprocessing identity mismatch")
    expected=set(learned_state(model))
    if set(ck["learned"])!=expected: raise ValueError("Checkpoint module mismatch")
    model.load_state_dict(ck["learned"],strict=False)
    if optimizer is not None:
        optimizer.load_state_dict(ck["optimizer"])
        torch.set_rng_state(ck["torch_rng"]);np.random.set_state(ck["numpy_rng"]);random.setstate(ck["python_rng"])
        if ck["cuda_rng"] and torch.cuda.is_available(): torch.cuda.set_rng_state_all(ck["cuda_rng"])
        if generator is not None: generator.set_state(ck["sampler_rng"])
    return ck["step"]
