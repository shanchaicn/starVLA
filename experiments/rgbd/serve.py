"""Reuse StarVLA websocket server; preserve native signed LIBERO actions."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from train import build_model
from starVLA.rgbd.checkpoint import restore_checkpoint,experiment_identity
from starVLA.rgbd.preprocessing import validate_contract
from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer

class Policy:
    def __init__(self,model):self.model=model
    def predict_action(self,examples,seed=None,**kwargs):
        if seed is not None:torch.manual_seed(int(seed));torch.cuda.manual_seed_all(int(seed))
        # Clients already apply the shared state/depth preprocessing; no second normalization.
        torch.cuda.synchronize();start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        result=self.model.predict_action(examples=examples)
        torch.cuda.synchronize()
        actions=np.clip(result["normalized_actions"],-1,1)
        actions[...,6]=np.where(actions[...,6]>=0,1.,-1.)
        return {"actions":actions,"inference_latency_s":time.perf_counter()-start,
                "gpu_peak_bytes":torch.cuda.max_memory_allocated()}

def main():
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",required=True);p.add_argument("--port",type=int,default=10093)
    a=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError("Enable GPU before serving the full model")
    ck=torch.load(a.checkpoint,map_location="cpu",weights_only=False);cfg=OmegaConf.create(ck["config"])
    contract=json.loads(Path(cfg.rgbd_experiment.contract).read_text())
    if cfg.framework.name!="QwenPI_v3":validate_contract(contract,lingbot=cfg.framework.rgbd.encoder=="lingbot")
    if not contract.get("simulation_interface_verified"):raise ValueError("Simulation/data interface unverified")
    model=build_model(cfg).cuda().eval();restore_checkpoint(a.checkpoint,model,cfg)
    meta={"action_chunk_size":int(model.action_horizon),"native_libero_actions":True,
          "identity":experiment_identity(cfg),"framework":cfg.framework.name,
          "encoder":cfg.framework.get("rgbd",{}).get("encoder","rgb"),
          "checkpoint":str(Path(a.checkpoint).resolve())}
    WebsocketPolicyServer(Policy(model),host="127.0.0.1",port=a.port,metadata=meta).serve_forever()
if __name__=="__main__":main()
