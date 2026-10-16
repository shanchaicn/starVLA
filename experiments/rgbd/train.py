"""Single-GPU frozen-VLM training. Explicit data contracts guard scientific runs."""
import argparse, json, math, random, time
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from starVLA.rgbd.dataset import Libero3D, collate_examples
from starVLA.rgbd.preprocessing import validate_contract
from starVLA.rgbd.checkpoint import save_checkpoint,restore_checkpoint

def build_model(cfg):
    if cfg.framework.name=="QwenPI_v3":
        from starVLA.rgbd.backbone import FrozenRGBBaseline
        model=FrozenRGBBaseline(cfg)
    else:
        from starVLA.model.framework.VLM4A.QwenPI_v3_RGBD import QwenPI_v3_RGBD
        model=QwenPI_v3_RGBD(cfg)
    model.qwen_vl_interface.requires_grad_(False).eval()
    return model

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",required=True);p.add_argument("--output",required=True)
    p.add_argument("--split",choices=["smoke","train","all"],default="smoke")
    p.add_argument("--epochs",type=float,default=1);p.add_argument("--max-steps",type=int)
    p.add_argument("--batch-size",type=int,default=1);p.add_argument("--accumulation",type=int,default=4)
    p.add_argument("--lr",type=float,default=1e-4);p.add_argument("--seed",type=int,default=7)
    p.add_argument("--resume");p.add_argument("--save-every",type=int,default=100)
    p.add_argument("--validate-every",type=int,default=100);p.add_argument("--patience",type=int,default=5)
    args=p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("GPU required for training; use test_rgbd.py for CPU checks")
    cfg=OmegaConf.load(args.config);exp=cfg.rgbd_experiment
    contract=json.loads(Path(exp.contract).read_text())
    if cfg.framework.name!="QwenPI_v3":
        validate_contract(contract,lingbot=cfg.framework.rgbd.encoder=="lingbot")
    elif not contract.get("temporal_alignment_verified") or not contract.get("action_semantics_verified"):
        raise ValueError("Confirm observation/action temporal and control semantics before RGB training")
    out=Path(args.output).resolve()
    if not out.is_relative_to('/root/data/outputs'):raise ValueError("Output must be in /root/data/outputs")
    out.mkdir(parents=True,exist_ok=bool(args.resume))
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    gen=torch.Generator().manual_seed(args.seed)
    depth=contract["depth_encoding"] if cfg.framework.name!="QwenPI_v3" else None
    ds=Libero3D(exp.data_root,exp.selected_tasks,exp.normalization,split=args.split,depth_encoding=depth)
    steps=args.max_steps or math.ceil(args.epochs*len(ds)/(args.batch_size*args.accumulation))
    if min(steps,args.batch_size,args.accumulation)<=0: raise ValueError("Positive budget required")
    model=build_model(cfg).cuda()
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.lr,weight_decay=.01)
    step=restore_checkpoint(args.resume,model,cfg,optimizer,gen) if args.resume else 0
    # Each next() draws a fresh batch; saving generator state is sufficient for sampler resumption.
    loader=DataLoader(ds,batch_size=args.batch_size,sampler=ds.sampler(args.batch_size,gen),
                      num_workers=0,collate_fn=collate_examples)
    val=None
    if args.split=="train":
        val=Libero3D(exp.data_root,exp.selected_tasks,exp.normalization,split="validation",depth_encoding=depth)
    OmegaConf.save(cfg,out/'config.yaml')
    (out/'run.json').write_text(json.dumps(vars(args)|{"target_steps":steps,"windows":len(ds),
        "trajectories":len(ds.episodes),"sampling":"equal task probability; uniform valid start within each task",
        "validation":"held-out episodes45..49" if val else "none; full/smoke losses are training losses"},indent=2))
    best=float("inf");stale=0
    while step<steps:
        model.train();model.qwen_vl_interface.eval();optimizer.zero_grad(set_to_none=True)
        total=0.;begin=time.perf_counter()
        for _ in range(args.accumulation):
            batch=next(iter(loader))
            loss=model(examples=batch)["action_loss"]/args.accumulation
            if not torch.isfinite(loss):raise FloatingPointError("Non-finite action loss")
            loss.backward();total+=loss.item()
        grad=float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0))
        optimizer.step();step+=1
        record={"step":step,"train_fm_loss":total,"grad_norm":grad,"seconds":time.perf_counter()-begin,
                "gpu_peak_bytes":torch.cuda.max_memory_allocated()}
        if val is not None and step%args.validate_every==0:
            model.eval()
            # Same held-out frames and RNG each validation; no claims of success rate.
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]),torch.no_grad():
                torch.manual_seed(1234)
                ids=np.random.default_rng(1234).integers(0,len(val),size=16)
                fm=[];mse=[]
                for i in ids:
                    e=val[int(i)];fm.append(float(model(examples=[e])["action_loss"]))
                    pred=model.predict_action(examples=[e])["normalized_actions"][0]
                    mse.append(float(np.mean((pred-e["action"])**2)))
                record.update(validation_fm_loss=float(np.mean(fm)),validation_action_mse=float(np.mean(mse)))
            score=record["validation_action_mse"]
            if score<best:best=score;stale=0
            else:stale+=1
        with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
        if step%args.save_every==0 or step==steps or stale>=args.patience:
            save_checkpoint(out/'latest.pt',model,optimizer,step,cfg,gen)
        if stale>=args.patience:break
if __name__=="__main__":main()
