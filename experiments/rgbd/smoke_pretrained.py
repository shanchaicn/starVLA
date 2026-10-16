"""Real-weight integration using a synthetic aligned plane; no robot training."""
import argparse,gc,json,time,tempfile
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from omegaconf import OmegaConf
from train import build_model
from starVLA.rgbd.checkpoint import learned_state

parser=argparse.ArgumentParser();parser.add_argument("--device",choices=["cpu","cuda"],default="cuda")
args=parser.parse_args()
if args.device=="cuda" and not torch.cuda.is_available():raise RuntimeError("GPU not available")
limit=Path("/sys/fs/cgroup/memory.max").read_text().strip()
if args.device=="cpu" and limit!="max" and int(limit)<16*1024**3:
    raise RuntimeError("Full 4B CPU smoke requires >=16 GiB cgroup RAM; current limit is too small")
torch.set_num_threads(1)
report=[]
for name in ["e0_rgb","e1_cnn","e2_lingbot"]:
    cfg=OmegaConf.load(f'experiments/rgbd/configs/{name}.yaml')
    # Same real 4B VLM and layer count; short action chunk minimizes CPU test work.
    cfg.framework.action_model.action_horizon=4 if args.device=="cpu" else 16
    cfg.framework.action_model.num_inference_timesteps=2 if args.device=="cpu" else 10
    if name=="e2_lingbot" and args.device=="cpu":cfg.framework.rgbd.num_tokens=64
    print('START',name,flush=True);start=time.perf_counter()
    model=build_model(cfg).to(args.device)
    h=w=56 if args.device=="cpu" else 256
    cfg.datasets.vla_data.obs_image_size=[h,w]
    yy,xx=np.mgrid[:h,:w]
    rgb=np.stack([xx/w*255,yy/h*255,np.full((h,w),128)],axis=-1).astype(np.uint8)
    e={"image":[Image.fromarray(rgb),Image.fromarray(rgb)],"lang":"move the bowl onto the plate",
       "state":np.zeros((1,8),np.float32),"action":np.zeros((int(cfg.framework.action_model.action_horizon),7),np.float32),
       "depth":np.full((2,h,w),1.2,np.float32),"depth_mask":np.ones((2,h,w),bool),
       "intrinsics":np.repeat(np.array([[60,0,28],[0,60,28],[0,0,1]],np.float32)[None],2,0)}
    model.train();torch.manual_seed(7)
    loss=model(examples=[e])["action_loss"];assert torch.isfinite(loss)
    loss.backward()
    grads={k:float(p.grad.abs().sum()) for k,p in model.named_parameters()
           if p.grad is not None and any(t in k for t in ('depth_fusion.gates','depth_fusion.projector.1.weight','depth_encoder.cnn.0.weight','action_decoder.layer2.weight'))}
    assert grads and all(v>0 for v in grads.values()),grads
    assert all(p.grad is None for p in model.qwen_vl_interface.parameters())
    if name=="e2_lingbot":assert all(p.grad is None for p in model.depth_encoder.parameters())
    optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=1e-4)
    weight=model.action_model.action_decoder.layer2.weight
    before=weight.detach().clone();optimizer.step();assert not torch.equal(before,weight)
    model.eval();torch.manual_seed(11)
    pred=model.predict_action(examples=[e])["normalized_actions"]
    assert pred.shape==(1,int(cfg.framework.action_model.action_horizon),7) and np.isfinite(pred).all()
    with tempfile.TemporaryDirectory(dir='/root/data/tmp',prefix='rgbd-cpu-restore-') as tmp:
        path=Path(tmp)/'learned.pt';torch.save(learned_state(model),path)
        with torch.no_grad():weight.add_(.1)
        model.load_state_dict(torch.load(path,weights_only=True),strict=False)
        torch.manual_seed(11)
        np.testing.assert_array_equal(pred,model.predict_action(examples=[e])["normalized_actions"])
    record={"experiment":name,"real_base_weights":True,"input":"synthetic aligned plane; NOT LIBERO-3D validation",
        "device":args.device,"action_horizon":int(cfg.framework.action_model.action_horizon),"sampler_steps":int(cfg.framework.action_model.num_inference_timesteps),"lingbot_input_tokens":int(cfg.framework.rgbd.num_tokens) if name=="e2_lingbot" else None,
        "forward_loss":float(loss.detach()),"gradient_l1":grads,"parameter_update":True,"predict_shape":list(pred.shape),
        "checkpoint_prediction_exact":True,"seconds":time.perf_counter()-start}
    report.append(record)
    Path(f'experiments/rgbd/reports/pretrained_{args.device}_smoke.json').write_text(json.dumps(report,indent=2))
    print('PASS',json.dumps(record),flush=True)
    del model,optimizer,weight,before,loss;gc.collect()
