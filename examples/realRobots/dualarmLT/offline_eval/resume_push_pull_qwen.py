#!/usr/bin/env python3
"""Create a new Push/Pull weights-only continuation; default is dry-run."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--run-id',required=True)
    p.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[4])
    p.add_argument('--outputs-root',type=Path,default=Path('/root/data/outputs/starVLA'))
    p.add_argument('--log-dir',type=Path,default=Path('/root/data/logs/starvla/push_pull_lt_v2'))
    p.add_argument('--total-steps',type=int,default=40000)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--num-processes',type=int,default=2)
    p.add_argument('--main-process-port',type=int,default=29615)
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--run',action='store_true')
    mode.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.run_id):p.error('run-id must contain letters, digits, _ or -')
    if args.batch_size<1 or args.num_processes<1:p.error('batch-size and num-processes must be positive')
    if not 1024<=args.main_process_port<=65535:p.error('port must be 1024..65535')
    checkpoint=args.checkpoint.resolve()
    match=re.fullmatch(r'steps_(\d+)_pytorch_model\.pt',checkpoint.name)
    if not match or not checkpoint.is_file():p.error('Use a completed steps_N_pytorch_model.pt checkpoint')
    completed=int(match[1])
    if args.total_steps<=completed:p.error('total-steps must exceed the source checkpoint step')
    import yaml
    config_path=checkpoint.parent.parent/'config.full.yaml'
    cfg=yaml.safe_load(config_path.read_text())
    dims=cfg['framework']['action_model']
    assert cfg['framework']['name']=='QwenPI_v3' and (dims['state_dim'],dims['action_dim'],dims['action_horizon'])==(24,9,50)
    assert cfg['datasets']['vla_data']['data_mix']=='dualarm_lt_push_pull_v2_h50_2cam'
    run_root=args.outputs_root.resolve()/args.run_id
    out=run_root/args.run_id
    if run_root.exists():p.error(f'Output already exists: {run_root}; choose a new run-id')
    for key in ('output_dir','config_yaml'):cfg.pop(key,None)
    cfg.update(run_id=args.run_id,run_root_dir=str(run_root),wandb_project=args.run_id)
    cfg['datasets']['vla_data']['per_device_batch_size']=args.batch_size
    cfg['trainer'].update(pretrained_checkpoint=str(checkpoint),is_resume=True,reset_steps=False,
                          random_init_action_model=False,max_train_steps=args.total_steps)
    config_target=out/'resume_config.yaml'
    repo=args.repo_root.resolve()
    cmd=[str(repo/'.venv/bin/accelerate'),'launch','--config_file',str(repo/'starVLA/config/deepseeds/deepspeed_zero2.yaml'),
         '--num_processes',str(args.num_processes),'--main_process_port',str(args.main_process_port),
         str(repo/'starVLA/training/train_starvla.py'),'--config_yaml',str(config_target)]
    record=dict(source_checkpoint=str(checkpoint),completed_steps=completed,target_total_steps=args.total_steps,
                additional_steps=args.total_steps-completed,batch_per_gpu=args.batch_size,num_processes=args.num_processes,
                effective_batch=args.batch_size*args.num_processes*cfg['trainer']['gradient_accumulation_steps'],
                optimizer_state_restored=False,random_init_action_model=False,output=str(out),command=cmd)
    print(json.dumps(record,ensure_ascii=False,indent=2),flush=True)
    if not args.run:return
    import socket
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count()<args.num_processes:
        raise RuntimeError('Not enough visible GPUs; CUDA_VISIBLE_DEVICES count must cover num-processes')
    with socket.socket() as sock:sock.bind(('127.0.0.1',args.main_process_port))
    interval=int(cfg['trainer']['save_interval'])
    saves=args.total_steps//interval-completed//interval
    required=(saves+1)*checkpoint.stat().st_size+4*1024**3
    if shutil.disk_usage(args.outputs_root).free<required:
        raise RuntimeError(f'Need at least {required/1024**3:.1f} GiB free for new checkpoints and final model')
    (out/'checkpoints').mkdir(parents=True,exist_ok=False)
    # Same-mount hard link, or a symlink across mounts: do not copy a 10.9GB seed.
    seed=out/'checkpoints'/checkpoint.name
    try:os.link(checkpoint,seed)
    except OSError:seed.symlink_to(checkpoint)
    config_target.write_text(yaml.safe_dump(cfg,sort_keys=False))
    args.log_dir.mkdir(parents=True,exist_ok=True)
    log=args.log_dir/f'qwenpi_{args.run_id}.log'
    env=dict(os.environ,WANDB_MODE='disabled',PYTHONUNBUFFERED='1',HF_HOME='/root/data/huggingface',
             TORCH_HOME='/root/data/torch-cache',TMPDIR='/root/data/tmp',XDG_CACHE_HOME='/root/data/.cache')
    env['PYTHONPATH']=str(repo)+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    with log.open('x') as stream:
        child=subprocess.Popen(cmd,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    record.update(pid=child.pid,log=str(log))
    log.with_suffix('.json').write_text(json.dumps(record,indent=2))
    log.with_suffix('.pid').write_text(str(child.pid)+'\n')
    print(f'LAUNCH_REQUESTED pid={child.pid}\nLOG={log}\nSTOP: kill -- -{child.pid}',flush=True)


if __name__=='__main__':main()
