#!/usr/bin/env python3
"""Paired, reproducible offline action prediction comparison; no robot imports/RPC."""
from pathlib import Path
import argparse
import contextlib
import copy
import gc
import hashlib
import io
import json
import os
import random
import time
import types
import sys

ROOT = Path(__file__).resolve().parents[4]
RUN = BASE = CORPUS = DATASET = CACHE_DIR = BASE_MODEL = None
DEVICE = "cuda:0"
VIDEO_SOURCE = "cache"


def emit(event, **kw):
    print(json.dumps(dict(event=event, **kw), ensure_ascii=False), flush=True)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(per_task):
    import numpy as np
    import pandas as pd
    from omegaconf import OmegaConf
    from examples.realRobots.dualarmLTPushPull.train_files.data_registry.data_config import (
        DualArmLTPushPullDataset, DualArmLTPushPullV2H50Config)
    cfg = OmegaConf.load(RUN/'config.full.yaml')
    dims=cfg.framework.action_model
    assert cfg.framework.name=='QwenPI_v3' and (dims.state_dim,dims.action_dim,dims.action_horizon)==(24,9,50), 'Expected Push/Pull QwenPI v3 V2'
    data_cfg = copy.deepcopy(cfg.datasets.vla_data)
    data_cfg.load_all_data_for_training = False
    data_cfg.state_action_dtype = 'float32'
    dc = DualArmLTPushPullV2H50Config()
    # This subclass only changes indexing/video access for evaluation. No raw metadata writes.
    class ReadOnlyDataset(DualArmLTPushPullDataset):
        def _get_all_steps(self):
            return []
    with contextlib.redirect_stdout(io.StringIO()):
        ds = ReadOnlyDataset(dataset_path=DATASET,
            modality_configs=dc.modality_config(), transforms=dc.transform(),
            embodiment_tag=dc.embodiment_tag, video_backend='pyav', data_cfg=data_cfg)
    expected_stats = json.loads((RUN/'dataset_statistics.json').read_text())

    metadata = pd.concat([pd.read_parquet(p) for p in sorted((ds.dataset_path/'meta/episodes').rglob('*.parquet'))], ignore_index=True)
    stats_count=expected_stats['new_embodiment']
    assert len(metadata)==stats_count['num_trajectories'] and int(metadata['length'].sum())==stats_count['num_transitions'], 'This command reproduces the original training dataset, not a new held-out dataset'
    lengths = dict(zip(map(int, ds.trajectory_ids), map(int, ds.trajectory_lengths)))
    offsets = dict(zip(map(int, ds.trajectory_ids), np.cumsum(ds.trajectory_lengths)-ds.trajectory_lengths))
    caches = {}
    total=int(np.sum(ds.trajectory_lengths))
    if VIDEO_SOURCE == 'cache':
        for camera in ('left_wrist','left_side'):
            p=(CACHE_DIR or Path(cfg.datasets.vla_data.video_cache_dir))/f'{camera}_224x224_uint8.npy'
            if not p.is_file():
                raise FileNotFoundError(f'{p}; set --cache-dir or use --video-source video')
            caches[camera]=np.load(p,mmap_mode='r')
            assert caches[camera].shape==(total,224,224,3) and caches[camera].dtype==np.uint8
    decoded={}
    def video(self, episode, key, index):
        assert len(self.delta_indices[key])==1 and self.delta_indices[key][0]==0
        camera=key.removeprefix('video.')
        if VIDEO_SOURCE == 'cache':
            return caches[camera][int(offsets[int(episode)]+index)][None]
        from starVLA.dataloader.gr00t_lerobot.video import get_all_frames
        # Keep at most the two current episode videos in RAM. Match training cv2 resize.
        if decoded.get('episode') != episode:
            decoded.clear();decoded['episode']=episode
        if camera not in decoded:
            path=self.get_video_path(episode,camera)
            frames=get_all_frames(str(path),video_backend='pyav',resize_size=(224,224))
            assert len(frames)==lengths[int(episode)], 'Expected the original per-episode video layout'
            decoded[camera]=frames
        return decoded[camera][index:index+1]
    ds.get_video=types.MethodType(video,ds)
    groups={'pull':[], 'push':[]}
    for _,row in metadata.iterrows():
        text=' '.join(map(str,row['tasks']))
        task='pull' if 'pull out' in text else 'push' if 'push in' in text else None
        assert task, text
        if int(row['length'])>=50:
            groups[task].append(int(row['episode_index']))
    rng=np.random.default_rng(20261125)
    selected=[]
    for task, episodes in groups.items():
        selected += [(task,int(e)) for e in sorted(rng.choice(episodes,min(per_task,len(episodes)),replace=False))]
    samples=[]; manifest=[]
    for task,ep in selected:
        data=ds.get_trajectory_data(ep)
        ds.curr_traj_id=ep; ds.curr_traj_data=data
        assert len(data)==lengths[ep]
        do=np.stack(data['action.digital_output'].to_numpy())[:,[8,11]]
        end=lengths[ep]-50
        anchors={int(round(end*f)):f'phase_{f:.1f}' for f in (.1,.5,.9)}
        changes=np.where(np.any(do[1:]!=do[:-1],axis=1))[0]+1
        if len(changes):
            event=int(changes[len(changes)//2])
            anchors.setdefault(int(np.clip(event-10,0,end)), 'io_transition')
        for anchor,kind in sorted(anchors.items()):
            raw=ds.get_step_data(ep,anchor)
            raw_gt=np.concatenate([np.asarray(raw[k]) for k in dc.action_keys],axis=1).astype('float32')
            observed=np.asarray(raw['state.arm_1_pose'])[0].astype('float32').copy()
            sample=ds._pack_sample(ds.transforms(copy.deepcopy(raw)))
            assert sample['action'].shape==(50,9) and sample['state'].shape==(1,24)
            assert np.isfinite(sample['state']).all() and np.isfinite(sample['action']).all()
            assert sample['lang']==data['task_index'].map(ds._tasks['task']).iloc[anchor]
            stat=expected_stats['new_embodiment']['action']
            lower=np.asarray(stat['q01'],dtype=np.float32)[:7]
            upper=np.asarray(stat['q99'],dtype=np.float32)[:7]
            check=np.clip(2*(raw_gt[:,:7]-lower)/(upper-lower)-1,-2.2,2.2)
            np.testing.assert_allclose(sample['action'][:,:7],check,atol=2e-5)
            np.testing.assert_array_equal(sample['action'][:,7:],raw_gt[:,7:])
            record=dict(id=len(samples),episode=ep,anchor=anchor,kind=kind,task=task,
                        length=lengths[ep],instruction=sample['lang'])
            manifest.append(record)
            samples.append(dict(image=np.stack([np.asarray(im) for im in sample['image']]),
                state=sample['state'], action=sample['action'], raw_action=raw_gt, observed_pose=observed))
        emit('prepared_episode',episode=ep,task=task,samples=len(samples))
    arrays={k:np.stack([s[k] for s in samples]) for k in samples[0]}
    np.savez_compressed(BASE/'samples.npz',**arrays)
    protocol=dict(dataset=str(DATASET.resolve()),data_view=str(ds.dataset_path),
        split='in_training_data_not_held_out',num_episodes=len(selected),num_samples=len(samples),
        available_episodes_by_task={k:len(v) for k,v in groups.items()},cameras=['left_wrist','left_side'],
        state_dim=24,action_dim=9,horizon=50,denoising_steps=4,do_threshold=.5,
        episode_selection_seed=20261125,noise_seed_formula='20261125 + sample_id*101 + repeat*1000003',
        sample_selection='uniform episode sampling within each task; 3 time phases plus one IO transition anchor if present',
        no_padded_target_frames=True,augmentation=False,video_source=VIDEO_SOURCE,
        stats_sha256=sha(RUN/'dataset_statistics.json'),config_sha256=sha(RUN/'config.full.yaml'),
        samples_sha256=sha(BASE/'samples.npz'),samples=manifest)
    (BASE/'dataset_statistics.json').write_bytes((RUN/'dataset_statistics.json').read_bytes())
    (BASE/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2))
    emit('samples_ready',episodes=len(selected),samples=len(samples))


def initial_model(checkpoint):
    import torch
    from omegaconf import OmegaConf
    from transformers import AutoConfig,Qwen3VLForConditionalGeneration
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding,Qwen3VLVisionRotaryEmbedding
    from starVLA.model.framework.VLM4A.QwenPI_v3 import Qwen_PI_v3
    import starVLA.model.modules.action_model.LayerwiseFM_ActionHeader as head_module
    from torch.distributions import Beta as OriginalBeta
    cfg=OmegaConf.load(RUN/'config.full.yaml')
    cfg.framework.qwenvl.attn_implementation='sdpa'
    if BASE_MODEL is not None:cfg.framework.qwenvl.base_vlm=str(BASE_MODEL)
    dims=cfg.framework.action_model
    assert cfg.framework.name=='QwenPI_v3' and (dims.state_dim,dims.action_dim,dims.action_horizon)==(24,9,50), 'This evaluator supports Push/Pull QwenPI v3 V2 only'
    original=Qwen3VLForConditionalGeneration.from_pretrained
    beta=head_module.Beta
    def meta_model(*args,**kwargs):
        c=AutoConfig.from_pretrained(cfg.framework.qwenvl.base_vlm,local_files_only=True)
        c._attn_implementation='sdpa'
        return Qwen3VLForConditionalGeneration(c)
    Qwen3VLForConditionalGeneration.from_pretrained=meta_model
    head_module.Beta=lambda a,b:OriginalBeta(torch.tensor(float(a)),torch.tensor(float(b)),validate_args=False)
    try:
        with torch.device('meta'):
            model=Qwen_PI_v3(cfg)
    finally:
        Qwen3VLForConditionalGeneration.from_pretrained=original
        head_module.Beta=beta
    state=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
    model.load_state_dict(state,strict=True,assign=True)
    del state
    vl=model.qwen_vl_interface.model
    vr=vl.model.visual.rotary_pos_emb
    vr.register_buffer('inv_freq',Qwen3VLVisionRotaryEmbedding(int(vr.inv_freq.numel())*2).inv_freq,persistent=False)
    tr=vl.model.language_model.rotary_emb
    tr.register_buffer('inv_freq',Qwen3VLTextRotaryEmbedding(vl.config.text_config,device='cpu').inv_freq,persistent=False)
    tr.original_inv_freq=tr.inv_freq
    assert not any(p.is_meta for p in model.parameters()) and not any(b.is_meta for b in model.buffers())
    model=model.eval().to(DEVICE)
    torch.cuda.synchronize()
    emit('model_ready',device=torch.cuda.get_device_name(),allocated_gib=torch.cuda.memory_allocated()/1024**3)
    return model


def infer(args):
    import numpy as np
    import torch
    from PIL import Image
    assert torch.cuda.is_available(), 'CUDA is required for inference; prepare/analyze are CPU operations'
    torch.cuda.set_device(DEVICE)
    torch.set_num_threads(4)
    # Fixed sampler noise across checkpoints; no reduced-precision/quantized replacement.
    torch.backends.cuda.matmul.allow_tf32=False
    protocol=json.loads((CORPUS/'protocol.json').read_text())
    with np.load(CORPUS/'samples.npz') as archive:
        arrays={k:archive[k] for k in archive.files}
    stats=json.loads((RUN/'dataset_statistics.json').read_text())['new_embodiment']['action']
    q01=np.asarray(stats['q01'],dtype=np.float32)
    q99=np.asarray(stats['q99'],dtype=np.float32)
    assert sha(RUN/'dataset_statistics.json')==protocol['stats_sha256'], 'Normalization statistics differ from the prepared corpus'
    limit=min(len(protocol['samples']),args.limit or len(protocol['samples']))
    model=None
    for step in args.steps:
        checkpoint=RUN/f'checkpoints/steps_{step}_pytorch_model.pt'
        assert checkpoint.is_file()
        path=BASE/f'predictions_{step}.npz'
        meta=BASE/f'predictions_{step}.json'
        if path.exists():
            if not args.skip_existing or not meta.is_file():
                raise FileExistsError(f'{path}; use a new --output-dir, or --skip-existing for matching completed outputs')
            previous=json.loads(meta.read_text())
            assert previous['config_sha256']==sha(RUN/'config.full.yaml') and previous['base_vlm_override']==(str(BASE_MODEL.resolve()) if BASE_MODEL else None), 'Configuration changed'
            assert previous['corpus_sha256']==protocol['samples_sha256'] and previous['repeats']==args.repeats and previous['samples']==limit
            assert previous['checkpoint_path']==str(checkpoint.resolve()) and previous['checkpoint_bytes']==checkpoint.stat().st_size and previous['checkpoint_mtime_ns']==checkpoint.stat().st_mtime_ns, 'Checkpoint changed'
            with np.load(path) as saved:assert saved['physical'].shape==(args.repeats,limit,50,9) and np.isfinite(saved['physical']).all()
            emit('reuse_complete',step=step);continue
        emit('checkpoint_start',step=step,checkpoint=str(checkpoint))
        if model is None:
            model=initial_model(checkpoint)
        else:
            state=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
            model.load_state_dict(state,strict=True)
            del state
            gc.collect()
            torch.cuda.synchronize()
        path=BASE/f'predictions_{step}.npz'
        norm=np.full((args.repeats,limit,50,9),np.nan,dtype=np.float32)
        latencies=np.zeros((args.repeats,limit))
        for repeat in range(args.repeats):
            for i in range(limit):
                seed=20261125+i*101+repeat*1000003
                random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                example=dict(image=[Image.fromarray(im) for im in arrays['image'][i]],
                    state=arrays['state'][i],lang=protocol['samples'][i]['instruction'])
                torch.cuda.synchronize();started=time.perf_counter()
                with torch.inference_mode():
                    pred=model.predict_action(examples=[example])['normalized_actions']
                torch.cuda.synchronize();latencies[repeat,i]=time.perf_counter()-started
                norm[repeat,i]=np.asarray(pred,dtype=np.float32)[0]
                assert np.isfinite(norm[repeat,i]).all(),(step,i,'non-finite prediction')
                if i%8==0 or i==limit-1:
                    emit('inference_progress',step=step,repeat=repeat,done=i+1,total=limit,
                         latency_s=round(float(latencies[repeat,i]),3),
                         peak_gib=round(torch.cuda.max_memory_allocated()/1024**3,3))
                    np.savez_compressed(BASE/f'partial_{step}.npz',normalized=norm,latency_s=latencies)
        physical=norm.copy()
        physical[...,:7]=(norm[...,:7]+1)*.5*(q99[:7]-q01[:7])+q01[:7]
        temporary=BASE/f'predictions_{step}.tmp.npz'
        np.savez_compressed(temporary,normalized=norm,physical=physical,latency_s=latencies)
        temporary.replace(path)
        meta.write_text(json.dumps(dict(step=step,checkpoint_path=str(checkpoint.resolve()),checkpoint_bytes=checkpoint.stat().st_size,checkpoint_mtime_ns=checkpoint.stat().st_mtime_ns,config_sha256=sha(RUN/'config.full.yaml'),base_vlm_override=str(BASE_MODEL.resolve()) if BASE_MODEL else None,corpus_sha256=protocol['samples_sha256'],repeats=args.repeats,samples=limit),indent=2))
        (BASE/f'partial_{step}.npz').unlink(missing_ok=True)
        emit('checkpoint_complete',step=step,mean_latency_s=float(latencies.mean()),samples=limit,repeats=args.repeats)
    emit('inference_complete')


def main():
    global ROOT,RUN,BASE,CORPUS,DATASET,CACHE_DIR,BASE_MODEL,DEVICE,VIDEO_SOURCE
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','infer'])
    parser.add_argument('--run-dir',type=Path,required=True,help='Completed run directory containing config.full.yaml and dataset_statistics.json')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--repo-root',type=Path,default=ROOT)
    parser.add_argument('--dataset-path',type=Path,default=Path('/root/data/datasets/push_pull_lt_v2'))
    parser.add_argument('--corpus-dir',type=Path,help='Output directory from prepare; required by infer')
    parser.add_argument('--cache-dir',type=Path,help='Existing 224x224 training RGB .npy cache')
    parser.add_argument('--video-source',choices=['cache','video'],default='cache')
    parser.add_argument('--base-vlm',type=Path,help='Override local Qwen3-VL-4B-Instruct-Action config/tokenizer directory')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--per-task',type=int,default=16)
    parser.add_argument('--steps',type=int,nargs='+',default=[20000,25000,30000,35000,40000])
    parser.add_argument('--repeats',type=int,default=2)
    parser.add_argument('--limit',type=int,help='Inference smoke test only; incomplete corpora cannot be used for full comparisons')
    parser.add_argument('--skip-existing',action='store_true')
    parser.add_argument('--dry-run',action='store_true',help='Validate paths and print plan; no CUDA/model initialization')
    args=parser.parse_args()
    for name in ('per_task','repeats'):
        if getattr(args,name)<1:parser.error(f'--{name.replace("_","-")} must be positive')
    if args.limit is not None and args.limit<1:parser.error('--limit must be positive')
    if len(set(args.steps))!=len(args.steps) or any(s<1 for s in args.steps):parser.error('--steps must contain unique positive steps')
    if not args.device.startswith('cuda:'):parser.error('infer supports cuda:N; prepare is CPU-only')
    ROOT=args.repo_root.resolve();RUN=args.run_dir.resolve();BASE=args.output_dir.resolve()
    CORPUS=(args.corpus_dir or BASE).resolve();DATASET=args.dataset_path.resolve()
    CACHE_DIR=args.cache_dir;BASE_MODEL=args.base_vlm;DEVICE=args.device;VIDEO_SOURCE=args.video_source
    sys.path.insert(0,str(ROOT))
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'
    os.environ['NO_ALBUMENTATIONS_UPDATE']='1'
    os.environ.setdefault('WANDB_MODE','disabled')
    for name in ('config.full.yaml','dataset_statistics.json'):
        if not (RUN/name).is_file():parser.error(f'Missing {RUN/name}')
    if args.mode=='infer':
        if args.corpus_dir is None:parser.error('infer requires --corpus-dir')
        for name in ('protocol.json','samples.npz'):
            if not (CORPUS/name).is_file():parser.error(f'Missing {CORPUS/name}; run prepare first')
        for step in args.steps:
            if not (RUN/f'checkpoints/steps_{step}_pytorch_model.pt').is_file():parser.error(f'Missing checkpoint step {step}')
    else:
        if not (DATASET/'meta/info.json').is_file():parser.error(f'Missing dataset {DATASET}')
        if (BASE/'samples.npz').exists() or (BASE/'protocol.json').exists():parser.error('Prepared corpus already exists; choose a new --output-dir')
    if args.dry_run:
        emit('dry_run',mode=args.mode,run_dir=str(RUN),output_dir=str(BASE),corpus_dir=str(CORPUS),steps=args.steps,repeats=args.repeats,video_source=VIDEO_SOURCE)
        return
    BASE.mkdir(parents=True,exist_ok=True)
    if args.mode=='prepare':prepare(args.per_task)
    else:infer(args)


if __name__=='__main__':main()
