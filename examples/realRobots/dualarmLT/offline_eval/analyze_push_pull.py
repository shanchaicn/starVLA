#!/usr/bin/env python3
from pathlib import Path
import csv
import argparse
import re
import json
import numpy as np

BASE=CORPUS=PREDICTIONS=None


def quat_degrees(p,g):
    pn=np.linalg.norm(p,axis=-1);gn=np.linalg.norm(g,axis=-1)
    dot=np.sum(p*g,axis=-1)/np.maximum(pn*gn,1e-12)
    angle=2*np.arccos(np.clip(np.abs(dot),0,1))*180/np.pi
    return np.where(pn>1e-8,angle,180.)


def do_metrics(pred,gt):
    result={}
    for k,name in enumerate(('DO14','DO17')):
        p=pred[...,k]>.5;g=gt[...,k]>.5
        tp=int(np.sum(p&g));tn=int(np.sum(~p&~g));fp=int(np.sum(p&~g));fn=int(np.sum(~p&g))
        recall=tp/(tp+fn) if tp+fn else None
        specificity=tn/(tn+fp) if tn+fp else None
        result[name]=dict(accuracy=float(np.mean(p==g)),balanced_accuracy=(recall+specificity)/2 if recall is not None and specificity is not None else None,
            f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,recall=recall,specificity=specificity,
            tp=tp,tn=tn,fp=fp,fn=fn,target_positive_rate=float(np.mean(g)),pred_positive_rate=float(np.mean(p)))
    return result


def main():
    # Analytic checks for quaternion double cover, 90 degrees and IO threshold.
    q=np.array([[0,0,0,1],[0,0,0,-1],[0,0,2**-.5,2**-.5]])
    np.testing.assert_allclose(quat_degrees(q,np.array([[0,0,0,1]]*3)),[0,0,90],atol=1e-4)
    assert do_metrics(np.array([[.5,1.]]),np.array([[0.,1.]]))['DO14']['accuracy']==1.
    proto=json.loads((CORPUS/'protocol.json').read_text())
    samples=proto['samples'];arr=np.load(CORPUS/'samples.npz')
    ep=np.array([s['episode'] for s in samples]);task=np.array([s['task'] for s in samples])
    kind=np.array([s['kind'] for s in samples])
    gt=arr['raw_action'];target_norm=arr['action']
    weights={int(m[1]):p for p in PREDICTIONS.glob('predictions_*.npz')
             if (m:=re.fullmatch(r'predictions_(\d+)\.npz',p.name))}
    if not weights:raise ValueError('No completed predictions found; run infer without --limit for all samples')
    results={};per_episode={};all_errors={}
    # A no-motion observed-pose + training-majority-DO baseline.
    baseline=np.repeat(arr['observed_pose'][:,None,:],50,axis=1)
    baseline=np.concatenate([baseline,np.broadcast_to(np.array([0.,1.]),(len(samples),50,2))],axis=-1)
    variants=[('baseline',baseline[None],None,None)]
    for step,p in sorted(weights.items()):
        pred=np.load(p)
        if 'physical' not in pred or pred['physical'].shape[1]!=len(samples):
            raise ValueError(f'{p}: incomplete corpus; --limit is a smoke test, not a full evaluation')
        assert np.isfinite(pred['physical']).all()
        variants.append((str(step),pred['physical'],pred['normalized'],pred['latency_s']))
    for name,pred,norm,latency in variants:
        assert pred.shape[-3:]==gt.shape
        pos=np.linalg.norm(pred[...,:3]-gt[None,...,:3],axis=-1)*1000
        rot=quat_degrees(pred[...,3:7],gt[None,...,3:7])
        io=((pred[...,7:]>.5)!=(gt[None,...,7:]>.5)).mean(axis=-1)*100
        metrics={};epmeans={}
        for horizon in (1,10,50):
            masks={'all':np.ones(len(samples),dtype=bool),'pull':task=='pull','push':task=='push',
                   'io_transition':kind=='io_transition'}
            metrics[str(horizon)]={}
            for group,mask in masks.items():
                if not mask.any():continue
                # Equal episode weight. Repeat seeds and anchors are first averaged within episode.
                selected_eps=np.unique(ep[mask])
                row={}
                for metric,values in [('position_mm',pos),('rotation_deg',rot),('do_error_pct',io)]:
                    point_values=values[...,:horizon].mean(axis=(0,2))
                    vals=np.array([point_values[(ep==e)&mask].mean() for e in selected_eps])
                    row[metric]=float(vals.mean())
                    if group=='all':
                        epmeans[f'{metric}_h{horizon}']={str(e):float(v) for e,v in zip(selected_eps,vals)}
                row['position_p95_mm']=float(np.percentile(pos[:,mask,:horizon],95))
                row['rotation_p95_deg']=float(np.percentile(rot[:,mask,:horizon],95))
                row['digital_outputs']=do_metrics(pred[:,mask,:horizon,7:],np.broadcast_to(gt[None,mask,:horizon,7:],pred[:,mask,:horizon,7:].shape))
                row['samples']=int(mask.sum());row['episodes']=len(selected_eps)
                metrics[str(horizon)][group]=row
        metrics['repeats']=pred.shape[0]
        metrics['per_noise_repeat']=[]
        for repeat in range(pred.shape[0]):
            row={}
            for metric,values in [('position_mm',pos),('rotation_deg',rot),('do_error_pct',io)]:
                per_sample=values[repeat].mean(axis=-1)
                row[metric]=float(np.mean([per_sample[ep==e].mean() for e in np.unique(ep)]))
            row['digital_outputs']=do_metrics(pred[repeat,...,7:],gt[...,7:])
            metrics['per_noise_repeat'].append(row)
        metrics['near_zero_quaternion_count']=int(np.sum(np.linalg.norm(pred[...,3:7],axis=-1)<1e-8))
        if norm is not None:
            metrics['normalized_mse']=float(np.mean((norm-target_norm[None])**2))
            metrics['latency_mean_s']=float(latency.mean())
        results[name]=metrics;per_episode[name]=epmeans
        all_errors[name]=dict(position_mm=pos.mean(axis=(0,1)).tolist(),rotation_deg=rot.mean(axis=(0,1)).tolist(),
            do14_error_pct=(((pred[...,7]>.5)!=(gt[None,...,7]>.5)).mean(axis=(0,1))*100).tolist(),
            do17_error_pct=(((pred[...,8]>.5)!=(gt[None,...,8]>.5)).mean(axis=(0,1))*100).tolist())
    comparisons={}
    enough_clusters=min(len(np.unique(ep[task==t])) for t in ('pull','push'))>=2
    if '20000' in per_episode and enough_clusters:
        rng=np.random.default_rng(777)
        all_eps=np.unique(ep)
        groups=[np.unique(ep[task==t]) for t in ('pull','push')]
        for name in results:
            if name in ('20000','baseline'):continue
            comparisons[name]={}
            for key in per_episode[name]:
                diff=np.array([per_episode[name][key][str(e)]-per_episode['20000'][key][str(e)] for e in all_eps])
                # Paired, task-stratified, episode-cluster bootstrap, 5000 draws.
                by_task=[np.array([per_episode[name][key][str(e)]-per_episode['20000'][key][str(e)] for e in group]) for group in groups]
                bootstrap=np.mean(np.stack([vals[rng.integers(0,len(vals),(5000,len(vals)))].mean(axis=1) for vals in by_task]),axis=0)
                comparisons[name][key]=dict(delta=float(diff.mean()),ci95=np.percentile(bootstrap,[2.5,97.5]).tolist(),negative_is_better=True)
    latest_vs_35k={}
    if '35000' in per_episode and '40000' in per_episode and enough_clusters:
        rng=np.random.default_rng(987)
        groups=[np.unique(ep[task==t]) for t in ('pull','push')]
        for key in per_episode['40000']:
            values=[np.array([per_episode['40000'][key][str(e)]-per_episode['35000'][key][str(e)] for e in group]) for group in groups]
            boot=np.mean(np.stack([v[rng.integers(0,len(v),(5000,len(v)))].mean(axis=1) for v in values]),axis=0)
            latest_vs_35k[key]=dict(delta=float(np.mean(np.concatenate(values))),ci95=np.percentile(boot,[2.5,97.5]).tolist(),negative_is_better=True)
    report=dict(protocol={k:v for k,v in proto.items() if k!='samples'},metrics=results,
                paired_delta_vs_20k=comparisons,weighting='pose/error means equally weight episodes; digital-output confusion metrics pool target steps',
                paired_40k_minus_35k=latest_vs_35k,
                limitations=['These samples were used in training: this is in-sample action prediction, not held-out validation.',
                             'Teacher-forced recorded observations; no closed-loop robot rollouts or task success measurements.',
                             'Shared noise repeats are recorded per checkpoint; unseen-data and training-run uncertainty remains unmeasured. Paired CIs require at least two episodes per task.',
                             '20k to 40k is a weights-only continuation with reset optimizer and recomputed LR schedule; improvement cannot be attributed solely to step count.'])
    (BASE/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    (BASE/'per_episode.json').write_text(json.dumps(per_episode,indent=2))
    (BASE/'horizon_errors.json').write_text(json.dumps(all_errors))
    with (BASE/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['checkpoint','position_h10_mm','rotation_h10_deg','position_h50_mm','rotation_h50_deg','DO14_accuracy','DO14_balanced_accuracy','DO17_accuracy','DO17_balanced_accuracy','normalized_mse'])
        writer.writeheader()
        for name,m in results.items():
            h10=m['10']['all'];h50=m['50']['all']
            writer.writerow(dict(checkpoint=name,position_h10_mm=h10['position_mm'],rotation_h10_deg=h10['rotation_deg'],
                position_h50_mm=h50['position_mm'],rotation_h50_deg=h50['rotation_deg'],
                DO14_accuracy=h50['digital_outputs']['DO14']['accuracy'],DO14_balanced_accuracy=h50['digital_outputs']['DO14']['balanced_accuracy'],
                DO17_accuracy=h50['digital_outputs']['DO17']['accuracy'],DO17_balanced_accuracy=h50['digital_outputs']['DO17']['balanced_accuracy'],normalized_mse=m.get('normalized_mse')))
    print(json.dumps(dict(checkpoints=list(results),samples=len(samples),episodes=len(np.unique(ep)),report=str(BASE/'metrics.json'))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Analyze completed Push/Pull predictions on CPU; no model or RPC.')
    parser.add_argument('--corpus-dir',type=Path,required=True)
    parser.add_argument('--prediction-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    CORPUS=args.corpus_dir.resolve();PREDICTIONS=args.prediction_dir.resolve();BASE=args.output_dir.resolve()
    BASE.mkdir(parents=True,exist_ok=True)
    main()
