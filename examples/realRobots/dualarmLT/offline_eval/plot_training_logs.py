#!/usr/bin/env python3
"""Render actual logged metrics, with no invented or interpolated training points."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS={'suck':'Suck / QwenPI v3 V2','place':'Place / QwenPI v3 V2',
        'snap':'Snap / ACT V2','insert':'Insert / ACT V2','push_pull':'Push-Pull / QwenPI v3 V2'}


def read_rows(path):
    with path.open() as stream:
        return [{k:float(v) for k,v in r.items() if k!='loss_key' and v!=''} for r in csv.DictReader(stream)]


def smooth(values,window=100):
    values=np.asarray(values,dtype=float)
    sums=np.r_[0,np.cumsum(values)]
    end=np.arange(1,len(values)+1);start=np.maximum(0,end-window)
    return (sums[end]-sums[start])/(end-start)


def plot(summary_path,output):
    document=json.loads(summary_path.read_text());data=summary_path.parent
    output.mkdir(parents=True,exist_ok=True)
    tasks=['suck','place','snap','insert','push_pull']
    runs={task:[r for r in document['runs'] if r['task']==task] for task in tasks}
    series={task:[(r,read_rows(data/r['history_file'])) for r in group if r['history_file']] for task,group in runs.items()}
    fig,axes=plt.subplots(3,2,figsize=(12,11),layout='constrained')
    for ax,task in zip(axes.flat,tasks):
        for index,(run,rows) in enumerate(series[task]):
            x=np.array([r['step'] for r in rows]);y=np.array([r['loss'] for r in rows])
            ax.plot(x/1000,y,color='tab:blue',alpha=.18,linewidth=.65)
            ax.plot(x/1000,smooth(y),color='tab:blue',linewidth=1.5,label='100 logged points trailing mean' if index==0 else None)
        ax.set(title=LABELS[task],xlabel='Optimizer steps (thousands)',ylabel='Logged training loss (log scale)',yscale='log')
        ax.grid(alpha=.2)
        if task=='push_pull':ax.axvline(20,color='tab:red',linestyle='--',linewidth=1,label='Weights-only continuation')
        ax.legend(frameon=False,fontsize=8)
    ax=axes.flat[-1]
    for key,label,color in [('learning_rate/qwen_vl_interface','VLM','tab:blue'),('learning_rate/action_model','Action head','tab:orange')]:
        for phase,(_,rows) in enumerate(series['push_pull']):
            present=[r for r in rows if key in r]
            ax.plot([r['step']/1000 for r in present],[r[key] for r in present],color=color,label=label if phase==0 else None)
    ax.axvline(20,color='tab:red',linestyle='--',linewidth=1)
    ax.set(title='Push-Pull learning-rate restart',xlabel='Optimizer steps (thousands)',ylabel='Learning rate',yscale='log')
    ax.grid(alpha=.2);ax.legend(frameon=False)
    fig.suptitle('LT five-action training histories | sampled logs every 10 updates\nLoss objectives differ across ACT / QwenPI; do not rank tasks by loss',fontsize=13)
    fig.savefig(output/'training_curves.png',dpi=150);plt.close(fig)
    for task in tasks:
        fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True,layout='constrained')
        for phase,(run,rows) in enumerate(series[task]):
            x=np.array([r['step'] for r in rows])/1000
            y=np.array([r['loss'] for r in rows])
            axes[0].plot(x,y,color='tab:blue',alpha=.2,linewidth=.6,label='All logged points' if phase==0 else None)
            axes[0].plot(x,smooth(y),color='tab:blue',linewidth=1.5,label='Trailing mean: 100 points' if phase==0 else None)
            for key,label,color in [('learning_rate/qwen_vl_interface','VLM','tab:blue'),('learning_rate/action_model','Action head','tab:orange')]:
                present=[r for r in rows if key in r]
                if present:axes[1].plot([r['step']/1000 for r in present],[r[key] for r in present],color=color,label=label if phase==0 else None)
            axes[2].plot(x,[r['timing/model'] for r in rows],color='tab:green',alpha=.2,linewidth=.6)
            axes[2].plot(x,smooth([r['timing/model'] for r in rows]),color='tab:green',label='Model time: trailing mean' if phase==0 else None)
        axes[0].set(ylabel='Training loss',yscale='log');axes[1].set(ylabel='Learning rate',yscale='log')
        axes[2].set(xlabel='Optimizer steps (thousands)',ylabel='Logged model time (s/update)')
        for ax in axes:
            ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=8)
            if task=='push_pull':ax.axvline(20,color='tab:red',linestyle='--',linewidth=1)
        fig.suptitle(LABELS[task]+' | actual training logs',fontsize=13)
        fig.savefig(output/f'{task}_training.png',dpi=150);plt.close(fig)
    print(json.dumps(dict(images=6,output=str(output))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();plot(args.summary,args.output_dir)
