#!/usr/bin/env python3
"""Extract numeric training histories from explicit run sources; never starts training."""
import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import re

ANSI = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
STEP = re.compile(r'\bStep\s+(\d+),\s+Loss:')


def parse_history(text):
    text = ANSI.sub('', text)
    rows, failures = [], []
    matches = list(STEP.finditer(text))
    for i, match in enumerate(matches):
        block = text[match.end():matches[i+1].start() if i+1<len(matches) else len(text)]
        body = re.search(r'\{.*?\}', block, re.S)
        try:
            if body is None:
                raise ValueError('metrics dictionary missing')
            metrics = ast.literal_eval(body[0])
            numeric = {k:float(v) for k,v in metrics.items() if isinstance(v,(int,float)) and not isinstance(v,bool)}
            loss_key = next(k for k in ('action_dit_loss','action_loss','loss','l1_loss') if k in numeric)
            if not all(math.isfinite(v) for v in numeric.values()):
                raise ValueError('non-finite metric')
            rows.append(dict(numeric,step=int(match[1]),loss=numeric[loss_key],loss_key=loss_key))
        except (ValueError,SyntaxError,StopIteration,TypeError) as exc:
            failures.append(dict(step=int(match[1]),reason=str(exc)))
    duplicates = len(rows)-len({r['step'] for r in rows})
    if duplicates:
        raise ValueError(f'{duplicates} duplicate logged steps: split resumed logs into explicit phases')
    return rows,failures


def collect(manifest,output):
    output.mkdir(parents=True,exist_ok=True)
    sources=json.loads(manifest.read_text())
    result=[]
    for run in sources['runs']:
        folder=Path(run['output'])
        config=folder/'config.full.yaml'
        summary=folder/'summary.jsonl'
        weight=folder/'final_model/pytorch_model.pt'
        assert config.is_file() and summary.is_file() and weight.is_file(),f'Incomplete run: {folder}'
        history=[];failures=[];text=''
        log=Path(run['log']) if run.get('log') else None
        if log is not None and log.is_file():
            text=log.read_text(errors='replace')
            history,failures=parse_history(text)
            if failures:
                raise ValueError(f'Unparsed/non-finite metrics in {log}: {failures[:4]}')
            assert history and history[-1]['step']==run['max_train_steps'],f'Final logged step mismatch: {log}'
        record=dict(task=run['task'],run_id=run['run_id'],role=run['role'],output=str(folder),
            log=str(log) if log else None,log_available=bool(history),logged_points=len(history),
            summary=[json.loads(line) for line in summary.read_text().splitlines() if line.strip()],
            target_steps=run['max_train_steps'],weight_bytes=weight.stat().st_size,
            num_processes=run.get('num_processes'),batch_per_gpu=run['batch_per_gpu'],
            effective_batch=run.get('effective_batch'),config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
            summary_sha256=hashlib.sha256(summary.read_bytes()).hexdigest(),
            source_log_sha256=hashlib.sha256(log.read_bytes()).hexdigest() if log and log.is_file() else None)
        if history:
            assert history[-1]['step']==max(r['steps'] for r in record['summary'])
            process=re.search(r'Num processes:\s*(\d+)',text)
            record['logged_num_processes']=int(process[1]) if process else None
            if process and record['num_processes'] is not None:
                assert record['logged_num_processes']==record['num_processes']
            elapsed=re.findall(rf"{run['max_train_steps']}/{run['max_train_steps']}\s+\[([0-9:]+)<",text)
            record.update(first_logged_step=history[0]['step'],last_logged_step=history[-1]['step'],
                first_logged_loss=history[0]['loss'],last_logged_loss=history[-1]['loss'],loss_key=history[0]['loss_key'],
                last_1000_steps_logged_mean=sum(r['loss'] for r in history if r['step']>history[-1]['step']-1000)/sum(r['step']>history[-1]['step']-1000 for r in history),
                loop_elapsed=elapsed[-1] if elapsed else None,training_complete_message='Training complete.' in text)
        else:
            record['missing_note']='逐步训练日志未保留；仅能核对配置、保存事件及最终权重，不补画 loss。'
        name=run['task']+('_resume40k' if run['role']=='continuation' else '')
        if history:
            fields=['step','loss','loss_key']+sorted(set().union(*(r.keys() for r in history))-{'step','loss','loss_key'})
            with (output/f'{name}_history.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(history)
        record['history_file']=f'{name}_history.csv' if history else None
        result.append(record)
        print(json.dumps(dict(task=run['task'],role=run['role'],logged_points=len(history),last_logged_step=record.get('last_logged_step')),ensure_ascii=False),flush=True)
    (output/'training_summary.json').write_text(json.dumps(dict(checked_date=sources['checked_date'],runs=result),ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    collect(args.manifest,args.output_dir)
