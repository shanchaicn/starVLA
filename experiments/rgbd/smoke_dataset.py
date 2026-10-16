"""Load one frame/chunk from each of 20 distinct real episodes. Units are NOT certified."""
import json
from pathlib import Path
import numpy as np
from starVLA.rgbd.dataset import Libero3D
root=Path(__file__).resolve().parent
ds=Libero3D('/root/data/datasets/LIBERO-3D-selected',root/'selected_tasks.json',root/'normalization.json',split='smoke',
           depth_encoding='meters',cache_episodes=1)
# Explicit diagnostic assumption lets us inspect dtype/mask plumbing; never clears contract.
first={}
for idx,(episode,t) in enumerate(ds.index):
    if episode not in first:first[episode]=idx
rows=[]
for ep,idx in first.items():
    e=ds[idx]
    assert e['state'].shape==(1,8) and e['action'].shape==(16,7)
    assert e['depth'].shape==(2,256,256) and e['depth'].dtype==np.float32
    assert e['depth_mask'].dtype==bool
    assert np.isfinite(e['state']).all() and np.isfinite(e['depth']).all()
    rows.append({'episode_id':ep,'task_id':e['task_id'],'instruction':e['lang'],'depth_range':[float(e['depth'].min()),float(e['depth'].max())]})
    print('PASS',ep,e['task_id'],flush=True)
assert len(rows)==20
report={'status':'pass','real_episodes':20,'depth_encoding':'meters assumed for interface diagnostic ONLY; not a verified unit claim','samples':rows}
(root/'reports/dataset_smoke.json').write_text(json.dumps(report,indent=2))
