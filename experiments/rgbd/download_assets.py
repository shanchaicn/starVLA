"""Pinned, selective downloads. Run on the mounted server data volume."""
import concurrent.futures, hashlib, json, os, pathlib, shutil, time
from huggingface_hub import hf_hub_download
ROOT = pathlib.Path(__file__).resolve().parent
assert os.path.ismount('/root/data')
selection = json.loads((ROOT/'selected_tasks.json').read_text())
jobs = []
for key, destination in [('dataset','/root/data/datasets/LIBERO-3D-selected'),('qwen','/root/data/models/Qwen3-VL-4B-Instruct'),('lingbot','/root/data/models/lingbot-depth-v0.5')]:
    inv=json.loads((ROOT/'reports'/f'{key}_inventory.json').read_text())
    for f in inv['files']:
        if key=='dataset' and not any(f['path'].startswith(t['relative_dir']+'/') for t in selection['tasks']):
            continue
        jobs.append((key, inv, f, destination))
remaining=sum(f['size'] or 0 for _,_,f,d in jobs if not (pathlib.Path(d)/f['path']).exists())
free=shutil.disk_usage('/root/data').free
if free < remaining + 10*1024**3:
    raise RuntimeError(f'Insufficient capacity: free={free}, remaining={remaining}, reserve=10GiB')
print(f'download files={len(jobs)} remaining_bytes={remaining} free_bytes={free}',flush=True)
def download(job):
    key,inv,f,d=job
    for attempt in range(4):
        try:
            p=hf_hub_download(inv['repo'],f['path'],repo_type='dataset' if key=='dataset' else 'model',revision=inv['revision'],local_dir=d)
            path=pathlib.Path(p)
            assert path.stat().st_size == f['size'], (p,'size mismatch')
            digest=hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda:stream.read(8*1024**2),b''): digest.update(chunk)
            sha=digest.hexdigest()
            if f['lfs']: assert sha==f['lfs']['sha256'], (p,'LFS SHA256 mismatch')
            print('VERIFIED',key,f['path'],flush=True)
            return {'path':p,'bytes':f['size'],'sha256':sha,'revision':inv['revision']}
        except Exception as exc:
            print('RETRY',key,f['path'],type(exc).__name__,flush=True)
            if attempt==3: raise
            time.sleep(5*(attempt+1))
# Put smoke episodes first so CPU inspection can begin while the rest downloads.
jobs.sort(key=lambda j: (0 if 'episode_000/' in j[2]['path'] else 1 if j[0]!='dataset' else 2))
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
    results=list(ex.map(download,jobs))
(ROOT/'reports/download_verified.json').write_text(json.dumps(results,indent=2))
print('ALL_DOWNLOADS_VERIFIED',len(results),flush=True)
