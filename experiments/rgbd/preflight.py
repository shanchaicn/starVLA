"""Read-only GPU-start checklist. Never start training or change cloud resources."""
import argparse,json,os,shutil,subprocess
from pathlib import Path
import torch
from safetensors import safe_open
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument("--strict",action="store_true");a=p.parse_args()
result={"data_mount":os.path.ismount('/root/data'),"free_bytes":shutil.disk_usage('/root/data').free,
        "torch":torch.__version__,"cuda_runtime":torch.version.cuda,"cuda_available":torch.cuda.is_available(),
        "cuda_devices":torch.cuda.device_count(),
        "cgroup_memory_max":Path('/sys/fs/cgroup/memory.max').read_text().strip(),
        "cgroup_cpu_max":Path('/sys/fs/cgroup/cpu.max').read_text().strip()}
verified=ROOT/'reports/download_verified.json'
result["download_manifest_present"]=verified.exists()
if verified.exists():
    manifest=json.loads(verified.read_text())
    result["download_files"]=len(manifest)
    result["npz_episodes"]=sum(x['path'].endswith('data.npz') for x in manifest)
    result["file_size_mismatches"]=[x["path"] for x in manifest if not Path(x["path"]).exists() or Path(x["path"]).stat().st_size!=x["bytes"]]
model_root=Path('/root/data/models/Qwen3-VL-4B-Instruct')
index=json.loads((model_root/'model.safetensors.index.json').read_text())
keys=set()
for path in model_root.glob('*.safetensors'):
    with safe_open(path,framework='pt') as f:keys.update(f.keys())
result["qwen_weight_index_complete"]=keys==set(index["weight_map"])
contract=json.loads((ROOT/'contract.json').read_text())
result["contract"]=contract
result["blockers"]=[]
for field in ("depth_units_verified","temporal_alignment_verified","action_semantics_verified","intrinsics_verified","simulation_interface_verified"):
    if not contract.get(field):result["blockers"].append(field)
if contract["alignment_status"]!="aligned":result["blockers"].append("LingBot requires aligned RGB-D")
if not result["cuda_available"]:result["blockers"].append("full pretrained model/GPU training and rendering pending")
if result.get("file_size_mismatches"):result["blockers"].append("download sizes mismatched")
for key,path in [('starvla','/root/data/projects/starVLA-rgbd'),('lingbot','/root/data/projects/lingbot-depth'),('libero','/root/data/projects/LIBERO-rgbd')]:
    result[key+"_commit"]=subprocess.check_output(['git','-C',path,'rev-parse','HEAD'],text=True).strip()
(ROOT/'reports/preflight.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
if a.strict and result["blockers"]:raise SystemExit(2)
