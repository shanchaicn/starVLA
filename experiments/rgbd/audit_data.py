"""Read-only NPZ audit, visual evidence, shared state statistics; no assumed alignment."""
import argparse, json
from pathlib import Path
import numpy as np
from scipy.ndimage import sobel, binary_dilation
from PIL import Image, ImageDraw
p=argparse.ArgumentParser()
p.add_argument("--all",action="store_true")
args=p.parse_args()
root=Path('/root/data/datasets/LIBERO-3D-selected')
out=Path(__file__).resolve().parent/'reports'
sel=json.loads((out.parent/'selected_tasks.json').read_text())
report={"alignment_status":"alignment_unknown","tasks":[],"notes":[
    "Edge overlap is diagnostic, not proof of calibration, depth scale, or synchronization.",
    "Public repository includes no data-generation source. Temporal and unit semantics remain unverified.",
    "Test all RGB/depth orientation hypotheses; no automatic registration based on edge score."
]}
state_lo=np.full(8,np.inf);state_hi=-state_lo
for task in sel["tasks"]:
    folder=root/task["relative_dir"]; meta=json.loads((folder/'metadata.json').read_text())
    record={"suite":task["suite"],"task_name":task["task_name"],"episodes":[]}
    panels=[]
    for i, ep in enumerate(meta["episodes"]):
        if not args.all and i!=0: continue
        path=folder/ep["episode_dir"]/'data.npz'
        if not path.exists(): raise FileNotFoundError(path)
        with np.load(path,allow_pickle=False) as z:
            fields={}
            for k in z.files:
                x=z[k]
                assert list(x.shape)==ep["shapes"][k],(path,k,x.shape)
                assert x.shape[0]==ep["num_steps"],(path,k,"time length")
                fields[k]={"shape":list(x.shape),"dtype":str(x.dtype),"min":float(np.nanmin(x)),"max":float(np.nanmax(x)),"nonfinite":int((~np.isfinite(x)).sum())}
                if k=="state" and i<45:
                    state_lo=np.minimum(state_lo,x.min(0));state_hi=np.maximum(state_hi,x.max(0))
                if k=="actions":
                    assert x.shape[-1]==7 and np.isfinite(x).all() and (abs(x)<=1.0001).all()
            record["episodes"].append({"episode":i,"fields":fields})
            if i==0:
                diag={}
                for cam in ("agentview","wrist"):
                    rgb=z[cam+"_rgb"][0]; d=z[cam+"_depth"][0]
                    gray=rgb.mean(-1).astype(float)
                    edge=np.hypot(sobel(gray,0),sobel(gray,1))
                    re=edge>np.quantile(edge,.85);re=binary_dilation(re,iterations=2)
                    scores={}
                    for key,dep in [("identity",d),("flip_y",d[::-1]),("flip_x",d[:,::-1]),("rotate_180",d[::-1,::-1])]:
                        de=np.hypot(sobel(dep,0),sobel(dep,1))
                        mask=de>np.quantile(de,.95)
                        scores[key]=float((re&mask).sum()/max(mask.sum(),1))
                    diag[cam]={"depth_edge_rgb_overlap":scores,"zero_depth":int((d==0).sum())}
                    dn=np.clip((d-np.quantile(d,.01))/(np.quantile(d,.99)-np.quantile(d,.01)),0,1)
                    depim=np.uint8(np.stack([dn,1-np.abs(2*dn-1),1-dn],-1)*255)
                    de=np.hypot(sobel(d,0),sobel(d,1));overlay=rgb.copy()
                    overlay[de>np.quantile(de,.95)]=[255,0,255]
                    panel=Image.new('RGB',(768,282),'white')
                    for x,im in enumerate([rgb,depim,overlay]): panel.paste(Image.fromarray(im),(256*x,26))
                    ImageDraw.Draw(panel).text((4,5),cam+" | RGB / relative depth color / raw depth edges",fill='black')
                    panels.append(panel)
                    # Unscaled camera-frame XYZ is saved for inspection, explicitly not world-frame proof.
                    K=np.array(meta["cameras"]["agentview"]["intrinsic_static"]) if cam=="agentview" else z["wrist_intrinsic"][0]
                    yy,xx=np.mgrid[0:d.shape[0]:4,0:d.shape[1]:4]; zz=d[::4,::4]
                    xyz=np.stack([(xx-K[0,2])/K[0,0]*zz,(yy-K[1,2])/K[1,1]*zz,zz],-1).reshape(-1,3)
                    colors=rgb[::4,::4].reshape(-1,3)
                    ply=out/(task["suite"]+"_"+cam+"_raw_units.ply")
                    with ply.open('w') as f:
                        f.write("ply\nformat ascii 1.0\ncomment Depth units and orientation unverified\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"%len(xyz))
                        np.savetxt(f,np.concatenate([xyz,colors],1),fmt="%.5f %.5f %.5f %d %d %d")
                record["diagnostics"]=diag
    canvas=Image.new('RGB',(768,564),'white')
    for j,im in enumerate(panels):canvas.paste(im,(0,j*282))
    canvas.save(out/(task["suite"]+"_alignment.png"))
    report["tasks"].append(record)
    print(task["suite"],len(record["episodes"]),record.get("diagnostics"),flush=True)
label='all' if args.all else 'sample'
(out/f'data_audit_{label}.json').write_text(json.dumps(report,indent=2))
if args.all:
    stats={"state":{"min":state_lo.tolist(),"max":state_hi.tolist(),"semantics":meta["state_semantics"]},
           "action":{"normalization":"identity_native_libero_minus1_plus1","gripper":"-1 open, +1 close; requires source confirmation"},
           "fit_episodes":"0..44 in each task (180 trajectories); reused unchanged for all E0/E1/E2, smoke and full training",
           "dataset_revision":sel["revision"]}
    (out.parent/'normalization.json').write_text(json.dumps(stats,indent=2))
