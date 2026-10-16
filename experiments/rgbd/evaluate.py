"""Paired 50-rollout LIBERO evaluation. Never synthesize success or offline MSE."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from libero.libero import benchmark,get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from starVLA.rgbd.simulation import observation_example
from starVLA.rgbd.libero_compat import load_init_states
from starVLA.rgbd.preprocessing import validate_contract
from starVLA.rgbd.checkpoint import sha256

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--port",type=int,default=10093);p.add_argument("--trials",type=int,default=50)
    p.add_argument("--seed",type=int,default=7);p.add_argument("--output",required=True)
    p.add_argument("--ablation",choices=["none","shuffle","invalid"],default="none")
    p.add_argument("--selection",default="experiments/rgbd/selected_tasks.json")
    p.add_argument("--contract",default="experiments/rgbd/contract.json")
    p.add_argument("--stats",default="experiments/rgbd/normalization.json")
    a=p.parse_args()
    if a.trials<50:raise ValueError("Formal comparison requires >=50 rollouts/task")
    sel=json.loads(Path(a.selection).read_text());contract=json.loads(Path(a.contract).read_text());stats=json.loads(Path(a.stats).read_text())
    client=WebsocketClientPolicy("127.0.0.1",a.port);meta=client.get_server_metadata()
    if not meta.get("native_libero_actions"):raise ValueError("This evaluator requires native signed actions")
    if not contract.get("simulation_interface_verified"):raise ValueError("Simulation/data interface unverified")
    expected={"selected_tasks":sha256(a.selection),"normalization":sha256(a.stats),"contract":sha256(a.contract)}
    if meta["identity"]!=expected:raise ValueError("Server and client preprocessing mismatch")
    if meta["encoder"]!="rgb":validate_contract(contract,lingbot=meta["encoder"]=="lingbot")
    out=Path(a.output).resolve()
    if not out.is_relative_to("/root/data/outputs"):raise ValueError("Output outside data outputs")
    out.mkdir(parents=True,exist_ok=False)
    (out/'protocol.json').write_text(json.dumps(vars(a)|{"server":meta,"control_hz":20,
        "warmup_steps":10,"execute_horizon":meta["action_chunk_size"],
        "shuffle":"depth+valid mask from fixed unrelated training frame, same task and camera; RGB/K remain current",
        "action_mse":"not defined online without expert action; compute on held-out demonstrations",
        "height_position":"initial MuJoCo object body world positions logged; no scene perturbation claimed"},indent=2))
    rows=[]
    limits={"libero_spatial":220,"libero_object":280,"libero_goal":300,"libero_10":520}
    for entry in sel["tasks"]:
        suite=benchmark.get_benchmark_dict()[entry["suite"]]()
        ids=[i for i in range(suite.n_tasks) if suite.get_task(i).name==entry["task_name"]]
        if len(ids)!=1:raise ValueError("Selected task missing/ambiguous in simulator")
        task_id=ids[0];task=suite.get_task(task_id);initial=load_init_states(suite,task_id)
        if a.trials>len(initial):raise ValueError("Not enough unique initial states")
        env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path("bddl_files"))/task.problem_folder/task.bddl_file),
            camera_heights=256,camera_widths=256,camera_depths=True,control_freq=20)
        try:
            donor=None
            if a.ablation=="shuffle":
                from starVLA.rgbd.preprocessing import metric_depth
                path=Path('/root/data/datasets/LIBERO-3D-selected')/entry["relative_dir"]/'episode_000/data.npz'
                with np.load(path,allow_pickle=False) as z:
                    # Retain all donor times, sample deterministically at every query.
                    donor=[z[c+"_depth"] for c in ("agentview","wrist")]
            for trial in range(a.trials):
                seed=a.seed+trial;np.random.seed(seed);env.seed(seed);env.reset()
                init=np.asarray(initial[trial]);obs=env.set_init_state(init)
                for _ in range(10):obs,_,_,_=env.step([0.]*6+[-1.])
                positions={}
                raw=env.env
                for name in getattr(raw,"obj_body_id",{}):
                    body=raw.obj_body_id[name];positions[name]=raw.sim.data.body_xpos[body].tolist()
                start=time.perf_counter();success=False;latencies=[];peak=[];chunk=None;cursor=0
                rng=np.random.default_rng(seed);actions_log=[]
                for step in range(limits[entry["suite"]]):
                    if chunk is None or cursor>=len(chunk):
                        e=observation_example(env,obs,task.language,stats,contract)
                        if a.ablation=="invalid":e["depth"][:]=0;e["depth_mask"][:]=False
                        elif a.ablation=="shuffle":
                            t=int(rng.integers(len(donor[0])))
                            pairs=[metric_depth(d[t],encoding=contract["depth_encoding"]) for d in donor]
                            e["depth"]=np.stack([x[0] for x in pairs]);e["depth_mask"]=np.stack([x[1] for x in pairs])
                        response=client.predict_action({"examples":[e],"seed":seed*10000+step})
                        if not response.get("ok"):raise RuntimeError(response)
                        data=response["data"];chunk=np.asarray(data["actions"])[0];cursor=0
                        latencies.append(data["inference_latency_s"]);peak.append(data["gpu_peak_bytes"])
                    action=chunk[cursor];cursor+=1
                    if action.shape!=(7,) or not np.isfinite(action).all():raise ValueError("Invalid server action")
                    obs,reward,done,info=env.step(action.tolist());actions_log.append(action.tolist())
                    if env.check_success():success=True;break
                record={"suite":entry["suite"],"task":entry["task_name"],"trial":trial,"seed":seed,
                    "init_sha256":hashlib.sha256(init.tobytes()).hexdigest(),"success":success,"steps":step+1,
                    "completion_sim_s":(step+1)/20 if success else None,"wall_s":time.perf_counter()-start,
                    "inference_latency_mean_s":float(np.mean(latencies)),"gpu_peak_bytes":max(peak),
                    "initial_object_positions":positions,"action_mse":None,"ablation":a.ablation}
                rows.append(record)
                with (out/'rollouts.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                np.save(out/(entry["suite"]+f'_{trial:03}_actions.npy'),np.array(actions_log,np.float32))
        finally:env.close()
    summary={t["suite"]:{"episodes":len(r:=[x for x in rows if x["suite"]==t["suite"]]),
                           "success_rate":float(np.mean([x["success"] for x in r]))} for t in sel["tasks"]}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
if __name__=="__main__":main()
