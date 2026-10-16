"""Physics/calibration preflight without an OpenGL renderer. No success-rate claim."""
import os
os.environ["MUJOCO_GL"]="glx"
import json
from pathlib import Path
import numpy as np
from libero.libero import benchmark,get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv
from robosuite.utils.camera_utils import get_camera_intrinsic_matrix,get_camera_extrinsic_matrix
from starVLA.rgbd.libero_compat import load_init_states
root=Path(__file__).resolve().parent
sel=json.loads((root/'selected_tasks.json').read_text())
records=[]
for entry in sel["tasks"]:
    suite=benchmark.get_benchmark_dict()[entry["suite"]]()
    idx=next(i for i in range(suite.n_tasks) if suite.get_task(i).name==entry["task_name"])
    task=suite.get_task(idx);initial=load_init_states(suite,idx)
    env=ControlEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),
                   has_renderer=False,has_offscreen_renderer=False,use_camera_obs=False,control_freq=20)
    try:
        env.seed(7);env.reset();obs=env.set_init_state(initial[0]);obs,_,_,_=env.step([0.]*6+[-1.])
        sim=env.env.sim
        folder=Path('/root/data/datasets/LIBERO-3D-selected')/entry['relative_dir']
        meta=json.loads((folder/'metadata.json').read_text())
        record={'suite':entry['suite'],'task':task.name,'task_id':idx,'language':task.language,
                'initial_states':len(initial),'physics_reset_step':True,'rendering_tested':False,
                'control_hz':env.env.control_freq,'action_dim':env.env.action_dim,
                'near_m':float(sim.model.vis.map.znear*sim.model.stat.extent),
                'far_m':float(sim.model.vis.map.zfar*sim.model.stat.extent),'cameras':{}}
        with np.load(folder/'episode_000/data.npz',allow_pickle=False) as z:
            for camera,alias in [('agentview','agentview'),('robot0_eye_in_hand','wrist')]:
                K=get_camera_intrinsic_matrix(sim,camera,256,256)
                ext=get_camera_extrinsic_matrix(sim,camera)
                expected=np.array(meta['cameras']['agentview']['intrinsic_static']) if alias=='agentview' else z['wrist_intrinsic'][0]
                c={'intrinsic':K.tolist(),'intrinsic_max_abs_diff_from_dataset':float(abs(K-expected).max()),'extrinsic':ext.tolist()}
                if alias=='agentview':
                    c['static_extrinsic_max_abs_diff']=float(abs(ext-np.array(meta['cameras']['agentview']['extrinsic_static'])).max())
                record['cameras'][alias]=c
        controller=env.env.robots[0].controller
        record['controller']={'class':type(controller).__name__}
        for key in ['input_min','input_max','output_min','output_max','control_delta']:
            val=getattr(controller,key,None)
            if val is not None:record['controller'][key]=np.asarray(val).tolist()
        records.append(record);print(json.dumps(record),flush=True)
    finally:env.close()
(root/'reports/simulator_cpu.json').write_text(json.dumps(records,indent=2))
