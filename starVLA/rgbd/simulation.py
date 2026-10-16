"""LIBERO observations -> the same metric-depth/state interface as NPZ training."""
import numpy as np
from .preprocessing import metric_depth, normalize_state

def transform_camera(rgb,depth,K,orientation):
    h,w=depth.shape
    A=np.eye(3,dtype=np.float32)
    if orientation in ("flip_x","rotate_180"):
        rgb=rgb[:,::-1];depth=depth[:,::-1]
        A=np.array([[-1,0,w-1],[0,1,0],[0,0,1]],np.float32)@A
    if orientation in ("flip_y","rotate_180"):
        rgb=rgb[::-1];depth=depth[::-1]
        A=np.array([[1,0,0],[0,-1,h-1],[0,0,1]],np.float32)@A
    if orientation not in ("identity","flip_x","flip_y","rotate_180"):raise ValueError("Unverified simulator image orientation")
    return np.ascontiguousarray(rgb),np.ascontiguousarray(depth),A@K

def observation_example(env,obs,instruction,stats,contract):
    from robosuite.utils.camera_utils import get_camera_intrinsic_matrix, get_camera_extrinsic_matrix
    from robosuite.utils.transform_utils import quat2axisangle
    if not contract.get("simulation_interface_verified"):
        raise ValueError("Compare simulator cameras/units/timing to dataset before evaluation")
    sim=env.env.sim
    near=float(sim.model.vis.map.znear*sim.model.stat.extent)
    far=float(sim.model.vis.map.zfar*sim.model.stat.extent)
    images=[];depths=[];masks=[];intrinsics=[];extrinsics=[]
    for camera,alias in [("agentview","agentview"),("robot0_eye_in_hand","wrist")]:
        rgb=obs[camera+"_image"];z=obs[camera+"_depth"].squeeze(-1)
        K=get_camera_intrinsic_matrix(sim,camera,*z.shape)
        depth,valid=metric_depth(z,encoding="mujoco_zbuffer",near=near,far=far)
        orientation=contract["simulation_orientation"][alias]
        rgb,depth,K=transform_camera(rgb,depth,K,orientation)
        # In this representation invalid pixels remain zero through the orientation transform.
        images.append(rgb);depths.append(depth);masks.append(depth>0);intrinsics.append(K)
        extrinsics.append(get_camera_extrinsic_matrix(sim,camera))
    state=np.concatenate([obs["robot0_eef_pos"],quat2axisangle(obs["robot0_eef_quat"]),obs["robot0_gripper_qpos"]])[None]
    return {"image":images,"lang":instruction,"state":normalize_state(state,stats["state"]),
            "depth":np.stack(depths),"depth_mask":np.stack(masks),"intrinsics":np.stack(intrinsics),
            "extrinsics":np.stack(extrinsics)}
