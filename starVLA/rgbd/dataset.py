"""Direct compressed NPZ adapter. No conversion of depth to images."""
import json
from collections import OrderedDict
from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from .preprocessing import CAMERAS, metric_depth, normalize_actions, normalize_state

class Libero3D(Dataset):
    def __init__(self, root, selection, stats, horizon=16, split="all", depth_encoding=None, cache_episodes=2):
        self.root = Path(root)
        self.selection = json.loads(Path(selection).read_text())
        self.stats = json.loads(Path(stats).read_text())
        self.horizon, self.depth_encoding = horizon, depth_encoding
        self.cache, self.cache_episodes = OrderedDict(), cache_episodes
        self.episodes, self.index, weights = [], [], []
        for task_id, task in enumerate(self.selection["tasks"]):
            folder = self.root / task["relative_dir"]
            meta = json.loads((folder/"metadata.json").read_text())
            assert meta["state_semantics"] == ["eef_pos_x_y_z","eef_axis_angle_x_y_z","gripper_qpos_2"]
            start = len(self.index)
            for i, ep in enumerate(meta["episodes"]):
                if split == "smoke" and i not in task["smoke_episodes"]: continue
                if split == "train" and i >= 45: continue
                if split == "validation" and i < 45: continue
                # RGB-D/calibration and action[t] are at the saved pre-action
                # MuJoCo state. The processed 8-D state[t] was copied from the
                # post-action observation, so state[t-1] is the aligned
                # proprioception. Frame zero has no aligned processed state.
                if ep["num_steps"] <= horizon: raise ValueError("Episode has no aligned action horizon")
                path = folder/ep["episode_dir"]/ep["npz_file"]
                if not path.exists(): raise FileNotFoundError(path)
                ep_id = len(self.episodes)
                self.episodes.append((path, meta, task_id))
                self.index.extend((ep_id, t) for t in range(1, ep["num_steps"]-horizon+1))
            count = len(self.index)-start
            if not count: raise ValueError("Empty task split")
            weights.extend([1./count]*count)
        self.weights = weights

    def __len__(self): return len(self.index)

    def sampler(self, num_samples, generator=None):
        return WeightedRandomSampler(self.weights, num_samples, replacement=True, generator=generator)

    def _load(self, ep_id):
        if ep_id not in self.cache:
            with np.load(self.episodes[ep_id][0], allow_pickle=False) as z:
                self.cache[ep_id] = {k: z[k] for k in z.files if k != "wrist_segmentation"}
            while len(self.cache) > self.cache_episodes: self.cache.popitem(last=False)
        self.cache.move_to_end(ep_id)
        return self.cache[ep_id]

    def __getitem__(self, idx):
        ep_id,t = self.index[idx]
        _, meta, task_id = self.episodes[ep_id]
        z = self._load(ep_id)
        name = meta["task_name"]
        # Standard LIBERO language convention removes uppercase scene prefix.
        words = name.split("_")
        first = next((i for i,w in enumerate(words) if not w.isupper() and not w.startswith("SCENE")), 0)
        instruction = self.selection["tasks"][task_id].get("instruction", " ".join(words[first:]))
        item = {"image":[Image.fromarray(z[c+"_rgb"][t]) for c in CAMERAS],
                "lang":instruction,
                "state":normalize_state(z["state"][t-1:t], self.stats["state"]),
                "action":normalize_actions(z["actions"][t:t+self.horizon]),
                "task_id":task_id, "episode_id":ep_id, "timestep":t}
        if self.depth_encoding:
            pairs=[metric_depth(z[c+"_depth"][t],encoding=self.depth_encoding) for c in CAMERAS]
            item.update(depth=np.stack([p[0] for p in pairs]),depth_mask=np.stack([p[1] for p in pairs]),
                        intrinsics=np.stack([np.array(meta["cameras"]["agentview"]["intrinsic_static"],np.float32),z["wrist_intrinsic"][t]]),
                        extrinsics=np.stack([np.array(meta["cameras"]["agentview"]["extrinsic_static"],np.float32),z["wrist_extrinsic"][t]]))
        return item

def collate_examples(examples): return examples
