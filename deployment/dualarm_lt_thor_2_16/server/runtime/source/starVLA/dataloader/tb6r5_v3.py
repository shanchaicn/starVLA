"""TB6-R5 archived-model 8D LeRobot v3 adapter; source dataset remains read-only.

No URDF transform or hardware gripper mapping is applied here. Artifact statistics
are fitted on the training split with interpolated pose labels excluded per cell.
"""
import csv
import hashlib
import json
from pathlib import Path

import av
import numpy as np
import pandas as pd
from PIL import Image

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset
from starVLA.dataloader.gr00t_lerobot.schema import DatasetMetadata, LeRobotModalityMetadata

PARTS = {'xyz': (0, 3), 'quaternion_xyzw': (3, 7), 'gripper': (7, 8)}


class TB6R5V3Dataset(LeRobotSingleDataset):
    """Uses native v3 table/task/index readers and transforms, with explicit masks.

    No constructor side effects, stats_gr00t.json, steps.pkl or writes in raw meta.
    Validation enumerates its own episodes rather than sampling with replacement.
    """

    def __init__(self, *args, **kwargs):
        cfg = kwargs['data_cfg']
        self.artifact_dir = Path(cfg['artifact_dir']).resolve()
        self.split_name = cfg['split']
        if self.split_name not in ('train', 'val'):
            raise ValueError('Explicit train/val split required')
        if cfg.get('action_mode') != 'abs' or int(cfg.get('action_horizon', 0)) != 8:
            raise ValueError('This contract is absolute 8D with H=8; no implicit conversion')
        if cfg.get('supervision_policy') != 'mask_tail_and_interpolated_pose':
            raise ValueError('Unknown supervision policy; masks must not silently disappear')
        if cfg.get('delete_pause_frame', False) or cfg.get('drop_incomplete_action_chunks', False):
            raise ValueError('This adapter uses all base frames with explicit masks, not silent filtering')
        if cfg.get('video_backend') != 'pyav_readonly' or cfg.get('state_action_dtype') != 'float32':
            raise ValueError('Contract requires read-only PyAV and float32 state/actions')
        self.split_manifest = json.loads((self.artifact_dir/'episode_split.json').read_text())
        dataset_path = Path(kwargs['dataset_path']).resolve()
        # Small audited dataset: verify each actual input, including video/task metadata.
        # Relative paths remain portable when an explicitly approved transfer occurs later.
        for item in self.split_manifest['dataset_files']:
            path = (dataset_path/item['relative_path']).resolve()
            if not path.is_relative_to(dataset_path):
                raise ValueError('Invalid manifest path escapes dataset')
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                while block := stream.read(8 * 1024 * 1024):
                    digest.update(block)
            if digest.hexdigest() != item['sha256']:
                raise ValueError(f'Dataset fingerprint differs from S0.1: {path.name}')
        self.fit_stats = json.loads((self.artifact_dir/'train_statistics.json').read_text())
        split_hash = hashlib.sha256((self.artifact_dir/'episode_split.json').read_bytes()).hexdigest()
        if self.fit_stats['split_sha256'] != split_hash:
            raise ValueError('Statistics belong to a different episode split')
        train, val = map(set, [self.split_manifest['train'], self.split_manifest['val']])
        if not train or not val or train & val:
            raise ValueError('Episode split is empty or overlaps')
        if self.fit_stats['train_episode_ids'] != self.split_manifest['train']:
            raise ValueError('Validation/global statistics cannot be used as train statistics')
        flags_file = self.artifact_dir/'sample_quality_flags.csv'
        if hashlib.sha256(flags_file.read_bytes()).hexdigest() != self.split_manifest['quality_flags_sha256']:
            raise ValueError('Quality sidecar fingerprint mismatch')
        self.quality_rows = list(csv.DictReader(flags_file.open()))
        self.flags = {}
        self._validated_episodes = set()
        for row in self.quality_rows:
            key = (int(row['episode_index']), int(row['frame_index']))
            if key in self.flags:
                raise ValueError('Duplicate frame in quality sidecar')
            self.flags[key] = row
        super().__init__(*args, **kwargs)

    def _get_lerobot_modality_meta(self):
        return LeRobotModalityMetadata.model_validate(json.loads((self.artifact_dir/'modality.json').read_text()))

    def _get_metadata(self, embodiment_tag):
        le = self._get_lerobot_modality_meta()
        info = json.loads((self.dataset_path/'meta/info.json').read_text())
        if info.get('codebase_version') != 'v3.0' or info['fps'] != 30:
            raise ValueError('Dataset version/FPS differs from audited contract')
        for key in ('action', 'observation.state'):
            if info['features'][key]['shape'] != [8]:
                raise ValueError('Action/state dimensions must both be exactly eight')
        modalities, statistics = {'state': {}, 'action': {}, 'video': {}}, {'state': {}, 'action': {}}
        for group, column in [('state', 'observation.state'), ('action', 'action')]:
            for key, (lo, hi) in PARTS.items():
                spec = getattr(le, group)[key]
                if (spec.start, spec.end, spec.original_key, spec.absolute) != (lo, hi, column, True):
                    raise ValueError('Modality mapping differs from audited source layout')
                modalities[group][key] = {'absolute': True, 'shape': [hi-lo], 'continuous': True,
                                          'rotation_type': 'quaternion' if lo == 3 else None}
                statistics[group][key] = {k: np.asarray(v[lo:hi], dtype=np.float64)
                                         for k, v in self.fit_stats['columns'][column].items()}
        for view in ('front', 'wrist'):
            original = 'observation.images.'+view
            if le.video[view].original_key != original or info['features'][original]['shape'] != [480, 640, 3]:
                raise ValueError('Camera mapping or audited image dimensions differ')
            modalities['video'][view] = {'resolution': [640, 480], 'channels': 3, 'fps': 30}
        return DatasetMetadata(statistics=statistics, modalities=modalities, embodiment_tag=embodiment_tag)

    def _get_trajectories(self):
        ids, lengths = super()._get_trajectories()
        selected = self.split_manifest[self.split_name]
        if set(ids.tolist()) != set(self.split_manifest['train'] + self.split_manifest['val']):
            raise ValueError('Split does not cover the actual episode inventory')
        keep = np.isin(ids, selected)
        return ids[keep], lengths[keep]

    def _get_all_steps(self):
        # All starts remain explicit; future out-of-episode cells carry false masks.
        steps = []
        for episode, length in zip(self.trajectory_ids, self.trajectory_lengths):
            for frame in range(int(length)):
                if (int(episode), frame) not in self.flags:
                    raise ValueError('Missing audit quality record')
                steps.append((int(episode), frame))
        if len(self.flags) != self.lerobot_info_meta['total_frames']:
            raise ValueError('Quality sidecar is not one row per dataset frame')
        return steps

    def get_trajectory_data_lerobot_v3(self, trajectory_id):
        if int(trajectory_id) not in set(self.trajectory_ids.tolist()):
            raise ValueError('Cross-split episode access rejected')
        data = super().get_trajectory_data_lerobot_v3(trajectory_id)
        expected = int(self.trajectory_lengths[self.get_trajectory_index(trajectory_id)])
        if len(data) != expected or not np.array_equal(data['frame_index'].to_numpy(), np.arange(expected)):
            raise ValueError('Episode table length/frame order differs from metadata')
        if trajectory_id not in self._validated_episodes:
            for _, row in data.iterrows():
                flag = self.flags[(int(trajectory_id), int(row['frame_index']))]
                if int(flag['index']) != int(row['index']):
                    raise ValueError('Audit sidecar global index differs from data')
            self._validated_episodes.add(trajectory_id)
        return data

    def get_video(self, trajectory_id, key, base_index):
        # Per-view v3 paths/offsets come from the native reader, not data file indices.
        path = self.get_video_path(trajectory_id, key.removeprefix('video.'))
        original = self.lerobot_modality_meta.video[key.removeprefix('video.')].original_key
        start = self.trajectory_ids_to_metadata[trajectory_id]['videos/from_timestamps'][original]
        local_time = float(self.curr_traj_data['timestamp'].iloc[base_index])
        target = start + local_time
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 1
            container.seek(max(0, int(target / stream.time_base)), stream=stream, backward=True, any_frame=False)
            for frame in container.decode(stream):
                pts = float(frame.pts * frame.time_base)
                if pts < target - 5e-4:
                    continue
                if abs(pts-target) > 5e-4:
                    raise ValueError(f'Video frame PTS mismatch: {pts} vs {target}')
                rgb = frame.to_ndarray(format='rgb24')
                if rgb.shape != (480, 640, 3):
                    raise ValueError('Decoded video shape changed')
                return rgb[None]
        raise ValueError(f'No frame decoded at required timestamp {target}: {path}')

    def get_step_data(self, trajectory_id, base_index):
        if trajectory_id not in self.trajectory_ids:
            raise ValueError('Cross-split access rejected')
        n = int(self.trajectory_lengths[self.get_trajectory_index(trajectory_id)])
        if not 0 <= base_index < n:
            raise IndexError('Base frame outside episode')
        data = super().get_step_data(trajectory_id, base_index)
        mask = np.zeros((8, 8), dtype=bool)
        reasons = []
        for step in range(8):
            f = base_index + step
            if f >= n:
                reasons.append({'frame': f, 'tail': True})
                continue
            quality = self.flags[(int(trajectory_id), f)]
            mask[step] = True
            if int(quality['source_target_pose_interpolated']):
                mask[step, :7] = False  # Preserve real gripper supervision.
            reasons.append(dict(quality))
        data['action_mask'] = mask
        data['provenance'] = {'episode_index': int(trajectory_id), 'frame_index': int(base_index),
                              'split': self.split_name, 'label_records': reasons,
                              'reference_frame': 'recorded_model_frame_unverified_physical_tcp',
                              'tail_storage': 'native_first_last_repeat_explicitly_masked_no_real_future_label'}
        return data

    def _pack_sample(self, data):
        sample = super()._pack_sample(data)
        mask = data.get('action_mask')
        if not isinstance(mask, np.ndarray) or mask.dtype != bool or mask.shape != (8, 8):
            raise ValueError('A boolean [H=8,D=8] action mask is required')
        if sample['action'].shape != (8, 8) or not np.isfinite(sample['action'][mask]).all():
            raise ValueError('Malformed or non-finite supervised actions')
        sample['action_mask'] = mask.copy()
        sample['provenance'] = data['provenance']
        return sample

    def save_dataset_statistics(self, save_path, **kwargs):
        # Both splits serialize the SAME train-only stats and identity; no fitting here.
        output = {}
        for group, column in [('action', 'action'), ('state', 'observation.state')]:
            output[group] = dict(self.fit_stats['columns'][column])
        output['action']['mask'] = [True] * 8  # Normalization selection, NOT loss mask.
        path = Path(save_path).resolve()
        if path.is_relative_to(self.dataset_path.resolve()):
            raise ValueError('Refuse to write statistics in the original dataset')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({self.tag: output}, indent=2)+'\n')
