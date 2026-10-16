"""Narrow compatibility for official LIBERO initial-state numpy pickles / torch 2.6."""
from pathlib import Path
import torch

def load_init_states(suite,task_id):
    from libero.libero import get_libero_path
    task=suite.get_task(task_id)
    root=Path(get_libero_path('init_states')).resolve()
    path=(root/task.problem_folder/task.init_states_file).resolve()
    if not path.is_relative_to(root):raise ValueError('Initial state path escapes official repository')
    # Only the pinned official LIBERO files. Never use this for arbitrary uploaded checkpoints.
    return torch.load(path,map_location='cpu',weights_only=False)
