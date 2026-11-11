# pp_white_red H=50 training

The registry defines two camera views (`front`, `wrist`), one observation state,
and 50 future absolute 8D actions. The action layout is
`[x, y, z, qx, qy, qz, qw, gripper_m]`. Both task strings come from the LeRobot
v3 `task_index` column. `meta/modality.json` is installed in the dataset.

Run from `cs`:

```bash
cd /root/data/starVLA
examples/realRobots/TB6R5PPWhiteRed/train_files/train_h50.sh act --check
examples/realRobots/TB6R5PPWhiteRed/train_files/train_h50.sh act --run
examples/realRobots/TB6R5PPWhiteRed/train_files/train_h50.sh qwengroot --run
examples/realRobots/TB6R5PPWhiteRed/train_files/train_h50.sh qwenpi_v3 --run
```

For a detached two-GPU run with timestamped logs:

```bash
cd /root/data/starVLA
CUDA_VISIBLE_DEVICES=0,1 examples/realRobots/TB6R5PPWhiteRed/train_files/run_h50_nohup.sh qwengroot
CUDA_VISIBLE_DEVICES=0,1 examples/realRobots/TB6R5PPWhiteRed/train_files/run_h50_nohup.sh qwenpi_v3
```

Follow the latest log with `tail -f` under
`/root/data/outputs/starVLA/pp_white_red_h50/logs/<model>_latest.log`.

When the GPUs are idle, run a ten-step checkpoint save/load smoke test with
`test_checkpoint_save.sh qwenpi_v3 10`. Validate an existing checkpoint without
using a GPU with `verify_checkpoint.sh /absolute/path/to/checkpoint.pt`.

Use `--dry-run` to print the command, append `--steps 500` for a short run,
or append `--num-processes 1` for one GPU. The default is two GPU processes.
Start one model at a time. Output directories are set by each YAML `run_id`.

ACT does not consume the text task instruction, so it is only an action
prediction baseline for the mixed red/white dataset. QwenGR00T and QwenPI_v3
consume task text. The Qwen models use the local Qwen3-VL-4B-Instruct-Action
weights as their visual-language base and fine-tune that VLM at 1e-5; their
action heads begin fresh and train at 1e-4. All
ACT and QwenGR00T use batch size 1 per GPU; QwenPI_v3 currently tests batch
size 2 per GPU. All use accumulation 1. ACT uses 10,000 steps; QwenGR00T and
QwenPI_v3 use 20,000 steps and save every 5,000 steps.
This is a first comparable run, not a tuned recipe. The dataset has no separate held-out
validation split, so training loss alone cannot establish task success.
