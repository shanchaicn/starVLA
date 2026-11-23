#!/usr/bin/env python3
"""Stop only Snap/Insert StarVLA training; --dry-run only lists matching processes."""
from pathlib import Path
import os,signal,sys,time
if sys.argv[1:] not in ([],['--dry-run']):
    raise SystemExit('Usage: stop_snap_insert_training.py [--dry-run]')
root='/root/data/starVLA/'
config_dir=root+'examples/realRobots/dualarmLTSnapInsert/train_files/'
def matching(pid):
    try:
        args=Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')
        if not args[0].startswith(root+'.venv/bin/python'):return None
        if '--config_yaml' not in args:return None
        config=args[args.index('--config_yaml')+1]
        if Path(config).parent!=Path(config_dir):return None
        if not any(Path(config).name.startswith(f'act_{task}_v2_h50') for task in ('snap','insert')):return None
        return args
    except (OSError,ValueError,IndexError):return None
pids=[int(p.name) for p in Path('/proc').iterdir() if p.name.isdigit() and matching(p.name)]
for pid in pids:print('MATCH',pid,' '.join(matching(pid) or []),flush=True)
if '--dry-run' in sys.argv:raise SystemExit(0)
for pid in pids:
    if matching(pid):
        try:os.kill(pid,signal.SIGTERM)
        except ProcessLookupError:pass
end=time.monotonic()+15
while time.monotonic()<end and any(matching(pid) for pid in pids):time.sleep(.25)
remaining=[pid for pid in pids if matching(pid)]
if remaining:raise SystemExit(f'Still exiting: {remaining}; wait and inspect before starting new training')
print('STOPPED: Snap/Insert training processes; checkpoints and logs preserved',flush=True)
