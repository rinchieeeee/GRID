"""Start durable TIGER training with the original config and max_epochs=100."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/toys_original_100epochs'


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def monitor(out):
    os.chdir(ROOT)
    os.environ.update(OMP_NUM_THREADS='4', TF_NUM_INTRAOP_THREADS='1', TF_NUM_INTEROP_THREADS='1', TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
    cmd = [sys.executable, '-u', '-m', 'src.train',
           'experiment=tiger_train_flat', 'data_dir=data/amazon_data/toys',
           'semantic_id_path=artifacts/toys_original_config/02_semantic_ids/pickle/merged_predictions_tensor.pt',
           'num_hierarchies=4', 'trainer.max_epochs=100', f'hydra.run.dir={out}']
    status = {'state':'starting', 'started_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
              'monitor_pid':os.getpid(), 'command':cmd, 'output_dir':str(out)}
    started = time.monotonic()
    save(out/'status.json', status)
    try:
        with (out/'console.log').open('a') as log:
            process = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            status.update(state='running', training_launcher_pid=process.pid)
            save(out/'status.json',status)
            code = process.wait()
        status.update(state='completed' if code == 0 else 'failed', exit_code=code)
    except Exception as exc:
        status.update(state='failed', error=repr(exc))
    status.update(finished_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),elapsed_seconds=time.monotonic()-started)
    save(out/'status.json',status)


if __name__ == '__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--monitor':
        monitor(Path(sys.argv[2]))
    else:
        BASE.mkdir(parents=True,exist_ok=True)
        latest=BASE/'latest_run.json'
        if latest.exists():
            previous=json.loads(latest.read_text())
            state_path=Path(previous['output_dir'])/'status.json'
            if state_path.exists():
                state=json.loads(state_path.read_text())
                if state['state'] in ['starting','running']:
                    try:
                        os.kill(state['monitor_pid'],0)
                    except ProcessLookupError:
                        pass
                    else:
                        raise SystemExit(f"Training already running: {state_path}")
        out=BASE/time.strftime('%Y%m%d_%H%M%S')
        out.mkdir()
        with (out/'monitor.log').open('a') as log:
            process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--monitor',str(out)],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        reference={'monitor_pid':process.pid,'output_dir':str(out)}
        save(latest,reference)
        print(json.dumps(reference,indent=2))
