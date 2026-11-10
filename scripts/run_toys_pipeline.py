"""Run the smallest Amazon category through embedding, R-KMeans and TIGER."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/toys_rkmeans_10epochs'
OUT.mkdir(parents=True, exist_ok=True)
os.chdir(ROOT)
os.environ.update(TF_NUM_INTRAOP_THREADS='1', TF_NUM_INTEROP_THREADS='1', OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false', HF_HUB_DISABLE_XET='1', PYTHONUNBUFFERED='1')
status = {'started_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'dataset': 'toys', 'recommendation_epochs': 10, 'rkmeans_epochs': 90, 'stages': {}}

if (OUT / 'status.json').exists():
    previous = json.loads((OUT / 'status.json').read_text())
    status['stages'] = previous.get('stages', {})

def save():
    (OUT / 'status.json').write_text(json.dumps(status, indent=2)+'\n')

def run(name, module, args):
    stage = OUT / name
    stage.mkdir(exist_ok=True)
    if (stage / '.complete').exists():
        status['stages'].setdefault(name, {})['state'] = 'completed'
        save()
        print(f'Already completed {name}', flush=True)
        return stage
    args = [a if a.startswith('experiment=') else '++' + a.lstrip('+') for a in args]
    cmd = [sys.executable, '-m', module, *args, 'trainer.devices=1', '++trainer.strategy=auto', 'trainer.precision=bf16-mixed', 'extras.print_config=false', f'hydra.run.dir={stage}']
    if module == 'src.train':
        cmd += ['++should_skip_retry=true']
    (stage/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
    status['current_stage'] = name
    status['stages'][name] = {'state':'running', 'started_at': time.time(), 'command':cmd}
    save()
    print(f'Starting {name}', flush=True)
    with (stage/'console.log').open('w') as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    status['stages'][name].update(state='completed', elapsed_seconds=time.time()-status['stages'][name]['started_at'])
    save()
    (stage / '.complete').touch()
    print(f'Completed {name}', flush=True)
    return stage

finite = 'src.data.loading.components.dataloading.FiniteSequenceIterable'
data = f'data_dir={ROOT}/data/amazon_data/toys'

def loader(prefix):
    return [f'{prefix}.num_workers=0', f'{prefix}.timeout=0', f'{prefix}.persistent_workers=false']

try:
    emb = run('01_embeddings','src.inference', ['experiment=sem_embeds_inference_flat', data, *loader('data_loading.datamodule.predict_dataloader_config')])/'pickle/merged_predictions_tensor.pt'
    import torch
    embeddings = torch.load(emb, map_location='cpu', weights_only=True)
    assert embeddings.shape == (11924,2048), embeddings.shape
    assert torch.isfinite(embeddings).all()
    batch=1024
    steps=math.ceil(11924/batch)*90
    rk = run('02_rkmeans','src.train', ['experiment=rkmeans_train_flat',data,f'embedding_path={emb}','embedding_dim=2048','num_hierarchies=3','codebook_width=256','trainer.max_epochs=90',f'trainer.max_steps={steps}','trainer.limit_val_batches=0', 'trainer.log_every_n_steps=1','model.verbose=true',f'data_loading.datamodule.train_dataloader_config.batch_size_per_device={batch}','data_loading.datamodule.train_dataloader_config.drop_last=false','data_loading.datamodule.train_dataloader_config.dataset_class._target_=src.data.loading.components.dataloading.CachedFiniteSequenceIterable',*loader('data_loading.datamodule.train_dataloader_config'),'callbacks.model_checkpoint.every_n_train_steps=null','callbacks.model_checkpoint.every_n_epochs=1','callbacks.model_checkpoint.save_on_train_epoch_end=true','callbacks.model_checkpoint.save_last=true'])
    ckpt = rk/'checkpoints/last.ckpt'
    assert ckpt.is_file(), ckpt
    sid = run('03_semantic_ids','src.inference',['experiment=rkmeans_inference_flat',data,f'embedding_path={emb}','embedding_dim=2048','num_hierarchies=3','codebook_width=256',f'ckpt_path={ckpt}','callbacks.bq_writer=null',*loader('data_loading.datamodule.predict_dataloader_config')])/'pickle/merged_predictions_tensor.pt'
    ids=torch.load(sid,map_location='cpu',weights_only=True)
    assert ids.shape == (4,11924), ids.shape
    assert torch.unique(ids.T,dim=0).shape[0] == 11924
    args=['experiment=tiger_train_flat',data,f'semantic_id_path={sid}','num_hierarchies=4','trainer.max_epochs=10','trainer.max_steps=-1','trainer.val_check_interval=1.0','trainer.log_every_n_steps=10','callbacks.early_stopping=null','callbacks.model_checkpoint.save_last=true','data_loading.train_dataloader_config.dataloader.collate_fn.max_batch_size=32','data_loading.train_dataloader_config.dataloader.drop_last=false','data_loading.train_dataloader_config.dataloader.assign_all_files_per_worker=false',f'data_loading.train_dataloader_config.dataloader.dataset_class._target_={finite}']
    for split in ['train','val','test']:
        args += loader(f'data_loading.{split}_dataloader_config.dataloader')
        if split != 'train':
            args += [f'data_loading.{split}_dataloader_config.dataloader.batch_size_per_device=128']
    run('04_tiger','src.train',args)
    status['state']='completed'
except Exception as e:
    status['state']='failed'
    status['error']=repr(e)
    save()
    raise
save()
