"""Summarize the recorded Toys experiment without loading model weights."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/toys_rkmeans_10epochs'
rows = []
for metrics in sorted((OUT/'04_tiger/csv').glob('version_*/metrics.csv')):
    with metrics.open() as stream:
        rows.extend(csv.DictReader(stream))

def collect(prefix):
    result=[]
    for row in rows:
        values={k:float(v) for k,v in row.items() if v and (k.startswith(prefix) or k in ['epoch','step'])}
        if any(k.startswith(prefix) for k in values): result.append(values)
    return result

validation=collect('val/')
testing=collect('test/')
summary={
    'training_checks':json.loads((OUT/'training_checks.json').read_text()) if (OUT/'training_checks.json').exists() else None,
    'strict_original_config_reproduction':False,
    'status':json.loads((OUT/'status.json').read_text()),
    'datasets':json.loads((ROOT/'data/dataset_stats.json').read_text()),
    'semantic_ids':json.loads((OUT/'semantic_id_checks.json').read_text()),
    'validation_history':validation,
    'best_validation':max(validation,key=lambda r:r.get('val/recall@5',-1),default=None),
    'test':testing[-1] if testing else None,
    'best_checkpoints':[str(p.relative_to(ROOT)) for p in (OUT/'04_tiger/checkpoints').glob('checkpoint*')],
}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
text=['# Toys experiment results','','Python 3.12.5, PyTorch 2.7.1+cu128, RTX 3090, BF16 mixed precision.','',
      'This is a local trial with overrides, not a strict reproduction of the original experiment configuration.', '', 'FLAN-T5-XL encoder embeddings; Residual K-means with 3 levels and 256 codes per level, 90 epochs; TIGER with 4 SID levels, 10 epochs.','',
      '| Dataset | Items | Training histories | Validation histories | Test histories |','|---|---:|---:|---:|---:|']
for name,counts in summary['datasets'].items():
    text.append('| '+name+' | '+' | '.join(str(counts[k]['records']) for k in ['items','training','evaluation','testing'])+' |')
text += ['', '| Epoch | Validation loss | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |','|---:|---:|---:|---:|---:|---:|']
for row in validation:
    text.append('| '+str(int(row['epoch'])+1)+' | '+' | '.join(f'{row.get(k,0):.6f}' for k in ['val/loss','val/recall@5','val/recall@10','val/ndcg@5','val/ndcg@10'])+' |')
text+=['','Test results use the checkpoint with the best validation Recall@5.','', '```json',json.dumps(summary['test'],indent=2),'```','','Checkpoints:']
text += ['- `'+p+'`' for p in summary['best_checkpoints']]
(OUT/'results.md').write_text('\n'.join(text)+'\n')
print(json.dumps({'completed_validation_epochs':len(validation),'best_validation':summary['best_validation'],'test':summary['test']},indent=2))
