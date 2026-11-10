"""Observe original training settings and stop after a short runtime check."""
import json
import time
from pathlib import Path

import torch
from lightning import Callback


class TrainingRuntimeProbe(Callback):
    def __init__(self, output_dir: str, stop_after_batches: int = 64):
        self.output_dir = Path(output_dir)
        self.stop_after_batches = stop_after_batches
        self.losses = []
        self.batch_sizes = []
        self.gradient_checks = 0

    def on_train_start(self, trainer, pl_module):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.started_at = time.monotonic()
        torch.cuda.reset_peak_memory_stats()

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        grads = [p.grad for group in optimizer.param_groups for p in group['params'] if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all().item() for g in grads):
            raise RuntimeError('Missing or non-finite gradients')
        self.gradient_checks += 1

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs['loss'] if isinstance(outputs, dict) else outputs
        if loss is None or not torch.isfinite(loss).all().item():
            raise RuntimeError('Missing or non-finite loss')
        self.losses.append(float(loss.detach()))
        model_input = batch
        for _ in range(4):
            if hasattr(model_input, 'mask'):
                self.batch_sizes.append(int(model_input.mask.shape[0]))
                break
            model_input = model_input[0]
        if len(self.losses) >= self.stop_after_batches:
            trainer.should_stop = True

    def on_train_end(self, trainer, pl_module):
        result = {
            'state': 'passed', 'training_batches': len(self.losses),
            'optimizer_steps': trainer.global_step, 'finite_gradient_checks': self.gradient_checks,
            'batch_size_min': min(self.batch_sizes), 'batch_size_max': max(self.batch_sizes),
            'precision': str(trainer.precision), 'strategy': type(trainer.strategy).__name__,
            'peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024**3,
            'peak_reserved_gib': torch.cuda.max_memory_reserved() / 1024**3,
            'elapsed_seconds': time.monotonic() - self.started_at,
            'losses': self.losses,
        }
        (self.output_dir / 'result.json').write_text(json.dumps(result, indent=2)+'\n')
        trainer.save_checkpoint(self.output_dir / 'probe.ckpt')
