from transformers import Trainer, TrainingArguments
import torch
import json
import wandb
import os

from torch.distributed import barrier

class CustomTrainer(Trainer):
    def __init__(self, cfg, cloze_task_fn, eval_query_dataset_path, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # wandb.config.update(dict(cfg))
        self.cloze_task_fn = cloze_task_fn  # A function to evaluate cloze task
        self.cfg = cfg

        self.eval_query_dataset_path = eval_query_dataset_path
        with open(eval_query_dataset_path, 'r') as f:
            eval_query_dataset = json.load(f)['samples']

            # take first 1000 samples
            eval_query_dataset = eval_query_dataset[:1000]

        self.eval_query_dataset = eval_query_dataset
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        # Call the default validation set evaluation
        self.model.eval()
        
        metrics = super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
        
        # Now handle the cloze task
        if self.is_world_process_zero():
            cloze_metrics = self.evaluate_cloze_task()
        
            # Merge validation set metrics and cloze task metrics
            metrics.update(cloze_metrics)
            
            # Log all metrics (validation and cloze)
            self.log(metrics)
            wandb.log(cloze_metrics)
        barrier()

        self.model.train()
        
        return metrics
    
    def evaluate_cloze_task(self):
        """Custom evaluation logic for the cloze task"""
        # Assume `self.cloze_task_fn()` returns a dictionary of cloze task metrics
        cloze_metrics = self.cloze_task_fn(self.model, self.tokenizer, self.eval_query_dataset, batch_size=100)
        
        return cloze_metrics
    
    def log_config(self):
        # Log the configuration to W&B
        wandb.config.update(self.cfg)