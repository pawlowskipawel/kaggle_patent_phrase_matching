from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import pandas as pd
import numpy as np

import torch.nn as nn
import torch

from phrase_matching.training import Trainer, get_optimizer, get_scheduler
from phrase_matching.models import PhraseModel
from phrase_matching.data import PhraseDataset
from phrase_matching.conf import parse_cfg

import random
import os
import gc

def seed_everything(seed=7777):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    

if __name__ == "__main__":
    seed_everything()
    cfg = parse_cfg()
           
    df = pd.read_csv(cfg.TRAIN_DATA_PATH)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    metrics_dict = None if not hasattr(cfg, "metrics_dict") else cfg.metrics_dict
    
    for fold_i in range(len(df["fold"].unique())): 
        if cfg.fold_to_run is not None:
            fold_i = cfg.fold_to_run
        
        print(f"---- FOLD {fold_i} ----")
        
        train_df = df[df["fold"] != fold_i].drop("fold", axis=1)
        valid_df = df[df["fold"] == fold_i].drop("fold", axis=1)
        
        train_dataset = PhraseDataset(
            dataset_df=train_df, 
            tokenizer=tokenizer,
            max_len=cfg.max_len,
            lowercase=cfg.lowercase,
            mode="train_val",
            task=cfg.task
        )
        
        valid_dataset = PhraseDataset(
            dataset_df=valid_df, 
            tokenizer=tokenizer,
            max_len=cfg.max_len,
            lowercase=cfg.lowercase,
            mode="train_val", 
            task=cfg.task
        )
        
        train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, pin_memory=True, num_workers=16, shuffle=True, drop_last=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=cfg.batch_size, pin_memory=True, num_workers=16, shuffle=False)
        
        model = PhraseModel(
            model_name=cfg.model_name, 
            embedding_dim=cfg.embedding_dim,
            task=cfg.task,
            pool_backbone_output=cfg.pool_backbone_output,
            backbone_pooler_dropout=cfg.backbone_pooler_dropout
        )
        model.to(cfg.device)
        
        optimizer = get_optimizer(
            optimizer_name=cfg.optimizer_name,
            model=model,
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay
        )
        
        criterion = nn.CrossEntropyLoss() if cfg.task == "classification" else nn.MSELoss()
        
        lr_scheduler = get_scheduler(cfg, optimizer, len(train_dataloader))
            
        trainer = Trainer(
            model_name=cfg.model_name.split("/")[-1],
            model=model, 
            criterion=criterion, 
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            metrics_dict=metrics_dict,
            task=cfg.task,
            device=cfg.device,
            allow_dynamic_padding=cfg.allow_dynamic_padding,
            grad_accum_iter=cfg.grad_accum_iter
        )
        
        trainer.fit(
            epochs=cfg.epochs,
            train_dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
            save_path=cfg.save_path,
            fold_i=fold_i
        )
        
        del model, criterion, optimizer, lr_scheduler, trainer
        torch.cuda.empty_cache()
        gc.collect()
        
        if cfg.fold_to_run is not None:
            break