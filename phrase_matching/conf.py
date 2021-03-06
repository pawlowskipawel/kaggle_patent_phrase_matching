# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/conf.ipynb (unless otherwise specified).

__all__ = ['parse_cfg']

# Cell
from types import SimpleNamespace
import importlib
import argparse
import wandb
import sys


# Cell
#export
def parse_cfg():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument("-C", "--config", help="config filename")
    parser.add_argument("--config-storage", help="config storage dir", default="configs", required=False)
    
    parser.add_argument("-WB", "--wandb-log", help="wandb logging", nargs="?", const=True, default=False)
    
    parser_args, _ = parser.parse_known_args(sys.argv)
    sys.path.append(parser_args.config_storage)
    
    print("Using config file", parser_args.config)

    args = importlib.import_module(parser_args.config).args
    
    if parser_args.wandb_log:
        config = {k: v for k, v in args.items() if k not in ["metrics_dict", "device", "save_path"]}
        project = "patent_phrase_matching"
        
        wandb.init(project=project, config=config)
        wandb.define_metric("train/step")
        wandb.define_metric("valid/step")
        
        wandb.define_metric("train/*", step_metric="train/step")
        wandb.define_metric("valid/*", step_metric="valid/step")
        
    return SimpleNamespace(**args), parser_args.wandb_log