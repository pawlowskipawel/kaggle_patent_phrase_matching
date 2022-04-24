# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/conf.ipynb (unless otherwise specified).

__all__ = ['parse_cfg']

# Cell
from types import SimpleNamespace
import importlib
import argparse
import sys


# Cell
def parse_cfg():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument("-C", "--config", help="config filename")
    parser.add_argument("--config-storage", help="config storage dir", default="configs", required=False)

    parser_args, _ = parser.parse_known_args(sys.argv)
    sys.path.append(parser_args.config_storage)

    print("Using config file", parser_args.config)

    args = importlib.import_module(parser_args.config).args

    args["experiment_name"] = parser_args.config

    return SimpleNamespace(**args)