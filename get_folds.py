from sklearn.model_selection import KFold, StratifiedKFold
from phrase_matching.defaults import score2class

import pandas as pd
import argparse
import random
import os


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--df', type=str, default='data/train.csv')
    argparser.add_argument('--nsplits', type=int, default=5)
    argparser.add_argument('--stratify', type=str, default=None)

    return argparser.parse_args()

def seed_everything(seed=7777):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
if __name__ == '__main__':
    seed_everything()
    args = parse_args()

    df = pd.read_csv(args.df)
    df["score"] = df["score"].map(score2class)
    
    if args.stratify is None:
        kf = KFold(n_splits=args.nsplits, shuffle=True)
        folds = kf.split(df)
    else:
        kf = StratifiedKFold(n_splits=args.nsplits, shuffle=True)
        folds = kf.split(df, df[args.stratify])

    df["fold"] = -1

    for i, (_, valid_idx) in enumerate(folds):
        df.loc[valid_idx, "fold"] = i
    
    output = f"stratified_train_{args.nsplits}fold.csv" if args.stratify else f"train_{args.nsplits}fold.csv"
    
    df.to_csv(f"data/{output}", index=False)