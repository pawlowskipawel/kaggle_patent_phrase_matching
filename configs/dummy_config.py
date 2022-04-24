from phrase_matching.metrics import PearsonCorrCoef
#no sigmoid in model
args = {
    # global
    "TRAIN_DATA_PATH": "data/stratified_train_5fold_regression.csv",
    "device": "cuda",
    "fold_to_run": 1,
    "task": "regression",
    "save_path": "checkpoints",
    
    # model
    "model_name": "microsoft/deberta-v3-large",
    "embedding_dim": 768,
    "pool_backbone_output": True, 
    "backbone_pooler_dropout": .1,
    
    # data
    "max_len": 256,
    "lowercase": True,
    "allow_dynamic_padding": True,
    
    # training
    "epochs": None,
    "batch_size": None,
    "grad_accum_iter": None,
    
    # optimizer
    "optimizer_name": None,
    "learning_rate": None,
    "weight_decay": None,
    
    # lr scheduler
    "lr_scheduler": None,
    "scheduler_warmup_epochs": None,
    
    # onecycle_lr
    "max_learning_rate": None,
    "div_factor": None,
    "final_div_factor": None
}

args["metrics_dict"] = {"pearson_corr": PearsonCorrCoef()}