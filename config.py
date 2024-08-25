CONFIG = {
    "project": "Fa",
    "entity": "LpoutYoumu",
    # run model
    "train": False,
    "test": False,
    "predict": False,
    "wandb": False,
    # arguments
    "model": None,
    # for models
    "private": {
        "in_channels": 3,
        "n_classes": 1,
        "classes": [],
    },
    # for common
    "device": "cuda",
    "amp": False,
    # for training
    "batch_size": 1, # also for testing
    "data_dir": "", # also for testing
    "dataset": "", # also for testing
    "epochs": 1,
    "learning_rate": 3e-6,
    "augment_boost": False,
    "save_n_epoch": 1,
    # for predicting
    "load": "", # load model file
    "input": "", # things to be predicted

    "save": {
        "predict_dir": "output/predict/",
        "train_dir": "output/train/",
        "valid_dir": "output/valid/",
        "test_dir": "output/test/",
        "model_dir": "output/model/",
        "model": "output/best_model.pth",
    }
}

from dotenv import load_dotenv
load_dotenv('.env')

import os
os.getenv('')
# env = os.getenv('XXX-XXX')
