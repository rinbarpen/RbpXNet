CONFIG = {
    "project": "SegmentTaskForMedical",
    "author": "LpoutYoumu",
    # run model
    "train": False,
    "test": False,
    "predict": False,
    "wandb": False,
    # arguments
    "model": "Select a Model", # Model Name
    # for models
    "private": {
        "n_channels": 1,
        "n_classes": 1,
        "classes": [],
    },
    "backbone": {
        "vgg19": {
            
        },
        "resnet34": {

        },
    },
    # for common
    "device": "cuda",
    "memory": {
        "only_fp16": False, # for the scene -- network is lack of memory
        "amp": False, # enable mixed precision if true (fp16, fp32)
    },
    "multi-gpu": False, # enable multi-gpu to [train | test | predict] network
    # for training
    "batch_size": 1, # also for testing
    "data_dir": "", # also for testing
    "dataset": "", # also for testing
    "epochs": 1,
    "learning_rate": 3e-6,
    "augment_boost": False,
    "save_every_n_epoch": 1,
    "weight_decay": 1e-8,
    "early_exit": False,
    # "extensions": {
    #     "augment_boost": False,
    #     "save_every_n_epoch": 1,
    #     "weight_decay": 1e-8,
    #     "early_exit": False,
    # },
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
    },
    "print": False,
}

# ENVIRONMENT VARIABLES
from dotenv import load_dotenv
load_dotenv('.env')

# GLOBAL CONSTANTS
RESIZE = (512, 512)
