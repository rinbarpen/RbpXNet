import logging
import os
import sys
from datetime import datetime

import torch
from torch import nn
from torch.cuda.amp.grad_scaler import GradScaler
from torch.optim.adam import Adam
from torch.optim.adamw import AdamW
import numpy as np
import random

from pathlib import Path

from matplotlib import pyplot as plt

import config
from config import CONFIG, USE_SEED
from options import parse_args
from ml_operations import Trainer, Tester, Predictor
from typing import Optional
from utils.datasets.dataset import get_train_valid_and_test_loader
from utils.metrics.Criterion import CombinedLoss
from utils.utils import create_file, print_model_info, summary_model_info, list2tuple
from models.models import select_model


LOG_LEVEL = logging.INFO
LOG_FORMAT = (
    "%(asctime)s %(name)s [%(levelname)s] %(filename)s:%(lineno)s | %(message)s"
)


def prepare_logging():
    log_file = os.path.join("logs", CONFIG["author"], CONFIG["project"], 
                            CONFIG['model'] + '-' + CONFIG['dataset'] + ".log")

    create_file(log_file)
    logging.basicConfig(
        level=LOG_LEVEL,
        format=LOG_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8", delay=True),
        ],
    )


def set_seed(seed: Optional[int] = None):
    if not seed:
        seed = CONFIG["seed"] = torch.seed()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    parse_args()

    prepare_logging()

    # plt.ion()

    if CONFIG["print"]:
        model_path = CONFIG["load"]
        print("Model Info: ")
        print_model_info(model_path, sys.stdout)
        # print("Model Summary: ")
        # summary_model_info(model_path, (1, 1, 512, 512))
        sys.exit(0)

    # if USE_SEED:
    #     set_seed(CONFIG["seed"])
    # else:
    #     set_seed()

    net = select_model(
        CONFIG["model"],
        n_channels=CONFIG["private"]["n_channels"],
        n_classes=CONFIG["private"]["n_classes"],
        use_bilinear=False,
    )

    # TODO: enable multi-gpu to train
    # if CONFIG['multi-gpu']:
    #     net = torch.nn.DataParallel(net)

    classes = CONFIG["private"]["classes"]
    device = CONFIG["device"]

    # predict with the model
    if CONFIG["predict"]:
        if os.path.isdir(CONFIG["input"]):
            input_dir = Path(CONFIG["input"])
            inputs = [
                input_dir / input
                for input in input_dir.glob("*.*")
                if input.suffix in [".png", ".jpg", ".jpeg"]
            ]
        else:
            inputs = [CONFIG["input"]]

        predictor = Predictor(net, classes=list2tuple(classes), device=device)
        predictor.predict(inputs)
        sys.exit(0)

    if CONFIG["train"] or CONFIG["test"]:
        # get my datium for next work
        data_dir = CONFIG["data_dir"] + CONFIG["dataset"]
        train_loader, valid_loader, test_loader = get_train_valid_and_test_loader(
            CONFIG["dataset"],
            data_dir,
            batch_size=CONFIG["batch_size"],
            train_valid_test=(0.9, 0, 0.1),
            use_augment_enhance=CONFIG["augment_boost"],
            resize=config.RESIZE,
            num_workers=0,
        )  # type: ignore
        # test my model
        if CONFIG["test"]:
            metrics = ["mIoU", "accuracy", "f1", "f2", "recall", "precision", "dice"]
            classes = ["background", *classes]
            tester = Tester(net, loader=test_loader, classes=list2tuple(classes), device=device)
            tester.test(selected_metrics=list2tuple(metrics))
            sys.exit(0)

        # train my model
        if CONFIG["train"]:
            # optimizer = AdamW(net.parameters(), lr=CONFIG['learning_rate'],
            #                 betas=(0.9, 0.999), eps=1e-8,
            #                 weight_decay=CONFIG['weight_decay'], amsgrad=True)
            optimizer = Adam(
                net.parameters(),
                lr=CONFIG["learning_rate"],
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=CONFIG["weight_decay"],
                amsgrad=True,
            )
            criterion = (
                nn.BCEWithLogitsLoss() if len(classes) == 1 else nn.CrossEntropyLoss()
            )
            # criterion = CombinedLoss(0.7, 0.3)
            scaler = GradScaler() if CONFIG["memory"]["amp"] else None
            trainer = Trainer(
                net,
                optimizer=optimizer,
                criterion=criterion,
                scaler=scaler,
                classes=classes,
                train_loader=train_loader,
                device=device,
            )
            trainer.train()
            sys.exit(0)
