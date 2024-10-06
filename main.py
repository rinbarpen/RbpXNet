import logging
import os
import sys
from datetime import datetime

import torch
from torch.cuda.amp.grad_scaler import GradScaler
import numpy as np
import random

from pathlib import Path

from matplotlib import pyplot as plt

import config
from config import CONFIG, USE_SEED
from options import parse_args
from train import Trainer
from test import Tester
from predict import Predictor
from typing import Optional
from utils.datasets.dataset import get_train_valid_and_test_loader
from utils.metrics.Criterion import CombinedLoss
from utils.utils import create_file, print_model_info, summary_model_info
from models.models import select_model

LOG_LEVEL = logging.INFO
LOG_FORMAT = (
    "%(asctime)s %(name)s [%(levelname)s] %(filename)s:%(lineno)s | %(message)s"
)
LOG_FILENAME = f"logs/{datetime.now().strftime('%Y-%m-%d %H_%M_%S')}.log"

create_file(LOG_FILENAME)
logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILENAME, encoding="utf-8"),
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

    plt.ion()

    if CONFIG["print"]:
        model_path = CONFIG["load"]
        print("Model Info: ")
        print_model_info(model_path, sys.stdout)
        print("Model Summary: ")
        summary_model_info(model_path, (1, 1, 512, 512))
        sys.exit(0)

    if USE_SEED:
        set_seed(CONFIG["seed"])
    else:
        set_seed()

    net = select_model(
        CONFIG["model"],
        n_channels=CONFIG["private"]["n_channels"],
        n_classes=CONFIG["private"]["n_classes"],
        use_bilinear=True,
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
            inputs = [input_dir / input for input in input_dir.glob("*.{png,jpg,jpeg}")]
        else:
            inputs = [CONFIG["input"]]

        predictor = Predictor(net, classes=classes, device=device)
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
            num_workers=4,
        )  # type: ignore
        # test my model
        if CONFIG["test"]:
            metrics = ["mIoU", "accuracy", "f1", "f2", "recall", "dice"]
            tester = Tester(net, loader=test_loader, classes=classes, device=device)
            tester.test(selected_metrics=metrics)
            sys.exit(0)

        # train my model
        if CONFIG["train"]:
            # optimizer = torch.optim.RMSprop(
            #     net.parameters(),
            #     lr=CONFIG["learning_rate"],
            #     eps=1e-8,
            #     weight_decay=CONFIG["weight_decay"],
            # )
            # optimizer = torch.optim.AdamW(net.parameters(), lr=CONFIG['learning_rate'],
            #                 betas=(0.9, 0.999), eps=1e-8,
            #                 weight_decay=CONFIG['weight_decay'], amsgrad=True)
            optimizer = torch.optim.Adam(
                net.parameters(),
                lr=CONFIG["learning_rate"],
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=CONFIG["weight_decay"],
                amsgrad=True,
            )
            # criterion = nn.BCEWithLogitsLoss() if len(classes) == 1 else nn.CrossEntropyLoss()
            criterion = CombinedLoss(0.7, 0.3)
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
