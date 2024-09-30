import logging
import os
import sys
import time
from datetime import datetime

from pathlib import Path

from matplotlib import pyplot as plt

from options import parse_args
from train import train
from test import test
from predict import predict
from utils.datasets.dataset import get_train_valid_and_test_loader
from utils.utils import create_file, print_model_info, summary_model_info
from models.models import select_model

LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s %(name)s [%(levelname)s] %(filename)s:%(lineno)s | %(message)s"
LOG_FILENAME = f"logs/{datetime.now().strftime('%Y-%m-%d %H_%M_%S')}.log"

create_file(LOG_FILENAME)
logging.basicConfig (
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_FILENAME, encoding='utf-8')]
)

if __name__ == '__main__':
    from config import CONFIG
    import config
    parse_args()
    
    plt.ion()

    if CONFIG['print']:
        model_path = CONFIG['load']
        print("Model Info: ")
        print_model_info(model_path, sys.stdout)
        print("Model Summary: ")
        summary_model_info(model_path, (1, 1, 512, 512))
        sys.exit(0)

    net = select_model(CONFIG["model"],
                       n_channels=CONFIG["private"]["n_channels"],
                       n_classes=CONFIG["private"]["n_classes"],
                       use_bilinear=True)

    # TODO: enable multi-gpu to train
    # if CONFIG['multi-gpu']:
    #     net = torch.nn.DataParallel(net)

    # predict with the model
    if CONFIG["predict"]:
        if os.path.isdir(CONFIG["input"]):
            input_dir = Path(CONFIG["input"])
            inputs = [input_dir / input for input in input_dir.glob("*.*")
                      if input.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".gif"]]
        else:
            inputs = [CONFIG["input"]]
        predict(net, inputs, classes=CONFIG["private"]["classes"], 
                device=CONFIG["device"])
        sys.exit(0)

    if CONFIG["train"] or CONFIG["test"]:
        # get my datium for next work
        data_dir = CONFIG["data_dir"]+CONFIG["dataset"]
        train_loader, valid_loader, test_loader = \
            get_train_valid_and_test_loader(CONFIG["dataset"], data_dir,
                                            batch_size=CONFIG["batch_size"],
                                            train_valid_test=(0.9, 0, 0.1),
                                            use_augment_enhance=CONFIG["augment_boost"],
                                            resize=config.RESIZE
                                            )  # type: ignore
        # test my model
        if CONFIG["test"]:
            metrics = ['mIoU', 'accuracy', 'f1', 'recall', 'dice']
            test(net, test_loader, 
                device=CONFIG["device"], classes=CONFIG["private"]["classes"], 
                selected_metrics=metrics)
            sys.exit(0)

        # train my model
        if CONFIG["train"]:
            train(net, train_loader, valid_loader,
                device=CONFIG["device"],
                n_classes=CONFIG["private"]["n_classes"])
