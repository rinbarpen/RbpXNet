import os
import sys
from pathlib import Path

from options import parse_args

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys
import time
from train import train
from test import test
from predict import predict
from utils.datasets.dataset import get_train_valid_and_test_loader

from utils.utils import *


LOG_LEVEL = logging.DEBUG
LOG_FORMAT = "%(asctime)s %(name)s [%(levelname)s] %(fileName)s:%(lineno)s %(funcName)s | %(message)s"
LOG_FILENAME = f"logs/{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}.log"

logging.basicConfig (
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler(LOG_FILENAME, encoding='utf-8')]
)

RESIZE = (512, 512)

if __name__ == '__main__':
    from config import CONFIG
    parse_args()

    create_dirs(CONFIG["save"]["train_dir"])
    create_dirs(CONFIG["save"]["valid_dir"])
    create_dirs(CONFIG["save"]["predict_dir"])
    create_dirs(CONFIG["save"]["model_dir"])

    net = select_model(CONFIG["model"],
                       in_channels=CONFIG["private"]["in_channels"],
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
        predict(net, inputs, classes=CONFIG["private"]["classes"], device=CONFIG["device"])
        sys.exit(0)

    # get my datium for next work
    data_dir = CONFIG["data_dir"]+CONFIG["dataset"]
    train_loader, valid_loader, test_loader = \
        get_train_valid_and_test_loader(CONFIG["dataset"], data_dir,
                                        batch_size=CONFIG["batch_size"],
                                        train_valid_test=[0.9, 0, 0.1],
                                        use_augment_enhance=CONFIG["augment_boost"],
                                        resize=RESIZE
                                        )  # type: ignore

    # test my model
    if CONFIG["test"]:
        metrics = ['mIoU', 'accuracy', 'f1', 'recall', 'dice']
        test(net, test_loader, device=CONFIG["device"], classes=CONFIG["private"]["classes"], selected_metrics=metrics)
        sys.exit(0)

    # train my model
    if CONFIG["train"]:
        # train my model
        train(net, train_loader, valid_loader if len(valid_loader) > 0 else None,
              device=CONFIG["device"],
              n_classes=CONFIG["private"]["n_classes"])
