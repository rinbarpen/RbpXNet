import json
import logging
import shutil
import os
import toml

from config import CONFIG
from typing import Type, Literal, Tuple
from utils.utils import create_dirs, save_model
from utils.writer import CSVWriter
from utils.visualization import draw_loss_graph, draw_metrics_graph

class Recorder:
    _instance = None
    def __init__(self):
        hash_code = str(hash(CONFIG['project'] + CONFIG['entity']))
        self.dst_dir = os.path.join('output', hash_code)

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls

    def record_train(self, **kwargs):
        train_dir = os.path.join(self.dst_dir, 'train')
        create_dirs(train_dir)

        loss_dict = kwargs['train_loss']
        if loss_dict:
            # train_loss.csv
            train_loss_csv = os.path.join(train_dir, 'train_loss.csv')
            writer = CSVWriter(train_loss_csv)
            writer.write('loss', loss_dict['loss'])

            # train_loss.png
            train_loss_graph = os.path.join(train_dir, 'train_loss.png')
            draw_loss_graph(loss_dict['loss'], 5, train_loss_graph)

            logging.info(f"save training loss data to {os.path.abspath(train_loss_csv)}, "
                        f"draw to {os.path.abspath(train_loss_graph)}")

    def record_test(self, **kwargs):
        test_dir = os.path.join(self.dst_dir, 'test')
        create_dirs(test_dir)

        loss_dict = kwargs['test_loss']
        if loss_dict:
            # test_loss.csv
            test_loss_csv = os.path.join(test_dir, 'test_loss.csv')
            writer = CSVWriter(test_loss_csv)
            writer.write('loss', loss_dict['loss']).flush()

            # test_loss.png
            test_loss_graph = os.path.join(test_dir, 'test_loss.png')
            draw_loss_graph(loss_dict['loss'], 5, test_loss_graph)

            logging.info(f"save testing loss data to {os.path.abspath(test_loss_csv)}, "
                        f"draw to {os.path.abspath(test_loss_graph)}")

        metric_dict = kwargs['metric']
        if metric_dict:
            # metric_dict: {
            #    "f1":        f1_score,
            #    "f2":        f2_score,
            #    "dice":      dice_score,
            #    "miou":      miou_score,
            #    "accuracy":  accuracy_score,
            #    "recall":    recall_score,
            #    "precision": precision_score,
            # }
            test_metric_csv = os.path.join(test_dir, 'metrics.csv')
            writer = CSVWriter(filename=test_metric_csv)
            writer.writes(metric_dict).flush()

            colors = ['purple', 'red', 'green', 'yellow', 'blue', 'brown', 'cyan']
            test_metric_graph =  os.path.join(test_dir, 'metrics.png')
            draw_metrics_graph(metric_dict, colors=colors, filename=test_metric_graph, title='Metrics')

            logging.info(f"save testing metric data to {os.path.abspath(test_metric_csv)}, "
                        f"draw to {os.path.abspath(test_metric_graph)}")

    def record_valid(self, **kwargs):
        valid_dir = os.path.join(self.dst_dir, 'valid')
        create_dirs(valid_dir)

        loss_dict = kwargs['valid_loss']
        if loss_dict:
            # valid_loss.csv
            valid_loss_csv = os.path.join(valid_dir, 'valid_loss.csv')
            writer = CSVWriter(valid_loss_csv)
            writer.write('loss', loss_dict['loss'])

            # valid_loss.png
            valid_loss_graph = os.path.join(valid_dir, 'valid_loss.png')
            draw_loss_graph(loss_dict['loss'], 5, valid_loss_graph)

            logging.info(f"save validating loss data to {os.path.abspath(valid_loss_csv)}, "
                        f"draw to {os.path.abspath(valid_loss_graph)}")
    def record_model(self, **kwargs):
        model_dir = os.path.join(self.dst_dir, 'model')
        create_dirs(model_dir)

        model_dict = kwargs['model']
        if model_dict:
            # model
            model_filepath = model_dict['filename']
            save_model(model_filepath,
                       model=model_dict['model'], optimizer=model_dict['optimizer'],
                       lr_scheduler=model_dict['lr_scheduler'], scaler=model_dict['scaler'])
            logging.info(f"save validating loss data to {os.path.abspath(model_filepath)}")

    @staticmethod
    def move(src_dir, dst_dir):
        shutil.move(src_dir, dst_dir)

    def backup(self, backup_dir):
        shutil.copytree(self.dst_dir, backup_dir)

    def dump(self, file_type: Literal['toml', 'json']='toml'):
        match file_type:
            case 'toml': self.dump_toml()
            case 'json': self.dump_json()

    def dump_toml(self):
        with open(os.path.join(self.dst_dir, "config.toml"), "w") as f:
            toml.dump(CONFIG, f)

    def dump_json(self):
        with open(os.path.join(self.dst_dir, "config.json"), "w") as f:
            json.dump(CONFIG, f, indent=4)
