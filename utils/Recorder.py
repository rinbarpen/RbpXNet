from datetime import datetime
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

import hashlib

class Recorder:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # hash_code = hashlib.md5((CONFIG['project'] 
            #                        + CONFIG['author']  
            #                        + str(datetime.now())).encode()).hexdigest()         
            # cls.dst_dir = os.path.join('output', hash_code)
            cls.dst_dir = 'output'
        return cls

    def record_train(self, **kwargs):
        train_dir = os.path.join(self.dst_dir, 'train')
        create_dirs(train_dir)

        if 'train_loss' in kwargs.keys():
            loss_dict = kwargs['train_loss']
            # train_loss.csv
            train_loss_csv = os.path.join(train_dir, 'train_loss.csv')
            writer = CSVWriter(train_loss_csv)
            writer.write('loss', loss_dict['loss'])

            # train_loss.png
            train_loss_graph = os.path.join(train_dir, 'train_loss.png')
            draw_loss_graph(loss_dict['loss'], loss_dict['step'], train_loss_graph)

            logging.info(f"save training loss data to {os.path.abspath(train_loss_csv)}, "
                         f"draw to {os.path.abspath(train_loss_graph)}")

        if CONFIG['wandb']:
            import wandb
            if loss_dict:
                wandb.log({'train_losses': os.path.abspath(train_loss_csv), 'train_loss_image': os.path.abspath(train_loss_graph)})

    def record_test(self, **kwargs):
        test_dir = os.path.join(self.dst_dir, 'test')
        create_dirs(test_dir)

        if 'test_loss' in kwargs.keys():
            loss_dict = kwargs['test_loss']
            # test_loss.csv
            test_loss_csv = os.path.join(test_dir, 'test_loss.csv')
            writer = CSVWriter(test_loss_csv)
            writer.write('loss', loss_dict['loss']).flush()

            # test_loss.png
            test_loss_graph = os.path.join(test_dir, 'test_loss.png')
            draw_loss_graph(loss_dict['loss'], loss_dict['step'], test_loss_graph)

            logging.info(f"save testing loss data to {os.path.abspath(test_loss_csv)}, "
                        f"draw to {os.path.abspath(test_loss_graph)}")

        if 'metric' in kwargs.keys():
            metric_dict = kwargs['metric']
            # metric_dict: {
            #    "f1":        f1_score,
            #    "f2":        f2_score,
            #    "dice":      dice_score,
            #    "miou":      miou_score,
            #    "accuracy":  accuracy_score,
            #    "recall":    recall_score,
            #    "precision": precision_score,
            # }
            all_dict = kwargs['all']
            mean_dict = kwargs['mean']

            test_metric_csv = os.path.join(test_dir, 'metrics.csv')
            writer = CSVWriter(filename=test_metric_csv)
            for name, values in all_dict.items():
                for label, value in values.items(): 
                    writer.write(f'{name}_{label}', value)
            writer.flush()

            colors = ['purple', 'red', 'green', 'yellow', 'blue', 'brown', 'cyan']
            test_metric_graph =  os.path.join(test_dir, 'metrics.png')
            draw_metrics_graph(mean_dict, colors=colors, filename=test_metric_graph, title='Metrics')

            logging.info(f"save testing metric data to {os.path.abspath(test_metric_csv)}, "
                         f"draw to {os.path.abspath(test_metric_graph)}")

        if CONFIG['wandb']:
            import wandb
            wandb_config = dict()
            if loss_dict:
                wandb_config = {**wandb_config, 'test_losses': os.path.abspath(test_loss_csv), 'test_loss_image': os.path.abspath(test_loss_graph)}
            if metric_dict:
                wandb_config = {**wandb_config, 'test_metric_csv': os.path.abspath(test_metric_csv), 'test_metric_image': os.path.abspath(test_metric_graph)}
            wandb.log(wandb_config)

    def record_valid(self, **kwargs):
        valid_dir = os.path.join(self.dst_dir, 'valid')
        create_dirs(valid_dir)

        if 'valid_loss' in kwargs.keys():
            loss_dict = kwargs['valid_loss']
            # valid_loss.csv
            valid_loss_csv = os.path.join(valid_dir, 'valid_loss.csv')
            writer = CSVWriter(valid_loss_csv)
            writer.write('loss', loss_dict['loss'])

            # valid_loss.png
            valid_loss_graph = os.path.join(valid_dir, 'valid_loss.png')
            draw_loss_graph(loss_dict['loss'], loss_dict['step'], valid_loss_graph)

            logging.info(f"save validating loss data to {os.path.abspath(valid_loss_csv)}, "
                        f"draw to {os.path.abspath(valid_loss_graph)}")

        if CONFIG['wandb']:
            import wandb
            if loss_dict:
                wandb.log({'valid_losses': os.path.abspath(valid_loss_csv), 'valid_loss_image': os.path.abspath(valid_loss_graph)})

    def record_model(self, **kwargs):
        model_dir = os.path.join(self.dst_dir, 'model')
        create_dirs(model_dir)

        if 'model' in kwargs.keys():
            model_dict = kwargs['model']
            # model
            model_filepath = model_dict['filename']
            save_model(model_filepath, **model_dict, seed=CONFIG['seed'])
                    #    model=model_dict['model'], optimizer=model_dict['optimizer'],
                    #    lr_scheduler=model_dict['lr_scheduler'], scaler=model_dict['scaler'])
            logging.info(f"save validating loss data to {os.path.abspath(model_filepath)}")

        if CONFIG['wandb']:
            import wandb
            if model_dict:
                wandb.log({'model_filepath': os.path.abspath(model_filepath), 
                           **model_dict,
                           'seed': CONFIG['seed']})
                        #    'model': model_dict['model'], 
                        #    'optimizer': model_dict['optimizer'], 
                        #    'lr_scheduler': model_dict['lr_scheduler'],
                        #    'scaler': model_dict['scaler']})

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
