import logging
import numpy as np
import os
from PIL import Image

from tqdm import tqdm, trange

import torch
from torch.cuda.amp.autocast_mode import autocast

from typing import Tuple, List, Literal

from utils.utils import (
    create_dirs,
    load_model,
    file_prefix_name,
    save_model,
    load_model,
    list2tuple,
    tuple2list,
)
from utils.metrics.metrics import MetricRecorder, MetricViewer
from utils.Recorder import Recorder
from utils.visualization import draw_attention_heat_graph, draw_heat_graph
from utils.metrics.scores import (
    f1_score,
    f_score,
    dice_score,
    iou_score,
    accuracy_score,
    precision_score,
    recall_score,
    calcuate_scores,
)


class Trainer:
    def __init__(
        self,
        net,
        optimizer,
        criterion,
        classes,
        train_loader,
        scaler=None,
        valid_loader=None,
        device=None,
    ):
        from config import CONFIG

        self.device = device if device else CONFIG["device"]
        self.net = net.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scaler = scaler

        self.train_loader = train_loader
        self.validator = (
            Validator(
                net, criterion=criterion, valid_loader=valid_loader, device=device
            )
            if valid_loader and len(valid_loader) > 0
            else None
        )
        self.classes = classes

        assert len(self.classes) >= 1, "The number of classes should be greater than 0"

    def train_one_epoch(self):
        self.net.train()
        train_loss = 0.0

        with tqdm(total=len(self.train_loader), desc=f"Training") as pbar:
            for inputs, targets in self.train_loader:
                self.optimizer.zero_grad()
                inputs, targets = inputs.to(
                    self.device, dtype=torch.float32
                ), targets.to(self.device, dtype=torch.float32)

                if self.scaler:
                    with autocast():
                        outputs = self.net(inputs)
                        loss = self.criterion(
                            outputs, targets
                        )  # + dice_loss(targets, outputs, 2)

                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.net(inputs)
                    loss = self.criterion(
                        outputs, targets
                    )  # + dice_loss(targets, outputs, 2)
                    loss.backward()
                    self.optimizer.step()

                train_loss += loss.item()

                pbar.update()
                pbar.set_postfix(**{"loss(batch)": loss.item()})

        train_loss /= len(self.train_loader)

        return train_loss

    def train_model(self):
        from config import CONFIG

        best_train_loss = float("inf")
        best_valid_loss = float("inf")

        epochs = CONFIG["epochs"]
        train_losses = np.zeros(epochs)
        valid_losses = np.zeros(epochs)
        for epoch in trange(epochs, desc="Epoch: "):
            train_loss = self.train_one_epoch()

            if self.validator:
                valid_loss = self.validator.valid_one_epoch()

                logging.info(
                    f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, "
                    f"Validation Loss: {valid_loss:.4f}"
                )
                valid_losses[epoch] = valid_loss
            else:
                logging.info(
                    f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}"
                )

            train_losses[epoch] = train_loss

            if (epoch + 1) % CONFIG["save_every_n_epoch"] == 0:
                save_model_dir = CONFIG["save"]["model_dir"]
                save_model_filename = os.path.join(
                    save_model_dir,
                    f'{CONFIG["model"]}-{epoch + 1}of{epochs}-{CONFIG["dataset"]}.pth',
                )
                save_model(save_model_filename, self.net)
                logging.info(
                    f"save model to {save_model_filename} "
                    f"when {epoch=}, {train_loss=}"
                )
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                best_model_filename = CONFIG["save"]["model"]
                save_model(best_model_filename, self.net, optimizer=self.optimizer, scaler=self.scaler)
                logging.info(
                    f"save model to {file_prefix_name(best_model_filename)} "
                    f"when {epoch=}, {train_loss=}"
                )
            # if self.validator:
            #     if valid_loss < best_valid_loss:
            #         best_valid_loss = valid_loss
            #         best_model_filename = CONFIG["save"]["model"]
            #         save_model(best_model_filename, self.net, optimizer=self.optimizer, scaler=self.scaler)
            #         logging.info(f'save model to {file_prefix_name(best_model_filename)} '
            #                     f'when {epoch=}, {valid_loss=}')

        return train_losses, valid_losses

    def train(self):
        train_losses, valid_losses = self.train_model()

        use_validate = self.validator is not None

        Recorder.record_train(Recorder(), train_loss={"loss": train_losses, "step": 1})

        if use_validate:
            Recorder.record_valid(Recorder(), valid_loss={"loss": valid_losses})


class Validator:
    def __init__(self, net, criterion, valid_loader, device=None):
        from config import CONFIG

        self.device = device if device else CONFIG["device"]
        self.net = net.to(self.device)
        self.criterion = criterion
        self.valid_loader = valid_loader

    @torch.no_grad()
    def valid_one_epoch(self):
        self.net.eval()
        valid_loss = 0.0

        with tqdm(total=len(self.valid_loader), desc=f"Validating") as pbar:
            for inputs, labels in self.valid_loader:
                inputs, labels = inputs.to(self.device, dtype=torch.float32), labels.to(
                    self.device, dtype=torch.float32
                )

                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                valid_loss += loss.item()

                pbar.update()
                pbar.set_postfix(**{"loss(batch)": loss.item()})

        valid_loss /= len(self.valid_loader)

        return valid_loss

    def validate(self, epochs):
        valid_loss = 0.0
        for epoch in trange(epochs):
            valid_loss += self.valid_one_epoch()
        valid_loss /= epochs

        return valid_loss


class Tester:
    def __init__(self, net, loader, classes: Tuple[str, ...], device=None):
        from config import CONFIG

        self.device = device if device else CONFIG["device"]
        self.net = net.to(self.device)
        self.loader = loader
        self.classes = classes

        assert len(self.classes) >= 1, "The number of classes should be greater than 0"

        model_filename = CONFIG["save"]["model"]
        logging.info(
            f"Loading model: {os.path.abspath(model_filename)} on {self.device}"
        )

        self.net.load_state_dict(load_model(model_filename, self.device)["model"])

    @torch.no_grad()
    def test_model(self, selected_metrics: Tuple[str, ...] = ("dice", "f1", "recall", "miou", "precision", "accuracy")):
        self.net.eval()

        # all_metrics = dict()
        # mean_metrics = dict()
        metric_recorder = MetricRecorder()
        with tqdm(total=len(self.loader), desc=f"Testing") as pbar:
            for inputs, targets in self.loader:
                inputs, targets = inputs.to(
                    self.device, dtype=torch.float32
                ), targets.to(self.device, dtype=torch.float32)

                preds = self.net(inputs)

                preds[preds >= 0.5] = 1
                preds[preds < 0.5] = 0

                metrics = calcuate_scores(targets, preds, self.classes)
                for name, value in metrics.items():
                    metric_recorder.record(name, value)

                pbar.update()
                pbar.set_postfix(**{"mPA": np.mean(metrics["accuracy"])})

        viewer = metric_recorder.filter(selected_metrics).view()
        return {
            'all': viewer.all_metrics('all'),
            'mean': viewer.all_metrics('mean'),
        }

    def test(self, selected_metrics: Tuple[str, ...]):
        metrics = self.test_model(selected_metrics=selected_metrics)
        from pprint import pprint
        pprint(metrics['all'])
        pprint(metrics['mean'])

        Recorder.record_test(Recorder(), metric={
            'all': metrics['all'],
            'mean': {name: value['mean'] for name, value in metrics['mean'].items()}
        })
        # Recorder.record_test(Recorder(), metric=all_metrics['vein'])


class Predictor:
    def __init__(self, net, classes: Tuple[str, ...], device=None):
        from config import CONFIG

        self.device = device if device else CONFIG["device"]
        self.net = net.to(self.device)
        self.classes = classes

        assert len(self.classes) >= 1, "The number of classes should be greater than 0"

        self.net.load_state_dict(
            load_model(CONFIG["load"], device=self.device)["model"]
        )

    @torch.inference_mode()
    def predict_one(self, input):
        # input = Image.open(input).convert('RGB')
        # original_size = input.size # (W, H)
        # input = input.resize((512, 512))  # TODO: to be more flexible
        # input = torch.from_numpy(np.array(input).transpose(2, 0, 1))
        # input = input.unsqueeze(0) # (1, 3, 512, 512)

        input = Image.open(input).convert("L")
        original_size = input.size
        input = input.resize((512, 512))
        input = torch.from_numpy(np.array(input))
        input = input.unsqueeze(0).unsqueeze(0)  # (1, 1, 512, 512)

        input = input.to(self.device, dtype=torch.float32)

        self.net.eval()

        predict = self.net(input)  # (1, N, H, W)
        # predict = F.sigmoid(predict)
        # predict = (predict - predict.min()) / (predict.max() - predict.min())
        # predict = predict.squeeze(0)
        predict = predict.squeeze(0)
        possibilities = predict.cpu().detach().numpy()  # (N, H, W)
        predict_image = possibilities.copy()  # (N, H, W)

        predict_image[predict_image >= 0.5] = 255
        predict_image[predict_image < 0.5] = 0
        predict_image = predict_image.astype(np.uint8)

        assert predict_image.shape[0] == len(
            self.classes
        ), "The number of classes should be equal to the number of ones predicted"
        result = dict()
        for i, name in enumerate(self.classes):
            possibility = possibilities[i]
            mask = Image.fromarray(predict_image[i], mode="L").resize(
                size=original_size
            )
            result[name] = {
                "possibility": possibility,
                "mask": mask,
            }
        return result

    def predict_more(self, inputs):
        results = dict()
        for input in inputs:
            result = self.predict_one(input)
            results[input.absolute()] = result

        return results

    def predict(self, inputs):
        from config import CONFIG

        for input in inputs:
            result = self.predict_one(input)

            predict_dir = CONFIG["save"]["predict_dir"]
            filename = input.stem
            predict_filename = f"{filename}_predict.png"
            heat_filename = f"{filename}_heat.png"
            fuse_filename = f"{filename}_fuse.png"
            for category, values in result.items():
                category_dir = os.path.join(predict_dir, category)
                predict_path = os.path.join(category_dir, predict_filename)
                heat_path = os.path.join(category_dir, heat_filename)
                fuse_path = os.path.join(category_dir, fuse_filename)
                title = filename

                create_dirs(category_dir)

                possibility = values["possibility"]
                mask_image = values["mask"]
                mask_image.save(predict_path)
                original_image = Image.open(input).convert("RGB")
                logging.info(f"Save predicted image to {os.path.abspath(predict_path)}")
                draw_heat_graph(possibility, filename=heat_path, title=title)
                logging.info(f"Save heat graph to {os.path.abspath(heat_path)}")
                draw_attention_heat_graph(
                    mask_image, original_image, filename=fuse_path, title=title
                )
                logging.info(f"Save attention image to {os.path.abspath(fuse_path)}")

        # results = self.predict_more(inputs)
        # for path, result in results:
        #     predict_dir = CONFIG["save"]["predict_dir"]
        #     filename = file_prefix_name(path)
        #     predict_filename = f"{filename}_predict.png"
        #     heat_filename = f"{filename}_heat.png"
        #     fuse_filename = f"{filename}_fuse.png"
        #     for category, values in result.items():
        #         category_dir = os.path.join(predict_dir, category)
        #         predict_path = os.path.join(category_dir, predict_filename)
        #         heat_path = os.path.join(category_dir, heat_filename)
        #         fuse_path = os.path.join(category_dir, fuse_filename)
        #         title = filename

        #         create_dirs(category_dir)

        #         possibility = values["possibility"]
        #         mask_image = values["mask"]
        #         mask_image.save(predict_path)
        #         original_image = Image.open(input).convert('RGB')
        #         logging.info(f"Save predicted image to {os.path.abspath(predict_path)}")
        #         draw_heat_graph(possibility,
        #                         filename=heat_path,
        #                         title=title)
        #         logging.info(f"Save heat graph to {os.path.abspath(heat_path)}")
        #         draw_attention_heat_graph(mask_image,
        #                                 original_image,
        #                                 filename=fuse_path,
        #                                 title=title)
        #         logging.info(f"Save attention image to {os.path.abspath(fuse_path)}")
