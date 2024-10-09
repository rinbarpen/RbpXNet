import os.path
from tqdm import tqdm

from evaluate import *
from torch.cuda.amp.autocast_mode import autocast
from utils.Recorder import Recorder
from utils.utils import *
from utils.visualization import *

class Trainer:
    def __init__(self, net, optimizer, criterion, classes, train_loader, scaler=None, valid_loader=None, device=None):
        from config import CONFIG

        self.device = device if device else CONFIG['device']
        self.net = net.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scaler = scaler
        
        self.train_loader = train_loader
        self.validator = Validator(net, criterion=criterion, valid_loader=valid_loader, device=device) if valid_loader and len(valid_loader) > 0 else None
        self.classes = classes
        
        assert len(self.classes) >= 1, 'The number of classes should be greater than 0'

    def train_one_epoch(self):
        self.net.train()
        train_loss = 0.0

        with tqdm(total=len(self.train_loader), desc=f'Training') as pbar:
            for inputs, targets in self.train_loader:
                self.optimizer.zero_grad()
                inputs, targets = inputs.to(self.device, dtype=torch.float32), targets.to(self.device, dtype=torch.float32)

                if self.scaler:
                    with autocast():
                        outputs = self.net(inputs)
                        loss = self.criterion(outputs, targets) # + dice_loss(targets, outputs, 2)
                        
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.net(inputs)
                    loss = self.criterion(outputs, targets) # + dice_loss(targets, outputs, 2)
                    loss.backward()
                    self.optimizer.step()

                train_loss += loss.item()

                pbar.update()
                pbar.set_postfix(**{'loss(batch)': loss.item()})

        train_loss /= len(self.train_loader)

        return train_loss


    def train_model(self):
        from config import CONFIG

        best_train_loss = float('inf')
        best_valid_loss = float('inf')

        epochs = CONFIG['epochs']
        train_losses = np.zeros(epochs)
        valid_losses = np.zeros(epochs)
        for epoch in trange(epochs, desc='Epoch: '):
            train_loss = self.train_one_epoch()
            
            if self.validator:
                valid_loss = self.validator.valid_one_epoch()

                logging.info(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, "
                            f"Validation Loss: {valid_loss:.4f}")
                valid_losses[epoch] = valid_loss
            else:
                logging.info(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}")

            train_losses[epoch] = train_loss

            if (epoch + 1) % CONFIG['save_every_n_epoch'] == 0:
                save_model_dir = CONFIG["save"]["model_dir"]
                save_model_filename = os.path.join(save_model_dir, f'{CONFIG["model"]}-{epoch + 1}of{epochs}-{CONFIG["dataset"]}.pth')
                save_model(save_model_filename, self.net)
                logging.info(f'save model to {save_model_filename} '
                            f'when {epoch=}, {train_loss=}')
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                best_model_filename = CONFIG["save"]["model"]
                save_model(best_model_filename, self.net)
                logging.info(f'save model to {file_prefix_name(best_model_filename)} '
                            f'when {epoch=}, {train_loss=}')
            # if self.validator:
            #     if valid_loss < best_valid_loss:
            #         best_valid_loss = valid_loss
            #         best_model_filename = CONFIG["save"]["model"]
            #         save_model(best_model_filename, self.net)
            #         logging.info(f'save model to {file_prefix_name(best_model_filename)} '
            #                     f'when {epoch=}, {valid_loss=}')

        return train_losses, valid_losses


    def train(self):
        train_losses, valid_losses = self.train_model()

        use_validate = (self.validator is not None)

        Recorder.record_train(Recorder(), train_loss={'loss': train_losses, 'step': 1})

        if use_validate:
            Recorder.record_valid(Recorder(), valid_loss={'loss': valid_losses})
