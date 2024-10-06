import torch
from tqdm import tqdm, trange


class Validator:
    def __init__(self, net, criterion, valid_loader, device=None):
        from config import CONFIG

        self.device = device if device else CONFIG['device']
        self.net = net.to(self.device)
        self.criterion = criterion
        self.valid_loader = valid_loader

    @torch.no_grad()
    def valid_one_epoch(self):
        self.net.eval()
        valid_loss = 0.0

        with tqdm(total=len(self.valid_loader), desc=f'Validating') as pbar:
            for inputs, labels in self.valid_loader:
                inputs, labels = inputs.to(self.device, dtype=torch.float32), labels.to(self.device, dtype=torch.float32)

                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                valid_loss += loss.item()

                pbar.update()
                pbar.set_postfix(**{'loss(batch)': loss.item()})

        valid_loss /= len(self.valid_loader)

        return valid_loss


    def validate(self, epochs):
        valid_loss = 0.0
        for epoch in trange(epochs):
            valid_loss += self.valid_one_epoch()
        valid_loss /= epochs

        return valid_loss
    