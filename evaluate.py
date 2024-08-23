import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.loss.dice_score import multiclass_dice_coeff, dice_coeff


def valid_one_epoch(model, device, epoch, valid_loader, criterion, n_classes, average):
  model.eval()
  val_loss = 0.0

  with tqdm(total=len(valid_loader), desc=f'Validating') as pbar:
    with torch.no_grad():
      for inputs, labels in valid_loader:
        inputs, labels = inputs.to(device, dtype=torch.float32), labels.to(device, dtype=torch.float32)
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        val_loss += loss.item()

        pbar.update()
        pbar.set_postfix(**{'loss(batch)': loss.item()})

  val_loss /= len(valid_loader)

  return val_loss
