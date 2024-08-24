import torch
from tqdm import tqdm
from utils.metrics.dice_score import dice_loss, multiclass_dice_coeff, dice_coeff


def valid_one_epoch(model, device, epoch, valid_loader, criterion, n_classes, average):
  """
  Validates the model for one epoch.

  Parameters:
  - model: The trained model to validate.
  - device: The device (CPU or GPU) to perform computations on.
  - epoch: The current epoch number.
  - valid_loader: The data loader for the validation dataset.
  - criterion: The loss function to calculate the loss.
  - n_classes: The number of classes in the dataset.
  - average: A flag indicating whether to average the loss over all classes.

  Returns:
  - val_loss: The average loss for the validation dataset.
  """
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
