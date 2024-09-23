import torch
from tqdm import tqdm, trange

@torch.no_grad
def valid_one_epoch(net, device, epoch, valid_loader, criterion):
    net.eval()
    valid_loss = 0.0

    with tqdm(total=len(valid_loader), desc=f'Validating') as pbar:
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device, dtype=torch.float32), labels.to(device, dtype=torch.float32)

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()

            pbar.update()
            pbar.set_postfix(**{'loss(batch)': loss.item()})

    valid_loss /= len(valid_loader)

    return valid_loss


def validate(net, valid_loader, device, epochs, criterion):

    valid_loss = 0.0
    for epoch in trange(epochs):
        valid_loss += valid_one_epoch(net, device, epoch, valid_loader, criterion)
    valid_loss /= epochs

    return valid_loss

