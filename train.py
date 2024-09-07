import os.path

from torch import nn, optim

from evaluate import *
from utils.utils import *
from utils.visualization import *
from utils.writer import CSVWriter


def train_one_epoch(model, device, epoch, train_loader, optimizer, criterion):
    model.train()
    train_loss = 0.0

    with tqdm(total=len(train_loader), desc=f'Training') as pbar:
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            inputs, labels = inputs.to(device, dtype=torch.float32), labels.to(device, dtype=torch.float32)

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()
            train_loss += loss.item()

            pbar.update()
            pbar.set_postfix(**{'loss(batch)': loss.item()})

    train_loss /= len(train_loader)

    return train_loss


def train_model(model, device,
                train_loader, valid_loader,
                n_classes: int):
    from config import CONFIG
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'],
                           betas=(0.9, 0.999), eps=1e-8,
                           weight_decay=CONFIG['weight_decay'], amsgrad=False)
    criterion = nn.BCEWithLogitsLoss() if n_classes == 1 else nn.CrossEntropyLoss()

    best_train_loss = float('inf')

    epochs = CONFIG['epochs']
    train_losses = np.zeros(epochs)
    valid_losses = np.zeros(epochs)
    for epoch in trange(epochs, desc='Epoch: '):
        train_loss = train_one_epoch(model, device, epoch,
                                     train_loader=train_loader, optimizer=optimizer, criterion=criterion)

        if valid_loader:
            valid_loss = valid_one_epoch(model, device, epoch,
                                         valid_loader=valid_loader, criterion=criterion)

            logging.info(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, "
                         f"Validation Loss: {valid_loss:.4f}")
            valid_losses[epoch] = valid_loss
        else:
            logging.info(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}")

        train_losses[epoch] = train_loss

        from config import CONFIG
        if (epoch + 1) % CONFIG['save_every_n_epoch'] == 0:
            save_model_dir = CONFIG["save"]["model_dir"]
            save_model_filename = \
                f'{save_model_dir}{CONFIG["model"]}-{epoch + 1}of{epochs}-{CONFIG["dataset"]}.pth'
            save_model(model, save_model_filename)
            logging.info(f'save model to {save_model_filename} '
                         f'when {epoch=}, {train_loss=}')
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_model_filename = CONFIG["save"]["model"]
            save_model(model, best_model_filename)
            logging.info(f'save model to {file_prefix_name(best_model_filename)} '
                         f'when {epoch=}, {train_loss=}')

    return train_losses, valid_losses


def train(net, train_loader, valid_loader, device, n_classes):
    """
    Trains a neural network model using the provided training and validation data.

    Parameters:
    - net: The neural network model to be trained.
    - train_loader: A data loader for the training dataset.
    - valid_loader: A data loader for the validation dataset.
    - device: The device (CPU or GPU) to be used for training.
    - epochs: The number of training epochs.
    - learning_rate: The learning rate for the optimizer.
    - n_classes: The number of output classes.
    - save_n_epoch: The frequency of saving the model (in epochs).
    - weight_decay (optional): The weight decay for the optimizer. Default is 1e-8.

    Returns:
    - train_losses: A numpy array containing the training losses for each epoch.
    - valid_losses: A numpy array containing the validation losses for each epoch.
    """
    train_losses, valid_losses = \
        train_model(net,
                    device=device,
                    train_loader=train_loader,
                    valid_loader=valid_loader,
                    n_classes=n_classes)
    from config import CONFIG
    use_validate = valid_loader is not None

    train_csv_file = f"{CONFIG['save']['train_dir']}train_loss.csv"
    writer = CSVWriter(train_csv_file)
    writer.writes({'loss': train_losses}).flush()
    logging.info(f"Save validate loss values to {os.path.abspath(train_csv_file)}")
    if use_validate:
        valid_csv_file = f"{CONFIG['save']['valid_dir']}valid_loss.csv"
        writer = CSVWriter(valid_csv_file)
        writer.writes({'loss': valid_losses}).flush()
        logging.info(f"Save validate loss values to {os.path.abspath(valid_csv_file)}")

    train_loss_image_path = f"{CONFIG['save']['train_dir']}train_loss.png"
    draw_loss_graph(losses=train_losses, title='Train Losses',
                    filename=train_loss_image_path)
    logging.info(f"Save train loss graph to {os.path.abspath(train_loss_image_path)}")

    if use_validate:
        valid_loss_image_path = f"{CONFIG['save']['valid_dir']}valid_loss.png"
        draw_loss_graph(losses=valid_losses, title='Validation Losses',
                        filename=valid_loss_image_path)
        logging.info(f"Save validate loss graph to {os.path.abspath(valid_loss_image_path)}")

    if CONFIG['wandb']:
        import wandb
        if use_validate:
            wandb.log({'train_losses': train_losses, 'valid_losses': valid_losses,
                       'train_loss_image': train_loss_image_path, 'valid_loss_image': valid_loss_image_path})
        else:
            wandb.log({'train_losses': train_losses, 'train_loss_image': train_loss_image_path})
