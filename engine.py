import torch
import torch.nn as nn


def train_one_epoch_akf(model, dataloader, criterion, optimizer, device, epoch, num_epochs, alpha_min, alpha_max, fixed_epochs, alpha_step):
    model.train()
    print('Epoch {}/{}'.format(epoch + 1, num_epochs))
    print('-' * 20)
    running_loss = 0.0
    running_corrects = 0
    n_imgs = len(dataloader.dataset)

    if epoch < (num_epochs - fixed_epochs):
        alpha = alpha_min + epoch * alpha_step
    else:
        alpha = alpha_max

    print("Alpha value --> {}".format(alpha))

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            outputs = model(inputs, alpha)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / n_imgs
    epoch_acc = running_corrects.double() / n_imgs

    return {'loss': epoch_loss, 'accuracy': epoch_acc}


def train_one_epoch_ckf(model, dataloader, criterion, optimizer, device, epoch, num_epochs):
    model.train()
    print('Epoch {}/{}'.format(epoch + 1, num_epochs))
    print('-' * 20)
    running_loss = 0.0
    running_corrects = 0
    n_imgs = len(dataloader.dataset)

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / n_imgs
    epoch_acc = running_corrects.double() / n_imgs

    return {'loss': epoch_loss, 'accuracy': epoch_acc}


@torch.no_grad()
def evaluate_akf(model, dataloader, device, epoch, num_epochs, alpha_min, alpha_max, fixed_epochs, alpha_step):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    n_imgs = len(dataloader.dataset)

    if epoch < (num_epochs - fixed_epochs):
        alpha = alpha_min + epoch * alpha_step
    else:
        alpha = alpha_max

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs, alpha)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / n_imgs
    epoch_acc = running_corrects.double() / n_imgs

    return {'loss': epoch_loss, 'accuracy': epoch_acc}


@torch.no_grad()
def evaluate_ckf(model, dataloader, device):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    n_imgs = len(dataloader.dataset)

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / n_imgs
    epoch_acc = running_corrects.double() / n_imgs

    return {'loss': epoch_loss, 'accuracy': epoch_acc}