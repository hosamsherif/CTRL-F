
import torch
import warnings
from torch import nn
from torchvision import datasets, models, transforms, utils
from dataset import get_transform
from timm.models import create_model
from models import *
import argparse

warnings.filterwarnings("ignore", category=UserWarning)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate_akf(model, dataloader,device, alpha_eval):

    criterion = nn.CrossEntropyLoss()
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    n_imgs = len(dataloader.dataset)

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs,alpha_eval)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / n_imgs
    epoch_acc = running_corrects.double() / n_imgs

    return {'loss':epoch_loss, 'accuracy':epoch_acc}


@torch.no_grad()
def evaluate_ckf(model, dataloader,device):

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

    return {'loss':epoch_loss, 'accuracy':epoch_acc}


def build_dataset(is_train,args):
    data_transforms = get_transform(is_train,args)
    image_datasets = datasets.ImageFolder(args.data_path, data_transforms)
    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size = args.batch_size,shuffle=False,num_workers=0)
    data_size=len(image_datasets)
    return dataloaders, data_size


def get_args_parser():
    parser = argparse.ArgumentParser('CTRL-F evaluation script', add_help=False)
    
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--model', default='CTRLF_B_AKF', type=str, metavar='MODEL',
                    help='Name of model to train')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--data-path', default='flowers/', type=str,
                        help='dataset path')
    parser.add_argument('--input-size', nargs=2, type=int, default=[224, 224], help='images input size')
    parser.add_argument('--n_classes', type=int, default=102, help='number of classes within the dataset')

    parser.add_argument('--scaling-factor', type=float, default=100,metavar='ALPHA',
                        help='scaling factor to be multiplied by the normalized output vectors produced from CNN and MFCA when using AKF fusion')

    return parser

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    is_akf = 'AKF' in args.model

    test_dataloader , _ = build_dataset(False, args)

    if is_akf:
      model = create_model( 
          args.model,
          num_classes=args.n_classes,
          scaling_factor=args.scaling_factor)
    else:
      model = create_model(
          args.model,
          num_classes=args.n_classes)

    model.to(device)

    if device == "cuda":
        checkpoint = torch.load(args.checkpoint)
    else:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')

    model_state_dict = checkpoint['model_state_dict']
    model.load_state_dict(model_state_dict)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)


    if is_akf:
        best_alpha = checkpoint['alpha']
        test_stats = evaluate_akf(model, test_dataloader, device, best_alpha)

    else:
        test_stats = evaluate_ckf(model, test_dataloader, device)

    print('Testing Loss: {:.4f} Acc: {:.4f}'.format(test_stats['loss'], test_stats['accuracy']))

    print('Acc: {:.4f}'.format(test_stats['accuracy']))


if __name__  == '__main__':
    parser = argparse.ArgumentParser('CTRL-F evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
