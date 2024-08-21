
import torch
import time
import datetime
import warnings
import numpy as np
import os
from torch import nn
from torchvision import datasets, models, transforms, utils
from dataset import build_dataset
from engine import train_one_epoch_akf, train_one_epoch_ckf, evaluate_akf, evaluate_ckf
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.models import create_model
from models import *
import argparse

warnings.filterwarnings("ignore", category=UserWarning)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_args_parser():
    parser = argparse.ArgumentParser('CTRL-F training and evaluation script', add_help=False)


    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--epochs', default=200, type=int)

    # model parameters
    parser.add_argument('--model', default='CTRLF_B_AKF', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', nargs=2, type=int, default=[224, 224], help='images input size')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.08,
                        help='weight decay (default: 0.08)')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-5, metavar='LR',
                        help='warmup learning rate (default: 1e-5)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic scheduxlers that hit 0 (1e-5)')


    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10)')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Dataset parameters 
    parser.add_argument('--data-path', default='flowers/', type=str,
                        help='dataset path')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for saving in the same directory')
    parser.add_argument('--seed', default=0, type=int)

    # Augmentation parameters
    parser.add_argument('--rotate', type=int, default=180,
                        help='Rotate the image by the given degrees (default: 180)')
    parser.add_argument('--horizontal_flip', action='store_true', default=True,
                        help='Apply horizontal flip (default: true)')
    parser.add_argument('--vertical_flip', action='store_true', default=True,
                        help='Apply vertical flip (default: true)')
    parser.add_argument('--brightness', type=float, default=0.3,
                        help='Adjust the brightness. 1 means no change (default:0.3)')
    parser.add_argument('--contrast', type=float, default=0.3,
                        help='Adjust the contrast. 1 means no change (default:0.3)')
    parser.add_argument('--saturation', type=float, default=None,
                        help='Adjust the saturation. 1 means no change (default:None)')
    parser.add_argument('--hue', type=float, default=None,
                        help='Adjust the hue. 0 means no change (default:None)')
    parser.add_argument('--sharpness', type=float, default=2,
                        help='Adjust the sharpness. 1 means no change (default:2)')
    parser.add_argument('--blur', type=int, default=3,
                        help='Apply Gaussian blur with the given radius (default:3)')

    # Knowledge Fusion parameters
    parser.add_argument('--alpha_min', type=float, default=0.3, metavar='ALPHA',
                        help=' beginning value of alpha (default: 0.3)')
    parser.add_argument('--alpha_max', type=float, default=0.7, metavar='ALPHA',
                        help=' last value of alpha (default: 0.7)')
    parser.add_argument('--steady_alpha_epochs', type=int, default=0, metavar='ALPHA',
                        help='number of epoches where alpha is fixed after reaching alpha_max (default: 0)')
    parser.add_argument('--scaling-factor', type=float, default=100,metavar='ALPHA',
                        help='scaling factor to be multiplied by the normalized output vectors produced from CNN and MFCA')
    parser.add_argument('--ckf_dropout', type=float, default=0.5, metavar='CKF_DROPOUT',
                        help='dropout value for collaborative knowledge fusion (default: 0.5)')

    return parser

def main(args):
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    is_akf = 'AKF' in args.model

    train_dataloader, args.n_classes , _ = build_dataset(True, args)

    test_dataloader, _ , _ = build_dataset(False, args)

    if is_akf:
      model = create_model(
          args.model,
          num_classes=args.n_classes,
          scaling_factor=args.scaling_factor)
    else:
      model = create_model(
          args.model,
          num_classes=args.n_classes,
          drop=args.ckf_dropout)

    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = create_optimizer(args,model)

    lr_scheduler, _ = create_scheduler(args, optimizer)
    num_epochs = args.epochs
    max_accuracy = 0.0

    alpha_step = (args.alpha_max - args.alpha_min) / (num_epochs - args.steady_alpha_epochs)
    for epoch in range(num_epochs):
        if is_akf:
            train_stats = train_one_epoch_akf(
                      model, train_dataloader, criterion,
                      optimizer, device, epoch , num_epochs, args.alpha_min,args.alpha_max,args.steady_alpha_epochs,alpha_step
            )
            test_stats = evaluate_akf(model, test_dataloader, device,epoch,num_epochs,args.alpha_min,args.alpha_max,args.steady_alpha_epochs,alpha_step)

        else:
            train_stats = train_one_epoch_ckf(
                      model, train_dataloader, criterion,
                      optimizer, device, epoch , num_epochs
            )
            test_stats = evaluate_ckf(model, test_dataloader, device)


        lr_scheduler.step(epoch)

        print('Training Loss: {:.4f} Acc: {:.4f}'.format(train_stats['loss'], train_stats['accuracy']))

        print('Testing Loss: {:.4f} Acc: {:.4f}'.format(test_stats['loss'], test_stats['accuracy']))

        print()

        max_accuracy = max(max_accuracy, test_stats["accuracy"])

        if max_accuracy == test_stats['accuracy']:
            if is_akf:
                best_alpha = args.alpha_min + epoch * alpha_step
                torch.save({
                  'model_state_dict': model.state_dict(),
                  'alpha': best_alpha,
                   } , args.output_dir + 'best_model.pth')
            else:
                torch.save({'model_state_dict': model.state_dict()},args.output_dir + 'best_model.pth')

    print("best testing accuracy: ",max_accuracy.item())
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__  == '__main__':
    parser = argparse.ArgumentParser('CTRL-F training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
