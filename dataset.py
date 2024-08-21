
import os
import torch
from torchvision import datasets, transforms, utils

def build_dataset(is_train,args):
    data_transforms = get_transform(is_train,args)

    folder = 'train' if is_train else 'test'
    image_datasets = datasets.ImageFolder(os.path.join(args.data_path,folder), data_transforms)
    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size = args.batch_size,shuffle=True,num_workers=0)
    data_size=len(image_datasets)
    classes_names=image_datasets.classes
    n_classes=len(classes_names)
    return dataloaders, n_classes, data_size



def get_transform(is_train, args):
    transform_list = []
    transform_list.append(transforms.Resize(tuple(args.input_size)))
    if is_train:
        if args.rotate:
            transform_list.append(transforms.RandomRotation(args.rotate))

        if args.horizontal_flip:
            transform_list.append(transforms.RandomHorizontalFlip())

        if args.vertical_flip:
            transform_list.append(transforms.RandomVerticalFlip())

        color_jitter_params = {}
        if args.brightness:
            color_jitter_params['brightness'] = args.brightness
        if args.contrast:
            color_jitter_params['contrast'] = args.contrast
        if args.saturation:
            color_jitter_params['saturation'] = args.saturation
        if args.hue:
            color_jitter_params['hue'] = args.hue

        if color_jitter_params:
            transform_list.append(transforms.ColorJitter(**color_jitter_params))

        if args.sharpness:
            transform_list.append(transforms.RandomAdjustSharpness(args.sharpness))

        if args.blur:
            transform_list.append(transforms.GaussianBlur(args.blur))

    transform_list.append(transforms.ToTensor())

    default_mean=[0.507, 0.487, 0.441]
    default_std=[0.267, 0.256, 0.276]
    transform_list.append(transforms.Normalize(default_mean,default_std))


    return transforms.Compose(transform_list)
