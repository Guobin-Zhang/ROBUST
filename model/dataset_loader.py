from torchvision import datasets

def get_multi_task_datasets(data_root='./data'):
    task_datasets = {
        'stl10': datasets.STL10(root=data_root, split='train', download=True,
            transform=transforms.Compose([
                transforms.Resize((192, 192)), transforms.ToTensor(),
                transforms.Normalize([0.5]*3, [0.5]*3)
            ])),
        'svhn': datasets.SVHN(root=data_root, split='train', download=True,
            transform=transforms.Compose([
                transforms.Resize((192, 192)), transforms.ToTensor(),
                transforms.Normalize([0.5]*3, [0.5]*3)
            ])),
        'cifar10': datasets.CIFAR10(root=data_root, train=True, download=True,
            transform=transforms.Compose([
                transforms.Resize((192, 192)), transforms.ToTensor(),
                transforms.Normalize([0.5]*3, [0.5]*3)
            ])),
        'imagenet': datasets.ImageNet(root=data_root, split='train',
            transform=transforms.Compose([
                transforms.Resize((192, 192)), transforms.ToTensor(),
                transforms.Normalize([0.5]*3, [0.5]*3)
            ]))
    }
    return task_datasets