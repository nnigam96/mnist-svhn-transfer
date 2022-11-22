import torch
from torchvision import datasets
from torchvision import transforms
from Hindi_dataset import *


def get_loader(config):
    """Builds and returns Dataloader for MNIST and SVHN dataset."""
    
    transform = transforms.Compose([
                    transforms.Resize(config.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5), (0.5))])
    
    #svhn = datasets.SVHN(root=config.svhn_path, download=True, transform=transform)
    mnist = datasets.MNIST(root=config.mnist_path, download=True, transform=transform)

#     img_rgb = Image.open('erock_rgb.jpg')

# img_gray = img_rgb.convert('L')

# img_gray.save('erock_gray.jpg')

    # svhn_loader = torch.utils.data.DataLoader(dataset=svhn,
    #                                           batch_size=config.batch_size,
    #                                           shuffle=True,
    #                                           num_workers=config.num_workers)

    mnist_loader = torch.utils.data.DataLoader(dataset=mnist,
                                               batch_size=config.batch_size,
                                               shuffle=True,
                                               num_workers=config.num_workers)

    hindi_mnist_loader = torch.utils.data.DataLoader(dataset=Hindi_Digits(),
                                               batch_size=config.batch_size,
                                               shuffle=True,
                                               num_workers=config.num_workers)

    return hindi_mnist_loader, mnist_loader