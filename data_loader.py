import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import RandomSampler
from Hindi_dataset import *
import copy

def get_loader(config):
    """Builds and returns Dataloader for MNIST and SVHN dataset."""
    
    transform = transforms.Compose([
                    transforms.Resize(config.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5), (0.5))])
    
    #svhn = datasets.SVHN(root=config.svhn_path, download=True, transform=transform)
    mnist = datasets.MNIST(root=config.mnist_path, download=True, transform=transform)
    
    #indices = mnist.targets == 1 # if you want to keep images with the label 5
    #mnist.data, mnist.targets = mnist.data[indices], mnist.targets[indices]

    def mnist_sampler(mnist):
        mnist_dict = {}
        for i in range(0,10):
            indices = mnist.targets == i # if you want to keep images with the label 5
            #temp_mnist.data, temp_mnist.targets = temp_mnist.data[indices], temp_mnist.targets[indices]
            mnist_dict[i]=mnist.data[indices]
        return mnist_dict



    mnist_loader = torch.utils.data.DataLoader(dataset=mnist,
                                               batch_size=config.batch_size,
                                               shuffle=True,
                                               num_workers=config.num_workers)



    hindi_mnist_loader = torch.utils.data.DataLoader(dataset=Hindi_Digits(),
                                               batch_size=config.batch_size,
                                               shuffle=True,
                                               num_workers=config.num_workers)

    mnist_dict = mnist_sampler(mnist)
    x = get_random_sample(mnist_dict, 2)
    return hindi_mnist_loader, mnist_loader, mnist_dict


def get_random_sample(mnist_dict, label):
    img_loader = torch.utils.data.DataLoader(dataset=mnist_dict[label],
                                               batch_size=1,
                                               shuffle=True,
                                               )
    
    itr = iter(img_loader)
    img = next(itr)
    return img
