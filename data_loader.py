import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import RandomSampler
from Hindi_dataset import *
import copy
from Hindi_dataset import Hindi_Digits
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

    mnist_loader = torch.utils.data.DataLoader(dataset=mnist,
                                               batch_size=config.batch_size,
                                               shuffle=True,
                                               num_workers=config.num_workers)



    hindi_mnist_loader = torch.utils.data.DataLoader(dataset=Hindi_Digits(),
                                               batch_size=config.batch_size,
                                               shuffle=True,
                                               num_workers=config.num_workers)

    
    def mnist_sampler(mnist_data):
        mnist_dict = {}
        for i in range(0,10):
            indices = mnist_data.targets == i # if you want to keep images with the label 5
            #temp_mnist.data, temp_mnist.targets = temp_mnist.data[indices], temp_mnist.targets[indices]
            mnist_dict[i]=mnist_data.data[indices]
        return mnist_dict

    def hindi_sampler(hindi=Hindi_Digits()):
        #def get_item_from_label(self, label):
        df  = hindi.annotation
        hindi_dict = {}
        for i in range(0,10):
            image_list = df.index[df['label']==i].tolist()
            hindi_dict[i]=image_list
        return hindi_dict

    def get_data_dicts(mnist_data, hindi):
        mnist_dict = mnist_sampler(mnist_data)
        hindi_dict = hindi_sampler(hindi)
        return mnist_dict, hindi_dict

    mnist_dict, hindi_dict = get_data_dicts(mnist, Hindi_Digits())


    return hindi_mnist_loader, mnist_loader, mnist_dict, hindi_dict


def get_random_mnist(mnist_dict, label):
    dataset = mnist_dict[label.item()]
    img_loader = torch.utils.data.DataLoader(dataset=mnist_dict[label.item()],
                                               batch_size=1,
                                               shuffle=True,
                                               )
    itr = iter(img_loader)
    img = next(itr)
    return img

def get_random_hindi(hindi_dict, label, hindi = Hindi_Digits()):
    df  = hindi.annotation
    image_list = hindi_dict[label.item()]
    rand_index = random.sample(image_list,1)
    image = Image.open(df.iloc[rand_index]['path'].item())
    image = transforms.PILToTensor()(image)
    image = image.to(torch.float32)
    return image 

def random_mnist_batch_gen(mnist_dict, labels):
    batch = torch.zeros([len(labels),1, 32, 32])
    transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5), (0.5))])
    for i,lb in enumerate(labels):
        img = get_random_mnist(mnist_dict, lb)
        img = transform(img.to(torch.uint8))
        batch[i] = torch.tensor(img)

    return batch

def random_hindi_batch_gen(hindi_dict, labels):
    batch = torch.zeros([len(labels), 1, 32, 32])
    transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5), (0.5))])
    for i,lb in enumerate(labels):
        img = get_random_hindi(hindi_dict, lb)
        img = transform(img.to(torch.uint8))

        batch[i] = torch.tensor(img)

    return batch