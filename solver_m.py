import torch
import torch.nn as nn
import torchvision
import os
import pickle
import scipy.io
import numpy as np
from PIL import Image
from torch.autograd import Variable
from torch import optim
from model import G12, G21
from model import D1, D2
import matplotlib.pyplot as plt
from data_loader import
from torchvision.utils import save_image

# import matlplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
from torchmetrics.functional import structural_similarity_index_measure as ssim


class Solver(object):
    def __init__(self, config, svhn_loader, mnist_loader):
        self.svhn_loader = svhn_loader
        self.mnist_loader = mnist_loader
        self.gen_HINDI = None
        self.gen_MNIST = None
        self.disc_MNIST = None
        self.disc_HINDI = None
        self.g_optimizer = None
        self.d_optimizer = None
        self.use_reconst_loss = config.use_reconst_loss
        self.use_labels = config.use_labels
        self.num_classes = config.num_classes
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.train_iters = config.train_iters
        self.batch_size = config.batch_size
        self.lr = config.lr
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.sample_path = config.sample_path
        self.model_path = config.model_path
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.build_model()

    def build_model(self):
        """Builds a generator and a discriminator."""
        self.gen_HINDI = G12(conv_dim=self.g_conv_dim).to(self.DEVICE)
        self.gen_MNIST = G21(conv_dim=self.g_conv_dim).to(self.DEVICE)
        self.disc_MNIST = D1(conv_dim=self.d_conv_dim, use_labels=self.use_labels)
        self.disc_HINDI = D2(conv_dim=self.d_conv_dim, use_labels=self.use_labels)

        g_params = list(self.gen_HINDI.parameters()) + list(self.gen_MNIST.parameters())
        d_params = list(self.disc_MNIST.parameters()) + list(self.disc_HINDI.parameters())

        self.opt_gen = optim.Adam(g_params, self.lr, betas=(self.beta1, self.beta2))
        self.opt_disc = optim.Adam(d_params, self.lr, betas=(self.beta1, self.beta2))

        self.g_scaler = torch.cuda.amp.GradScaler()
        self.d_scaler = torch.cuda.amp.GradScaler()

        if torch.cuda.is_available():
            self.gen_HINDI.cuda()
            self.gen_MNIST.cuda()
            self.disc_MNIST.cuda()
            self.disc_HINDI.cuda()

    def merge_images(self, sources, targets, k=10):
        _, _, h, w = sources.shape
        row = int(np.sqrt(self.batch_size))
        merged = np.zeros([1, row * h, row * w * 2])
        for idx, (s, t) in enumerate(zip(sources, targets)):
            i = idx // row
            j = idx % row
            merged[:, i * h:(i + 1) * h, (j * 2) * h:(j * 2 + 1) * h] = s
            merged[:, i * h:(i + 1) * h, (j * 2 + 1) * h:(j * 2 + 2) * h] = t
        return merged.transpose(1, 2, 0)

    def to_var(self, x):
        """Converts numpy to variable."""
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x)

    def to_data(self, x):
        """Converts variable to numpy."""
        if torch.cuda.is_available():
            x = x.cpu()
        return x.data.numpy()

    def reset_grad(self):
        """Zeros the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    # def mse(self, x, y):
    #     mse = torch.nn.MSELoss()
    #     result = mse(x, y)
    #     return result

    def ssim_criterion(self,inp,op):
        result=ssim(inp,op)
        loss = 1 -result
        return loss

    def train(self):

        gen_HINDI = G12(conv_dim=self.g_conv_dim).to(self.DEVICE)
        gen_MNIST = G12(conv_dim=self.g_conv_dim).to(self.DEVICE)
        disc_MNIST = D1(conv_dim=self.d_conv_dim, use_labels=self.use_labels).to(self.DEVICE)
        disc_HINDI = D1(conv_dim=self.d_conv_dim, use_labels=self.use_labels).to(self.DEVICE)

        # g_params = list(gen_HINDI.parameters()) + list(gen_MNIST.parameters())
        # d_params = list(disc_MNIST.parameters()) + list(disc_HINDI.parameters())

        opt_gen = optim.Adam(list(gen_HINDI.parameters()) + list(gen_MNIST.parameters()), self.lr,
                             betas=(self.beta1, self.beta2))
        opt_disc = optim.Adam(list(disc_MNIST.parameters()) + list(disc_HINDI.parameters()), self.lr,
                              betas=(self.beta1, self.beta2))

        g_scaler = torch.cuda.amp.GradScaler()
        d_scaler = torch.cuda.amp.GradScaler()

        svhn_iter = iter(self.svhn_loader)
        mnist_iter = iter(self.mnist_loader)
        iter_per_epoch = min(len(svhn_iter), len(mnist_iter))

        # fixed mnist and svhn for sampling
        if torch.cuda.is_available():
            fixed_svhn = next(svhn_iter)[0].cuda()
            fixed_mnist = next(mnist_iter)[0].cuda()
        else:
            fixed_svhn = next(svhn_iter)[0].cpu()
            fixed_mnist = next(mnist_iter)[0].cpu()
            # loss if use_labels = True
        criterion = nn.CrossEntropyLoss()
        mse = nn.MSELoss()
        L1 = nn.L1Loss()

        # Add a summary writer
        writer = SummaryWriter()

        gen_loss = []
        disc1_loss = []
        disc2_loss = []
        disc_loss = []
        for step in range(self.train_iters + 1):
            # reset data_iter for each epoch
            if (step + 1) % iter_per_epoch == 0:
                mnist_iter = iter(self.mnist_loader)
                svhn_iter = iter(self.svhn_loader)

            # load svhn and mnist dataset
            hindi, h_labels = next(svhn_iter)
            hindi, s_labels = self.to_var(hindi), self.to_var(h_labels).long().squeeze()
            mnist, m_labels = next(mnist_iter)
            mnist, m_labels = self.to_var(mnist), self.to_var(m_labels)

            if self.use_labels:
                mnist_fake_labels = self.to_var(
                    torch.Tensor([self.num_classes] * hindi.size(0)).long())
                hindi_fake_labels = self.to_var(
                    torch.Tensor([self.num_classes] * mnist.size(0)).long())

            # Train discriminator MNIST and HINDI

            opt_disc.zero_grad()
            with torch.cuda.amp.autocast():

                # Plot the input mnist image

                mnist_plot = mnist[0, :, :, :].clone().detach().cpu().numpy().squeeze(0)
                plt.figure()
                plt.imshow(mnist_plot, cmap='gray')
                writer.add_figure('Input mnist image', plt.gcf(), global_step=step)

                # Plot the input Hindi image

                hindi_plot = hindi[0, :, :, :].clone().detach().cpu().numpy().squeeze(0)
                plt.figure()
                plt.imshow(hindi_plot, cmap='gray')
                writer.add_figure('Input Hindi image', plt.gcf(), global_step=step)

                # Train the hindi Discriminator
                fake_hindi = gen_HINDI(mnist)
                D_H_real = disc_HINDI(hindi)
                D_H_fake = disc_HINDI(fake_hindi.detach())
                D_H_real_loss = mse(D_H_real, torch.ones_like(D_H_real))
                D_H_fake_loss = mse(D_H_fake, torch.zeros_like(D_H_fake))
                D_H_loss = D_H_real_loss + D_H_fake_loss

                writer.add_scalar('Hindi discriminator loss', D_H_loss.item(), global_step=step)

                # plot the fake hindi

                fake_HINDI_plot = fake_hindi[1, :, :, :].clone().detach().cpu().squeeze(0).squeeze(0).numpy()
                plt.figure()
                plt.imshow(fake_HINDI_plot, cmap='gray')
                writer.add_figure('CycleMHM (Hindi)', plt.gcf(), global_step=step)

                fake_mnist = gen_MNIST(hindi)
                D_M_real = disc_MNIST(mnist)
                D_M_fake = disc_MNIST(fake_mnist.detach())
                D_M_real_loss = mse(D_M_real, torch.ones_like(D_M_real))
                D_M_fake_loss = mse(D_M_fake, torch.zeros_like(D_M_fake))
                D_M_loss = D_M_real_loss + D_M_fake_loss

                writer.add_scalar('MNIST Discriminator loss', D_M_loss.item(), global_step=step)

                # plot the fake mnist

                fake_MNIST_plot = fake_mnist[1, :, :, :].clone().detach().cpu().squeeze(0).squeeze(0).numpy()
                plt.figure()
                plt.imshow(fake_MNIST_plot, cmap='gray')
                writer.add_figure('CycleHMH (MNIST)', plt.gcf(), global_step=step)

            D_loss = (D_H_loss + D_M_loss) / 2
            d_scaler.scale(D_loss).backward()
            d_scaler.step(opt_disc)
            d_scaler.update()
            writer.add_scalar('Combined Discriminator loss', D_loss.item(), global_step=step)

           #==================================# Train Generators: =========================================


            opt_gen.zero_grad()
            with torch.cuda.amp.autocast():

                # Generator loss

                D_H_fake2 = disc_HINDI(fake_hindi)
                D_M_fake2 = disc_MNIST(fake_mnist)
                loss_Gen_HINDI = mse(D_H_fake2, torch.ones_like(D_H_fake2))
                loss_Gen_MNIST = mse(D_M_fake2, torch.ones_like(D_M_fake2))

                # Cycle loss

                reconst_MNIST = gen_MNIST(fake_hindi)
                cycle_MHM_loss = L1(mnist, reconst_MNIST)

                reconst_HINDI = gen_HINDI(fake_mnist)
                cycle_HMH_loss = L1(hindi, reconst_HINDI)

                # Plot the reconstructed images

                reconst_MNIST_plot = reconst_MNIST[0, :, :, :].clone().detach().cpu().numpy().squeeze(0)
                plt.figure()
                plt.imshow(reconst_MNIST_plot, cmap='gray')
                writer.add_figure('Cycle_MHM (Reconstructed MNIST)', plt.gcf(), global_step=step)

                reconst_HINDI_plot = reconst_HINDI[0, :, :, :].clone().detach().cpu().numpy().squeeze(0)
                plt.figure()
                plt.imshow(reconst_HINDI_plot, cmap='gray')
                writer.add_figure('Cycle_MHM (Reconstructed MNIST)', plt.gcf(), global_step=step)

                G_loss = (
                        loss_Gen_MNIST
                        + loss_Gen_HINDI
                        + cycle_MHM_loss
                        + cycle_HMH_loss

                )
                g_scaler.scale(G_loss).backward()
                g_scaler.step(opt_gen)
                g_scaler.update()
                writer.add_scalar('Combined Generator Loss', G_loss.item(), global_step=step)

            # print the log info
            if (step + 1) % self.log_step == 0:
                print('Step [%d/%d], Hindi_DISC_loss: %.4f, mnist_DISC_loss: %.4f, Combined Disc Loss: %.4f, '
                      'G_loss_combined: %.4f '
                      % (step + 1, self.train_iters, D_H_loss.item(), D_M_loss.item(),
                         D_loss.item(), G_loss.item()))
