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
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data_loader import *
from torchvision.utils import save_image

from torch.utils.tensorboard import SummaryWriter
from torchmetrics.functional import structural_similarity_index_measure as ssim


class Solver(object):
    def __init__(self, config, svhn_loader, mnist_loader, mnist_dict, hindi_dict):
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
        self.mnist_dict = mnist_dict
        self.hindi_dict = hindi_dict
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

    def ssim_criterion(self, inp, op):
        result = ssim(inp, op)
        loss = 1 - result
        return loss

    def train(self):

        # Generator expects 2 channels
        gen_HINDI = G12(conv_dim=self.g_conv_dim).to(self.DEVICE)
        gen_MNIST = G12(conv_dim=self.g_conv_dim).to(self.DEVICE)

        # Disc expects 1 channel
        disc_MNIST = D1(conv_dim=self.d_conv_dim, use_labels=self.use_labels).to(self.DEVICE)
        disc_HINDI = D1(conv_dim=self.d_conv_dim, use_labels=self.use_labels).to(self.DEVICE)

        # Semantic disc are 1 channel
        semantic_disc_HINDI = G21(conv_dim=self.g_conv_dim).to(self.DEVICE)
        semantic_disc_MNIST = G21(conv_dim=self.g_conv_dim).to(self.DEVICE)

        # g_params = list(gen_HINDI.parameters()) + list(gen_MNIST.parameters())
        # d_params = list(disc_MNIST.parameters()) + list(disc_HINDI.parameters())

        opt_gen = optim.AdamW(list(gen_HINDI.parameters()) + list(gen_MNIST.parameters()), self.lr,
                              betas=(self.beta1, self.beta2))

        opt_disc = optim.AdamW(list(disc_MNIST.parameters()) + list(disc_HINDI.parameters()), self.lr,
                               betas=(self.beta1, self.beta2))

        # opt_disc = optim.AdamW(list(disc_MNIST.parameters()) + list(disc_HINDI.parameters())
        #                        + list(semantic_disc_MNIST.parameters()) + list(semantic_disc_HINDI.parameters()),
        #                        self.lr,
        #                        betas=(self.beta1, self.beta2))

        opt_disc_MNIST = optim.Adam(list(disc_MNIST.parameters())
                                    + list(semantic_disc_MNIST.parameters()), self.lr,
                                    betas=(self.beta1, self.beta2))

        opt_disc_HINDI = optim.Adam(list(disc_HINDI.parameters())
                                    + list(semantic_disc_HINDI.parameters()),
                                    self.lr,
                                    betas=(self.beta1, self.beta2))

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
            # hindi, h_labels = next(svhn_iter)
            # hindi, s_labels = self.to_var(hindi), self.to_var(h_labels).long().squeeze()
            # mnist, m_labels = next(mnist_iter)
            # mnist, m_labels = self.to_var(mnist), self.to_var(m_labels)

            hindi, h_labels = next(svhn_iter)
            hindi, s_labels = self.to_var(hindi), self.to_var(h_labels).long().squeeze()
            mnist, m_labels = random_mnist_batch_gen(mnist_dict=self.mnist_dict, labels=h_labels), h_labels
            mnist, m_labels = self.to_var(mnist), self.to_var(m_labels)

            # Append zero channels to MNIST

            mnist = torch.stack([mnist, torch.zeros_like(mnist)], 2).squeeze(1)
            hindi = torch.stack([hindi, torch.zeros_like(hindi)], 2).squeeze(1)

            if self.use_labels:
                mnist_fake_labels = self.to_var(
                    torch.Tensor([self.num_classes] * hindi.size(0)).long())
                hindi_fake_labels = self.to_var(
                    torch.Tensor([self.num_classes] * mnist.size(0)).long())

            # Train discriminator MNIST and HINDI

            opt_disc.zero_grad()


            # ================= =========        INPUT : MNIST - HINDI-- MNIST         ==================================

            # ==========Train the HINDI Distributive Discriminator for the MHM cycle  =======================


            # Makes sure the dataset generated is in HINDI (Does not have one-to one correspondence to MNIST Inputs)
            # Given MNIST (1,3,5) can generate HINDI (1,2,6).

            gen_Hindi_op = gen_HINDI(mnist).detach()

            encA1 = gen_Hindi_op[:, 1, :, :].unsqueeze(1) # Get an encoded channel encA for
            fake_hindi = gen_Hindi_op[:, 0, :, :].unsqueeze(1)  #

            D_H_real = disc_HINDI(hindi[:, 0, :, :].unsqueeze(1))
            D_H_fake = disc_HINDI(fake_hindi.detach())
            D_H_encA_loss = L1(encA1, torch.zeros_like(encA1))

            # l1_lambda = 0.01
            # l1_norm = sum(p.abs().sum()
            #               for p in gen_HINDI.parameters())
            #
            # gen_Hindi_encA_loss_reg = D_H_encA_loss + l1_lambda * l1_norm

            D_H_real_loss = L1(D_H_real, torch.ones_like(D_H_real))
            D_H_fake_loss = L1(D_H_fake, torch.zeros_like(D_H_fake))
            D_H_loss = D_H_real_loss + D_H_fake_loss
            writer.add_scalar('Hindi discriminator loss', D_H_loss.item(), global_step=step)

            # ====================-   Train the semantic Hindi discriminator     ====================

            # SemanticDisc_HINDI_images = semantic_disc_HINDI(fake_hindi.detach())

            # Sample from the HINDI dataset given label from input MNIST
            # Then compare with the generated HINDI
            # ( Ensure one-to-one correspondence between inputs(MNIST:1,3,7) and generated(HINDI:1,3,5) )
            # Compares the generated (HINDI: 1,3,5) to (sampled_HINDI:1,3,7)
            # Forces the Generator to produce 1,3,7 in Hindi.
            # MNIST Labels: m_labels
            # sampled_Hindi_batch = random_hindi_batch_gen(hindi_dict=self.hindi_dict, labels=m_labels)
            # sampled_Hindi_batch = HINDI_BATCH_SAMPLER() # Nikhil will add this
            # semantic_D_H_loss = L1(SemanticDisc_HINDI_images, sampled_Hindi_batch.to(self.DEVICE))

            # # plot the fake hindi
            # fake_HINDI_plot = fake_hindi[1, :, :, :].clone().detach().cpu().squeeze(0).squeeze(0).numpy()
            # plt.figure()
            # plt.imshow(fake_HINDI_plot, cmap='gray')
            # writer.add_figure('CycleMHM (Hindi)', plt.gcf(), global_step=step)

            # ================================ INPUT : HINDI -- MNIST-- HINDI ==================================

            # ================================ Train the MNIST Discriminator ==============================

            gen_MNIST_op = gen_MNIST(hindi).detach()
            D_M_real = disc_MNIST(mnist[:, 0, :, :].unsqueeze(1))
            encA2 = gen_MNIST_op[:, 1, :, :].unsqueeze(1)
            fake_mnist = gen_MNIST_op[:, 0, :, :].unsqueeze(1)
            D_M_encB_loss = L1(encA2, torch.zeros_like(encA2))

            # l1_lambda = 0.01
            # l1_norm = sum(p.abs().sum()
            #               for p in gen_MNIST.parameters())
            #
            # gen_MNIST_encB_loss_reg = D_M_encB_loss + l1_lambda * l1_norm

            D_M_fake = disc_MNIST(fake_mnist.detach())
            D_M_real_loss = L1(D_M_real, torch.ones_like(D_M_real))
            D_M_fake_loss = L1(D_M_fake, torch.zeros_like(D_M_fake))
            D_M_loss = D_M_real_loss + D_M_fake_loss

            writer.add_scalar('MNIST Discriminator loss', D_M_loss.item(), global_step=step)

            # ================================= Train the semantic MNIST discriminator ======================
            # SemanticDisc_MNIST_images = semantic_disc_MNIST(fake_mnist.detach())
            # sampled_MNIST_batch = random_mnist_batch_gen(mnist_dict=self.mnist_dict, labels=h_labels)
            # sampled_MNIST_batch = MNIST_BATCH_SAMPLER() # Nikhil to add
            # semantic_D_M_loss = L1(SemanticDisc_MNIST_images, sampled_MNIST_batch.to(self.DEVICE))

            # # plot the fake mnist
            #
            # fake_MNIST_plot = fake_mnist[1, :, :, :].clone().detach().cpu().squeeze(0).squeeze(0).numpy()
            # plt.figure()
            # plt.imshow(fake_MNIST_plot, cmap='gray')
            # writer.add_figure('CycleHMH (MNIST)', plt.gcf(), global_step=step)

            # D_loss = (D_H_loss + D_M_loss + semantic_D_M_loss + semantic_D_H_loss) / 4
            D_loss = (D_H_loss + D_M_loss) / 2
            D_loss.backward()
            opt_disc.step()

            writer.add_scalar('Combined Discriminator loss', D_loss.item(), global_step=step)

            # ==================================# Train Generators: =========================================

            opt_gen.zero_grad()
            # with torch.cuda.amp.autocast():

            # Generator loss

            # D_H_fake2 = disc_HINDI(fake_hindi.detach())
            # D_M_fake2 = disc_MNIST(fake_mnist.detach())
            # loss_Gen_HINDI = L1(D_H_fake2.detach(), torch.ones_like(D_H_fake2))
            # loss_Gen_MNIST = L1(D_M_fake2.detach(), torch.ones_like(D_M_fake2))

            # Cycle loss

            reconst_MNIST = gen_MNIST(gen_Hindi_op.detach())
            cycle_MHM_loss = L1(mnist, reconst_MNIST[:, 0, :, :].unsqueeze(1))

            # Plot the reconstructed images

            # reconst_MNIST_plot = reconst_MNIST[0, :, :, :].clone().detach().cpu().numpy().squeeze(0)
            # plt.figure()
            # plt.imshow(reconst_MNIST_plot, cmap='gray')
            # writer.add_figure('Cycle_MHM (Reconstructed MNIST)', plt.gcf(), global_step=step)

            # if (step % 5) == 0:
            #     G_loss_cycleMHM = (
            #             loss_Gen_MNIST
            #             + loss_Gen_HINDI
            #             + cycle_MHM_loss)

                # G_loss_cycleMHM.backward()
                # opt_gen.step()
            # writer.add_scalar('Combined Generator Loss', G_loss.item(), global_step=step)

            # ================================ Cycle HMH loss ======================================

            D_H_fake2 = disc_HINDI(fake_hindi.detach())
            D_M_fake2 = disc_MNIST(fake_mnist.detach())
            loss_Gen_HINDI = L1(D_H_fake2.detach(), torch.ones_like(D_H_fake2))
            loss_Gen_MNIST = L1(D_M_fake2.detach(), torch.ones_like(D_M_fake2))

            reconst_HINDI = gen_HINDI(gen_MNIST_op.detach())
            cycle_HMH_loss = L1(hindi, reconst_HINDI[:, 0, :, :].unsqueeze(1))
            #
            # if step % 1  == 0:
            #     G_loss_cycleHMH = (
            #             loss_Gen_MNIST
            #             + loss_Gen_HINDI
            #             + cycle_HMH_loss
            #
            #     )

                # G_loss_cycleHMH.backward()
                # opt_gen.step()


            # ============================== Idnetitiy loss ========================================================= #

            identity_MNIST = gen_MNIST(mnist)
            identity_HINDI = gen_HINDI(hindi)
            identity_MNIST_loss = L1(mnist, identity_MNIST[:, 0, :, :].unsqueeze(1))
            identity_HINDI_loss = L1(hindi, identity_HINDI[:, 0, :, :].unsqueeze(1))



           # Combined Generator loss

            G_loss =  (
                        loss_Gen_MNIST
                        + loss_Gen_HINDI
                        + cycle_MHM_loss
                        + cycle_HMH_loss+
                        +D_H_encA_loss
                        +D_M_encB_loss
                        + identity_HINDI_loss
                        + identity_MNIST_loss)

            G_loss.backward()
            opt_gen.step()

            # reconst_HINDI_plot = reconst_HINDI[0, :, :, :].clone().detach().cpu().numpy().squeeze(0)
            # plt.figure()
            # plt.imshow(reconst_HINDI_plot, cmap='gray')
            # writer.add_figure('Cycle_HMH (Reconstructed HINDI)', plt.gcf(), global_step=step)

            # =============================== Adding regularization to the Generators =========================

            # params_gen_HINDI = torch.cat([x.view(-1) for x in gen_HINDI.parameters()])
            # params_gen_MNIST = torch.cat([x.view(-1) for x in gen_MNIST.parameters()])
            #
            # lambda1 = 1
            # lambda2 = 1
            #
            # l1_gen_HINDI = lambda1 * torch.norm(params_gen_HINDI, 1)
            # l1_gen_MNIST = lambda2 * torch.norm(params_gen_MNIST, 1)

            # Plot images for Cycle MHM

            mnist_plot = mnist[0, 0, :, :].clone().detach().cpu().numpy()
            fake_HINDI_plot = fake_hindi[0, 0, :, :].clone().detach().cpu().numpy()
            encA1_plot = encA1[0, 0, :, :].clone().detach().cpu().numpy()
            reconst_MNIST_plot = reconst_MNIST[0, 0, :, :].clone().detach().cpu().numpy()

            # Plot images for Cycle HMH

            hindi_plot = hindi[0, 0, :, :].clone().detach().cpu().numpy()
            fake_MNIST_plot = fake_mnist[0, 0, :, :].clone().detach().cpu().numpy()
            encA2_plot = encA2[0, 0, :, :].clone().detach().cpu().numpy()
            reconst_HINDI_plot = reconst_HINDI[0, 0, :, :].clone().detach().cpu().numpy()

            ####

            plt.figure()
            plt.subplot(2, 4, 1)
            plt.imshow(mnist_plot, cmap='gray')
            plt.xlabel('MNIST')
            plt.subplot(2, 4, 2)
            plt.imshow(fake_HINDI_plot, cmap='gray')
            plt.xlabel('Fake HINDI')
            plt.subplot(2, 4, 3)
            plt.imshow(encA1_plot, cmap='gray')
            plt.xlabel('EncA1')
            plt.subplot(2, 4, 4)
            plt.imshow(reconst_MNIST_plot, cmap='gray')
            plt.xlabel('Reconst MNIST')

            plt.subplot(2, 4, 5)
            plt.imshow(hindi_plot, cmap='gray')
            plt.xlabel('Hindi')
            plt.subplot(2, 4, 6)
            plt.imshow(fake_MNIST_plot, cmap='gray')
            plt.xlabel('Fake MNIST')
            plt.subplot(2, 4, 7)
            plt.imshow(encA2_plot, cmap='gray')
            plt.xlabel('Enc A2')
            plt.subplot(2, 4, 8)
            plt.imshow(reconst_HINDI_plot, cmap='gray')
            plt.xlabel('Reconst HINDI')
            writer.add_figure('Cycle MHM + HMH', plt.gcf(), global_step=step)

            # print the log info
            if (step + 1) % self.log_step == 0:
                # print('Step [%d/%d], Hindi_DISC_loss: %.4f, mnist_DISC_loss: %.4f, Combined Disc Loss: %.4f, '
                #       'G_loss_combined_HMH: %.4f ', 'G_loss_combined_MHM: %.4f' % (step + 1, self.train_iters, D_H_loss.item(), D_M_loss.item(),
                #          D_loss.item(),G_loss_cycleHMH.item(),G_loss_cycleMHM.item()))

                print('Step [%d/%d]' % (step + 1, self.train_iters))


            #writer.add_text()
