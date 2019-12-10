#!/usr/bin/python3

import argparse
import itertools
import os
import sys
from time import gmtime, strftime

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch

from models import Generator
from models import Discriminator
from utils import ReplayBuffer
from utils import LambdaLR
from utils import Logger
from utils import weights_init_normal
from datasets import ImageDataset

from torchvision.utils import save_image

class experiment():
    
    def __init__(self, epoch=0, n_epochs=1000, batchSize=1, lr=0.0002, decay_epoch=100, size=256, input_nc=3, output_nc=3, cuda=True, n_cpu=8, load_from_ckpt=False):
        
        self.epoch = epoch
        self.n_epochs = n_epochs
        self.batchSize = batchSize
        self.lr = lr
        self.decay_epoch = decay_epoch
        self.size = size
        self.input_nc = input_nc
        self.output_nc = output_nc
        self. cuda = cuda
        self. n_cpu = n_cpu


        rootA = "../dataset/monet_field_data"
        rootB = "../dataset/field_data"

        if torch.cuda.is_available() and not self.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")

        ###### Definition of variables ######
        # Networks
        self.netG_A2B = Generator(self.input_nc, self.output_nc)
        self.netG_B2A = Generator(self.output_nc, self.input_nc)
        self.netD_A = Discriminator(self.input_nc)
        self.netD_B = Discriminator(self.output_nc)
        
        if load_from_ckpt:
            print("loading from ckpt")
            self.netG_A2B.load_state_dict(torch.load('output/netG_A2B.pth'))
            self.netG_B2A.load_state_dict(torch.load('output/netG_B2A.pth'))
            self.netD_A.load_state_dict(torch.load('output/netD_A.pth'))
            self.netD_B.load_state_dict(torch.load('output/netD_B.pth'))
        else:
            self.netG_A2B.apply(weights_init_normal)
            self.netG_B2A.apply(weights_init_normal)
            self.netD_A.apply(weights_init_normal)
            self.netD_B.apply(weights_init_normal)      
            
        if self.cuda:
            self.netG_A2B.cuda()
            self.netG_B2A.cuda()
            self.netD_A.cuda()
            self.netD_B.cuda()

        # Lossess
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_cycle = torch.nn.L1Loss()
        self.criterion_identity = torch.nn.L1Loss()

        # Optimizers & LR schedulers
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A2B.parameters(), self.netG_B2A.parameters()),
                                        lr=self.lr, betas=(0.5, 0.999))
        self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=self.lr, betas=(0.5, 0.999))

        self.lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(self.optimizer_G, lr_lambda=LambdaLR(self.n_epochs, self.epoch, self.decay_epoch).step)
        self.lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D_A, lr_lambda=LambdaLR(self.n_epochs, self.epoch, self.decay_epoch).step)
        self.lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D_B, lr_lambda=LambdaLR(self.n_epochs, self.epoch, self.decay_epoch).step)

        if load_from_ckpt:
            print('load states')
            checkpoint = torch.load('output/states.pth')
            '''
            self.optimizer_G.load_state_dict(checkpoint['optimizer_G'])
            self.optimizer_D_A.load_state_dict(checkpoint['optimizer_D_A'])
            self.optimizer_D_B.load_state_dict(checkpoint['optimizer_D_B'])
            
            self.lr_scheduler_G.load_state_dict(checkpoint['lr_scheduler_G'])
            self.lr_scheduler_D_A.load_state_dict(checkpoint['lr_scheduler_D_A'])
            self.lr_scheduler_D_B.load_state_dict(checkpoint['lr_scheduler_D_B'])
            '''
            
            self.lr = checkpoint['lr']
            self.epoch = checkpoint['epoch']+1
            
        
        # Inputs & targets memory allocation
        Tensor = torch.cuda.FloatTensor if self.cuda else torch.Tensor
        self.input_A = Tensor(self.batchSize, self.input_nc, self.size, self.size)
        self.input_B = Tensor(self.batchSize, self.output_nc, self.size, self.size)
        self.target_real = Variable(Tensor(self.batchSize).fill_(1.0), requires_grad=False)
        self.target_fake = Variable(Tensor(self.batchSize).fill_(0.0), requires_grad=False)

        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()

        # Dataset loader
        transforms_ = [ transforms.Resize((int(self.size*1.12), int(self.size*1.12)), Image.BICUBIC), 
                        transforms.RandomCrop(self.size), 
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
        self.dataloader = DataLoader(ImageDataset(rootA, rootB, transforms_=transforms_, unaligned=True), 
                                batch_size=self.batchSize, shuffle=True, num_workers=self.n_cpu)

        # Loss plot
        #logger = Logger(self.n_epochs, len(dataloader))
        ###################################

    def train(self):
        ###### Training ######
        for epoch in range(self.epoch, self.n_epochs):
            for i, batch in enumerate(self.dataloader):
                
                # Set model input
                real_A = Variable(self.input_A.copy_(batch['A']))
                real_B = Variable(self.input_B.copy_(batch['B']))

                ###### Generators A2B and B2A ######
                self.optimizer_G.zero_grad()

                # Identity loss
                # G_A2B(B) should equal B if real B is fed
                same_B = self.netG_A2B(real_B)
                loss_identity_B = self.criterion_identity(same_B, real_B)*5.0
                # G_B2A(A) should equal A if real A is fed
                same_A = self.netG_B2A(real_A)
                loss_identity_A = self.criterion_identity(same_A, real_A)*5.0

                # GAN loss
                fake_B = self.netG_A2B(real_A)
                pred_fake = self.netD_B(fake_B)
                loss_GAN_A2B = self.criterion_GAN(pred_fake, self.target_real)

                fake_A = self.netG_B2A(real_B)
                pred_fake = self.netD_A(fake_A)
                loss_GAN_B2A = self.criterion_GAN(pred_fake, self.target_real)

                # Cycle loss
                recovered_A = self.netG_B2A(fake_B)
                loss_cycle_ABA = self.criterion_cycle(recovered_A, real_A)*10.0

                recovered_B = self.netG_A2B(fake_A)
                loss_cycle_BAB = self.criterion_cycle(recovered_B, real_B)*10.0

                # Total loss
                loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
                loss_G.backward()
                
                self.optimizer_G.step()
                ###################################

                ###### Discriminator A ######
                self.optimizer_D_A.zero_grad()

                # Real loss
                pred_real = self.netD_A(real_A)
                loss_D_real = self.criterion_GAN(pred_real, self.target_real)

                # Fake loss
                fake_A = self.fake_A_buffer.push_and_pop(fake_A)
                pred_fake = self.netD_A(fake_A.detach())
                loss_D_fake = self.criterion_GAN(pred_fake, self.target_fake)

                # Total loss
                loss_D_A = (loss_D_real + loss_D_fake)*0.5
                loss_D_A.backward()

                self.optimizer_D_A.step()
                ###################################

                ###### Discriminator B ######
                self.optimizer_D_B.zero_grad()

                # Real loss
                pred_real = self.netD_B(real_B)
                loss_D_real = self.criterion_GAN(pred_real, self.target_real)
                
                # Fake loss
                fake_B = self.fake_B_buffer.push_and_pop(fake_B)
                pred_fake = self.netD_B(fake_B.detach())
                loss_D_fake = self.criterion_GAN(pred_fake, self.target_fake)

                # Total loss
                loss_D_B = (loss_D_real + loss_D_fake)*0.5
                loss_D_B.backward()

                self.optimizer_D_B.step()

                
                
                ###################################
                if i%100 == 0:
                    text = [strftime("%Y-%m-%d %H:%M:%S", gmtime()), "epoch:{} batch_id:{}".format(epoch, i), "loss_g: {:.4f}".format(loss_G), "loss_DA: {:.4f} loss_DB: {:.4f}".format(loss_D_A, loss_D_B)]
                    with open('logs.txt', 'a') as f:
                        for t in text:
                            print(t)
                            f.write(t+'\n')
                        
                
                # Progress report (http://localhost:8097)
                '''
                logger.log({'loss_G': loss_G, 'loss_G_identity': (loss_identity_A + loss_identity_B), 'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
                            'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB), 'loss_D': (loss_D_A + loss_D_B)}, 
                            images={'real_A': real_A, 'real_B': real_B, 'fake_A': fake_A, 'fake_B': fake_B})
                '''

            
            # test
            # self.test(self.netG_A2B, self.netG_B2A, epoch)

            # Update learning rates
            self.lr_scheduler_G.step()
            self.lr_scheduler_D_A.step()
            self.lr_scheduler_D_B.step()

            # Save models checkpoints
            torch.save(self.netG_A2B.state_dict(), 'output/netG_A2B.pth')
            torch.save(self.netG_B2A.state_dict(), 'output/netG_B2A.pth')
            torch.save(self.netD_A.state_dict(), 'output/netD_A.pth')
            torch.save(self.netD_B.state_dict(), 'output/netD_B.pth')
            torch.save({'epoch': epoch, 
                        'lr':self.lr}, 'output/states.pth')
            
        ###################################

    def test(self, rootA, rootB, netG_A2B_path, netG_B2A_path, target_A, target_B):
        '''
        rootA = "../dataset/monet_field_data"
        rootB = "../dataset/landscape_test"
        '''

        ###### Definition of variables ######
        # Networks
        netG_A2B = Generator(self.input_nc, self.output_nc)
        netG_B2A = Generator(self.output_nc, self.input_nc)

        if self.cuda:
            netG_A2B.cuda()
            netG_B2A.cuda()

        # Load state dicts
        netG_A2B.load_state_dict(torch.load(netG_A2B_path))
        netG_B2A.load_state_dict(torch.load(netG_B2A_path))

        # Set model's test mode
        netG_A2B.eval()
        netG_B2A.eval()

        # Inputs & targets memory allocation
        Tensor = torch.cuda.FloatTensor if self.cuda else torch.Tensor
        input_A = Tensor(1, self.input_nc, self.size, self.size)
        input_B = Tensor(1, self.output_nc, self.size, self.size)

        # Dataset loader
        transforms_ = [ transforms.Resize([self.size, self.size], Image.BICUBIC),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
        dataloader = DataLoader(ImageDataset(rootA, rootB, transforms_=transforms_, unaligned=False), 
                                batch_size=1, shuffle=False, num_workers=self.n_cpu)

        ###### Testing######

        # Create output dirs if they don't exist
        if not os.path.exists(target_A):
            os.makedirs(target_A)
        if not os.path.exists(target_B):
            os.makedirs(target_B)

        for i, batch in enumerate(dataloader):
            # Set model input
            real_A = Variable(input_A.copy_(batch['A']))
            real_B = Variable(input_B.copy_(batch['B']))

            # Save image files
            save_image(0.5*(real_A+1.0), target_A+'/real{}.png'.format(i))
            save_image(0.5*(real_B+1.0), target_B+'/real{}.png'.format(i))

            # Generate output
            fake_B = 0.5*(netG_A2B(real_A).data + 1.0)
            fake_A = 0.5*(netG_B2A(real_B).data + 1.0)

            # Save image files
            save_image(fake_A, target_A+'/fake{}.png'.format(i))
            save_image(fake_B, target_B+'/fake{}.png'.format(i))

            sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader)))

        sys.stdout.write('\n')
        ###################################