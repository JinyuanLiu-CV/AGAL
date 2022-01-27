import numpy as np
import torch
import os
from torch.autograd import Variable
from .base_model import BaseModel
from . import networks

def latent2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].detach().cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    image_numpy = np.maximum(image_numpy, 0)
    image_numpy = np.minimum(image_numpy, 255)
    return image_numpy.astype(imtype)


class SingleModel(BaseModel):
    def name(self):
        return 'SingleGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)


        nb = opt.batchSize
        size = opt.fineSize
        self.opt = opt
        self.input_A = self.Tensor(nb, opt.input_nc, 600, 400)
        self.input_B = self.Tensor(nb, opt.output_nc, 600, 400)
        self.input_C = self.Tensor(nb, opt.output_nc, 600, 400)
        self.input_C_gray = self.Tensor(nb, opt.output_nc, 600, 400)

        skip = True if opt.skip > 0 else False
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids,
                                        skip=skip, opt=opt)
        self.d_net = networks.define_d(self.gpu_ids, skip=skip, opt=opt)
        self.h_net = networks.define_H(self.gpu_ids, skip=skip, opt=opt)


        print("---is not train----")
        which_epoch = opt.which_epoch
        print("---model is loaded---")
        self.load_network(self.netG_A, 'G_A', which_epoch)
        self.load_network(self.d_net, 'G_V', which_epoch)
        self.load_network(self.h_net, 'G_H', which_epoch)

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG_A)
        networks.print_network(self.d_net)
        networks.print_network(self.h_net)
        self.netG_A.eval()
        self.d_net.eval()
        self.h_net.eval()

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A']
        input_B = input['B']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']



    def predict(self):
        nb = self.opt.batchSize
        size = self.opt.fineSize
        self.real_A = Variable(self.input_A, volatile=True)
        self.real_B = Variable(self.input_B, volatile=True)

        self.a1 = self.d_net.forward(self.real_A)
        self.a2 = self.netG_A.forward(self.real_B)
        self.output1 = self.a1 * self.real_A + self.a2 * self.real_B
        self.latent, self.output ,self.edge = self.h_net.forward(self.output1)
        output = latent2im(self.output.data)
        return output

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    
