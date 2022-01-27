import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import PIL
import random
import torch
from pdb import set_trace as st
import numpy as np
import cv2
import time

class PairDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
        # self.dir_E = os.path.join(opt.dataroot, opt.phase + 'E')
        # self.dir_C = os.path.join(opt.dataroot, opt.phase + 'C')

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)
        # self.E_paths = make_dataset(self.dir_E)
        # self.C_paths = make_dataset(self.dir_C)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        # self.E_paths = sorted(self.E_paths)
        # self.C_paths = sorted(self.C_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        # self.E_size = len(self.E_paths)
        # self.C_size = len(self.C_paths)
        
        transform_list = []
        
        transform_list += [transforms.ToTensor()]
        # transform_list = [transforms.ToTensor()]

        self.transform = transforms.Compose(transform_list)
        # self.transform = get_transform(opt)

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[index % self.B_size]
        # C_path = self.C_paths[index % self.C_size]
        # E_path = self.E_paths[index % self.E_size]

        A_img = Image.open(A_path).convert('RGB')
        # C_img = Image.open(C_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        # E_img = Image.open(E_path).convert('RGB')
        # a = Image.open(A_path).convert('RGB')
        # B_img = Image.open(A_path.replace("low", "high").replace("A", "B")).convert('RGB')
        # b = Image.open(A_path.replace("low", "high").replace("A", "B")).convert('RGB')
        # C_img = Image.open(A_path.replace("low", "gt").replace("A", "C")).convert('RGB')
        #print("kkkkkkkk:",len(C_img.split()))

        A_img = self.transform(A_img)
        # a = self.transform(a)
        B_img = self.transform(B_img)
        # b = self.transform(b)
        # C_img = self.transform(C_img)
        # E_img = self.transform(E_img)

        # if self.opt.is_haze:
        #     A_img = -A_img
        #     B_img = -B_img

        w = A_img.size(2)
        h = A_img.size(1)
        w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

        # A_img = A_img[:, h_offset:h_offset + self.opt.fineSize,
               # w_offset:w_offset + self.opt.fineSize]
        # a =  a[:, h_offset:h_offset + self.opt.fineSize,
               # w_offset:w_offset + self.opt.fineSize]
        # B_img = B_img[:, h_offset:h_offset + self.opt.fineSize,
               # w_offset:w_offset + self.opt.fineSize]
        # b =  b[:, h_offset:h_offset + self.opt.fineSize,
               # w_offset:w_offset + self.opt.fineSize]
        # C_img = C_img[:, h_offset:h_offset + self.opt.fineSize,
                # w_offset:w_offset + self.opt.fineSize]


        if self.opt.resize_or_crop == 'no':
            r,g,b = A_img[0]+1, A_img[1]+1, A_img[2]+1
            A_gray = 1. - (0.299*r+0.587*g+0.114*b)/2.
            A_gray = torch.unsqueeze(A_gray, 0)
            input_img = A_img
            # A_gray = (1./A_gray)/255.
        else:
            
            
            # A_gray = (1./A_gray)/255.
            if (not self.opt.no_flip) and random.random() < 0.5:
                idx = [i for i in range(A_img.size(2) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                A_img = A_img.index_select(2, idx)
                a = a.index_select(2, idx)
                B_img = B_img.index_select(2, idx)
                b = b.index_select(2, idx)
                # C_img = C_img.index_select(2, idx)
            if (not self.opt.no_flip) and random.random() < 0.5:
                idx = [i for i in range(A_img.size(1) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                A_img = A_img.index_select(1, idx)
                a = a.index_select(1, idx)
                B_img = B_img.index_select(1, idx)
                b = b.index_select(1, idx)
                # C_img = C_img.index_select(1, idx)
            if (not self.opt.no_flip) and random.random() < 0.5:
                times = random.randint(self.opt.low_times,self.opt.high_times)/100.
                input_img = (A_img+1)/2./times
                input_img = input_img*2-1
            else:
                input_img = A_img
            r,g,b = input_img[0]+1, input_img[1]+1, input_img[2]+1
            A_gray = 1. - (0.299*r+0.587*g+0.114*b)/2.
            A_gray = torch.unsqueeze(A_gray, 0)
        # start_time = time.time()
        # h = A_img.shape[1]
        # w = A_img.shape[2]
        # ww = np.zeros((3,h, w), np.float32)
        #C = np.zeros((3,h, w), np.float32)
        #C[0] = C_img
        #C[1] = C_img
        #C[2] = C_img
        #C = torch.tensor(C)
        
        # V = np.zeros((h, w), np.float32)
        # print("V:")
        # print(torch.tensor(V).size())
        # r = np.asarray(A_img[0], np.float32)
        # g = np.asarray(A_img[1], np.float32)
        # b = np.asarray(A_img[2], np.float32)
        # print(r)
        # print("-------------------------------------")
        # for i in range(0, h):
            # for j in range(0, w):
                # mx = max((b[i, j], g[i, j], r[i, j]))
                # V[i, j] =np.float32(mx)
        # V = torch.tensor(V)
        # ww[0]=V
        # ww[1]=V
        # ww[2]=V
        # ww = torch.tensor(ww)
        #t = time.time() - start_time
        # print("tttttt",t)
        # print("ww",ww.size(),ww)
        return {'A': A_img, 'B': B_img, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return self.A_size

    def name(self):
        return 'PairDataset'
