## input: event, blur and last sharp
## output: delta L

import torch
from .models import BaseModel
from networks.networks import NetworksFactory
from utils import cv_utils, criterion
from collections import OrderedDict
import os
from time import time
import torch.nn.functional as F
import cv2
import numpy as np

class Model(BaseModel):
    def __init__(self, opt):
        super(Model, self).__init__(opt)
        self._name = 'DeblurOnly'
        self.eventbins_between_frames = self._opt.eventbins_between_frames
        self.inter_num = self._opt.inter_num
        self._init_create_networks()
        if self._opt.is_train:
            self._init_train_vars()
        self._init_train_inputs()
        self._init_losses()
        self.seqcount = 0

    def _init_create_networks(self):

        self._G = NetworksFactory.get_by_name('DeblurOnly',
                                              eventbins_between_frames = self.eventbins_between_frames,
                                              if_RGB=self._opt.channel, inter_num=self.inter_num)

        self._load_network(self._G, self._opt.load_G)
        self._G.cuda()

        n_parameters = sum(p.numel() for p in self._G.parameters() if p.requires_grad)
        print('number of parameters of G:', n_parameters)

    def _init_train_vars(self):
        self._optimizer_G = torch.optim.AdamW([{'params': self._G.parameters(), 'lr': self._opt.lr_G}],
                                              betas=[self._opt.G_adam_b1, self._opt.G_adam_b2],
                                              weight_decay = 1e-4)
        lr_steps = [int(v) for v in self._opt.lr_steps.strip().split(',')]
        self._lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self._optimizer_G, lr_steps)

    def _init_train_inputs(self):
        self._input_gt = self._Tensor()
        self._input_blurred = self._Tensor()
        self._input_event = self._Tensor()

    def _init_losses(self):
        if self._opt.VerifyL2:
            self.criterion_L1 = torch.nn.MSELoss().cuda()
        else:
            self.criterion_L1 = torch.nn.L1Loss().cuda()

    def set_input(self, input):
        self._input_gt.resize_(input['gts'].size()).copy_(input['gts'])
        self._input_blurred.resize_(input['blurred'].size()).copy_(input['blurred'])
        self._input_event.resize_(input['events'].size()).copy_(input['events'])


        self._input_gt = self._input_gt.cuda()
        self._input_blurred = self._input_blurred.cuda()
        self._input_event = self._input_event.cuda()
        self.dataname = input['dataname'][0]
        self.imgcount = 0

    def set_train(self):
        self._G.train()
        self._is_train = True

    def set_eval(self):
        self._G.eval()
        self._is_train = False

    def forward(self, keep_data_for_visuals=False, isTrain=True, step=1):
        self._input_event[:,0::2,:,:] = self._input_event[:,0::2,:,:] * -1.
        self.output = self._G(self._input_blurred, self._input_event)
        self.loss_l1 = self.criterion_L1(self.output, self._input_gt)

        if self._is_train:
            self._optimizer_G.zero_grad()
            self.loss_l1.backward()
            self._optimizer_G.step()

        if keep_data_for_visuals:
            self.im_output_gt1 = cv_utils.tensor2im(self._input_gt, if_RGB=self._opt.channel)
            self.im_input1 = cv_utils.tensor2im(self._input_blurred, if_RGB=self._opt.channel)
            self.im_output1 = cv_utils.tensor2im(self.output, if_RGB=self._opt.channel)

            # self.im_events = cv_utils.events2im(self._input_event[:,:self.eventbins_between_frames,:,:])
            # self.im_events = self.im_events.transpose(2,0,1)

            self.im_events = cv_utils.tensor2im(torch.cat([torch.abs(self._input_event[0,:2]), torch.zeros_like(self._input_event[0,:1])], dim=0))

    def save(self, label):
        self._save_network(self._G, 'G', label)

    def update_learning_rate(self):
        org_lr = self._optimizer_G.param_groups[0]["lr"]
        self._lr_scheduler.step()
        cur_lr = self._optimizer_G.param_groups[0]["lr"]
        print('update learning rate: %f -> %f' %  (org_lr, cur_lr))

    def get_current_scalars(self):
        ## evaluate:
        final_psnr_im = criterion.psnr(self.output, self._input_gt)
        final_ssim_im = criterion.ssim(self.output, self._input_gt)

        if self._is_train:
            loss_dict = OrderedDict([('L1', self.loss_l1.item()),
                                     ('psnr_im', final_psnr_im.item()),
                                     ('ssim_im', final_ssim_im.item()),
                                     ('lr_G', self._optimizer_G.param_groups[0]["lr"])])
        else:
            
            loss_dict = OrderedDict([('psnr_im', final_psnr_im.item()),
                                     ('ssim_im', final_ssim_im.item()),
                                     ('L1', self.loss_l1.item())])
        return loss_dict

    def get_current_visuals(self):
        visuals = OrderedDict()

        visuals['output_gt1'] = self.im_output_gt1
        visuals['output_1'] = self.im_output1
        visuals['input_1'] = self.im_input1

        visuals['events'] = self.im_events

        return visuals

    def save_output(self, output_dir, output_gt, output_blur):
        print(self.dataname)
        output_path = os.path.join(output_dir, self.dataname[:-4]+'_output.png')
        cv_utils.debug_save_tensor(self.output[0], output_path, rela=False, rgb=True)
        if output_gt:
            output_path = os.path.join(output_dir, self.dataname[:-4]+'_gt.png')
            cv_utils.debug_save_tensor(self._input_gt[0], output_path, rela=False, rgb=True)

        if output_blur:
            output_path = os.path.join(output_dir, self.dataname[:-4]+'_blur.png')
            cv_utils.debug_save_tensor(self._input_blurred[0], output_path, rela=False, rgb=True)
