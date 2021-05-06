import argparse
import os, sys

class BaseOpt():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--root_dir', type=str, default='.')
        self.parser.add_argument('--name', type=str, default='DeblurAndInterpolation')

        self.parser.add_argument('--toy', action='store_true', help='Toy experiment')

        # dataset
        self.parser.add_argument('--sequence_num', type=int, default=10,
                                 help='number of blurred frames in a sequence')
        self.parser.add_argument('--dataset_mode', type=str, default='Beijing',
                                 help='Beijing or Real')
        self.parser.add_argument('--event_name', type=str, default='EventBin3')
        self.parser.add_argument('--eventbins_between_frames', type=int, default=3,
                            help='number of event bins between frames')
        self.parser.add_argument('--scale', default=1, type=int, help='scale of events')
        self.parser.add_argument('--sharp_needed', action='store_true', help='Sharp intensity image is needed')
        self.parser.add_argument('--blur_needed', action='store_true', help='Sharp intensity image is needed')
        self.parser.add_argument('--eventHR_needed', action='store_true', help='Sharp intensity image is needed')
        self.parser.add_argument('--eventUP_needed', action='store_true', help='Sharp intensity image is needed')

        # dataloader
        self.parser.add_argument('--n_threads', default=4, type=int, help='# threads for data')

        # model
        self.parser.add_argument('--model', type=str, default='VerifyEventDenoise',
                                 help='model to run')
        self.parser.add_argument('--load_G', type=str, default='', help='path of the pretrained model G')
        self.parser.add_argument('--load_D', type=str, default='', help='path of the pretrained model D')
        self.parser.add_argument('--load_KF_Gain', type=str, default='', help='path of the pretrained model KF_Gain')

        self.parser.add_argument('--load_X_pre', type=str, default='', help='path of the pretrained model X_pre')
        self.parser.add_argument('--load_Z', type=str, default='', help='path of the pretrained model Z')

        self.parser.add_argument('--patch_size', default=8, type=int, help='patch size while predicting matrix A')
        self.parser.add_argument('--overlap', default=2, type=int, help='patch size while predicting matrix A')

        self.parser.add_argument('--n_resblocks', type=int, default=16,
                    help='number of residual blocks')
        self.parser.add_argument('--n_feats', type=int, default=64,
                            help='number of feature maps')
        self.parser.add_argument('--res_scale', type=float, default=1,
                            help='residual scaling')

        # other options
        self.parser.add_argument('--VerifyL2', action='store_true',
                                 help='set this option to verify the theory of the interpolation task.')
        self.parser.add_argument('--AdaptiveL2', action='store_true',
                                 help='use adaptive l2 loss.')
        self.parser.add_argument('--SynNet', action='store_true',
                                 help='set this option to synthesize a realistic dataset.')
        self.parser.add_argument('--DeconvNet', action='store_true',
                                 help='set this option to use deconvolution theory.')
        self.parser.add_argument('--OrderNet', action='store_true',
                                 help='set this option to use an specific order.')
        self.parser.add_argument('--SemiSuperNet', action='store_true',
                                 help='set this option to train in semi-supervised manner.')
        self.parser.add_argument('--cpu', action='store_true', help='Use CPU')
        self.parser.add_argument('--Gopro', action='store_true',
                                 help='set this option to train on Gopro dataset.')
        self.parser.add_argument('--GoproSingle', action='store_true',
                                 help='set this option to train on Gopro dataset with sequence of one.')
        self.parser.add_argument('--qualitative', action='store_true',
                                 help='set this option to qualitative evaluation.')
        self.parser.add_argument('--is_mc', action='store_true', help='Use Memory Cache')


        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        if not os.path.exists(self.opt.root_dir):
            os.makedirs(self.opt.root_dir)

        if self.opt.VerifyL2:
            self.opt.checkpoints_dir = self.opt.root_dir+'/checkpoints_verifyL2'

        if self.opt.Gopro:
            self.opt.inter_num = 10
            self.opt.checkpoints_dir = self.opt.checkpoints_dir + '_gopro'
            self.opt.channel = 3

        self.opt.is_train = self.is_train

        return self.opt
