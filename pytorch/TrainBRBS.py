import os
from os.path import join
import SimpleITK as sitk
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np

from models.UNet import UNet_reg, UNet_seg
from utils.STN import SpatialTransformer, Re_SpatialTransformer
from utils.augmentation import SpatialTransform
from utils.dataloader_heart_train import DatasetFromFolder3D as DatasetFromFolder3D_train
from utils.dataloader_heart_test_reg import DatasetFromFolder3D as DatasetFromFolder3D_test_reg
from utils.dataloader_heart_test_seg import DatasetFromFolder3D as DatasetFromFolder3D_test_seg
from utils.losses import gradient_loss, ncc_loss, MSE, dice_loss
from utils.utils import AverageMeter

class BRBS(object):
    def __init__(self, k=0,
                 n_channels=1,
                 n_classes=8,
                 lr=1e-4,
                 epoches=200,
                 iters=200,
                 batch_size=1,
                 is_aug=True,
                 shot=5,
                 labeled_dir='',
                 unlabeled_dir='',
                 checkpoint_dir='weights',
                 result_dir='results',
                 model_name='BRBS'):
        super(BRBS, self).__init__()
        # initialize parameters
        self.k = k
        self.n_classes = n_classes
        self.epoches = epoches
        self.iters = iters
        self.lr = lr
        self.is_aug = is_aug
        self.shot = shot

        self.labeled_dir = labeled_dir
        self.unlabeled_dir = unlabeled_dir

        self.results_dir = result_dir
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name

        # tools
        self.stn = SpatialTransformer() # Spatial Transformer
        self.rstn = Re_SpatialTransformer() # Spatial Transformer-inverse
        self.softmax = nn.Softmax(dim=1)

        # data augmentation
        self.spatial_aug = SpatialTransform(do_rotation=True,
                                            angle_x=(-np.pi / 9, np.pi / 9),
                                            angle_y=(-np.pi / 9, np.pi / 9),
                                            angle_z=(-np.pi / 9, np.pi / 9),
                                            do_scale=True,
                                            scale=(0.75, 1.25))

        # initialize networks
        self.Reger = UNet_reg(n_channels=n_channels)
        self.Seger = UNet_seg(n_channels=n_channels, n_classes=n_classes)

        if torch.cuda.is_available():
            self.Reger = self.Reger.cuda()
            self.Seger = self.Seger.cuda()

        # initialize optimizer
        self.optR = torch.optim.Adam(self.Reger.parameters(), lr=lr)
        self.optS = torch.optim.Adam(self.Seger.parameters(), lr=lr)

        # initialize dataloader
        train_dataset = DatasetFromFolder3D_train(self.labeled_dir, self.unlabeled_dir, self.n_classes, shot=self.shot)
        self.dataloader_train = DataLoader(train_dataset, batch_size=batch_size)
        test_dataset_seg = DatasetFromFolder3D_test_seg(self.labeled_dir, self.n_classes, shot=self.shot)
        self.dataloader_test_seg = DataLoader(test_dataset_seg, batch_size=batch_size)
        test_dataset_reg = DatasetFromFolder3D_test_reg(self.labeled_dir, self.n_classes, shot=self.shot)
        self.dataloader_test_reg = DataLoader(test_dataset_reg, batch_size=batch_size)

        # define loss
            # losses in registration
        self.L_sim = ncc_loss
        self.L_smooth = gradient_loss
        self.L_SeC = dice_loss
        self.L_I = MSE
            # losses in segmentation
        self.L_seg = dice_loss
        self.L_Mix = MSE

        # define loss log
        self.L_smooth_log = AverageMeter(name='L_smooth')
        self.L_sim_log = AverageMeter(name='L_sim')
        self.L_i_log = AverageMeter(name='L_I')
        self.L_SeC_log = AverageMeter(name='L_SeC')

        self.L_seg_log = AverageMeter(name='L_seg')
        self.L_mix_log = AverageMeter(name='L_Mix')

    def train_iterator(self, labed_img, labed_lab, unlabed_img1, unlabed_img2):
        # train Reger
        for p in self.Seger.parameters():  # reset requires_grad
            p.requires_grad = False  # they are set to False below in Seger update
        for p in self.Reger.parameters():  # reset requires_grad             -
            p.requires_grad = True  # they are set to True below in Reger update

        # random construct training data pairs
        rand = np.random.randint(low=0, high=3)
        if rand == 0:
            img1 = labed_img
            lab1 = labed_lab
            img2 = unlabed_img1
            lab2 = None
        elif rand == 1:
            img1 = labed_img
            lab1 = labed_lab
            img2 = unlabed_img2
            lab2 = None
        else:
            img1 = unlabed_img2
            lab1 = None
            img2 = unlabed_img1
            lab2 = None

        rand = np.random.randint(low=0, high=2)
        if rand == 0:
            tmp = img1
            img1 = img2
            img2 = tmp
            tmp = lab1
            lab1 = lab2
            lab2 = tmp

        # forward deformation
        w_1_to_2, w_2_to_1, w_label_1_to_2, w_label_2_to_1, flow = self.Reger(img1, img2, lab1, lab2)

        # inverse deformation
        i_w_2_to_1, i_w_1_to_2, i_w_label_2_to_1, i_w_label_1_to_2, i_flow = self.Reger(img2, img1, lab2, lab1)

        # calculate loss
        loss_smooth = self.L_smooth(flow) + self.L_smooth(i_flow)   # smooth loss
        self.L_smooth_log.update(loss_smooth.data, labed_img.size(0))

        loss_sim = self.L_sim(w_1_to_2, img2) + self.L_sim(i_w_2_to_1, img1)    # similarity loss
        self.L_sim_log.update(loss_sim.data, labed_img.size(0))

        loss_i = self.L_I(-1*self.stn(flow, flow), i_flow)   # inverse loss
        self.L_i_log.update(loss_i.data, labed_img.size(0))

        # calculate SeC loss
        if lab1 is not None and lab2 is not None:
            w_1_l = self.stn(lab1, flow)
            w_2_l = self.stn(lab2, i_flow)
            loss_sec = self.L_SeC(w_1_l, lab2) + self.L_SeC(w_2_l, lab1)
        elif lab1 is not None and lab2 is None:
            w_1_l = self.stn(lab1, flow)
            s_2 = self.softmax(self.Seger(img2)).detach()
            w_2_s = self.stn(s_2, i_flow)
            loss_sec = self.L_SeC(w_1_l, s_2) + self.L_SeC(w_2_s, lab1)
        elif lab1 is None and lab2 is not None:
            s_1 = self.softmax(self.Seger(img1)).detach()
            w_1_s = self.stn(s_1, flow)
            w_2_l = self.stn(lab2, i_flow)
            loss_sec = self.L_SeC(w_1_s, lab2) + self.L_SeC(w_2_l, s_1)
        else:
            s_1 = self.softmax(self.Seger(img1)).detach()
            w_1_s = self.stn(s_1, flow)
            s_2 = self.softmax(self.Seger(img2)).detach()
            w_2_s = self.stn(s_2, i_flow)
            loss_sec = self.L_SeC(w_1_s, s_2) + self.L_SeC(w_2_s, s_1)

        self.L_SeC_log.update(loss_sec.data, labed_img.size(0))

        loss_Reg = loss_smooth + loss_sim + 100*loss_sec + 0.1*loss_i

        loss_Reg.backward()
        self.optR.step()
        self.Reger.zero_grad()
        self.optR.zero_grad()

        # train Seger
        for p in self.Seger.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to True below in Seger update
        for p in self.Reger.parameters():  # reset requires_grad             -
            p.requires_grad = False  # they are set to False below in Reger update

        # S3P
        with torch.no_grad():
            w_u_to_l, w_l_to_u, w_label_u_to_l, w_label_l_to_u, flow = self.Reger(unlabed_img1, labed_img, None, labed_lab) # unlabeled image1 --> labeled image
            w_l_to_u2, w_u2_to_l, w_label_l_to_u2, w_label_u2_to_l, flow2 = self.Reger(labed_img, unlabed_img2, labed_lab, None)    # labeled image --> unlabeled image2
        beta = np.random.beta(0.3, 0.3)
        alpha = np.random.beta(0.3, 0.3)
        sty = beta * (w_u_to_l - labed_img)
        spa = alpha * flow2
        new_img = self.stn(labed_img + sty, spa)
        new_lab = self.stn(labed_lab, spa)


        # MCC
        w_u_to_u2 = self.stn(w_u_to_l, flow2)
        s_w_u_to_u2 = self.Seger(w_u_to_u2)
        s_unlabed_img2 = self.Seger(unlabed_img2)

        gamma = np.random.beta(0.3, 0.3)
        x_mix = gamma * w_u_to_u2 + (1 - gamma) * unlabed_img2
        s_mix = self.Seger(x_mix)
        y_mix = gamma * s_w_u_to_u2 + (1 - gamma) * s_unlabed_img2
        loss_mix = self.L_Mix(s_mix, y_mix)
        self.L_mix_log.update(loss_mix.data, labed_img.size(0))

        s_gen = self.Seger(new_img)

        loss_seg = self.L_seg(self.softmax(s_gen), new_lab)
        self.L_seg_log.update(loss_seg.data, labed_img.size(0))

        loss_Seg = 0.01*loss_mix + loss_seg
        loss_Seg.backward()
        self.optS.step()
        self.Seger.zero_grad()
        self.optS.zero_grad()

    def train_epoch(self, epoch):
        self.Seger.train()
        self.Reger.train()
        for i in range(self.iters):
            labed_img, labed_lab, unlabed_img1, unlabed_img2 = next(self.dataloader_train.__iter__())

            if torch.cuda.is_available():
                labed_img = labed_img.cuda()
                labed_lab = labed_lab.cuda()
                unlabed_img1 = unlabed_img1.cuda()
                unlabed_img2 = unlabed_img2.cuda()

            if self.is_aug:
                code_spa = self.spatial_aug.rand_coords(labed_img.shape[2:])
                labed_img = self.spatial_aug.augment_spatial(labed_img, code_spa)
                labed_lab = self.spatial_aug.augment_spatial(labed_lab, code_spa, mode='nearest')
                unlabed_img1 = self.spatial_aug.augment_spatial(unlabed_img1, code_spa)
                unlabed_img2 = self.spatial_aug.augment_spatial(unlabed_img2, code_spa)

            self.train_iterator(labed_img, labed_lab, unlabed_img1, unlabed_img2)
            res = '\t'.join(['Epoch: [%d/%d]' % (epoch + 1, self.epoches),
                             'Iter: [%d/%d]' % (i + 1, self.iters),
                             self.L_smooth_log.__str__(),
                             self.L_sim_log.__str__(),
                             self.L_SeC_log.__str__(),
                             self.L_i_log.__str__(),
                             self.L_mix_log.__str__(),
                             self.L_seg_log.__str__()])
            print(res)


    def test_iterator_seg(self, mi):
        with torch.no_grad():
            # Seg
            s_m = self.Seger(mi)
        return s_m

    def test_iterator_reg(self, mi, fi, ml=None, fl=None):
        with torch.no_grad():
            # Reg
            w_m_to_f, w_f_to_m, w_label_m_to_f, w_label_f_to_m, flow = self.Reger(mi, fi, ml, fl)

        return w_m_to_f, w_label_m_to_f, flow

    def test(self):
        self.Seger.eval()
        self.Reger.eval()
        for i, (mi, ml, name) in enumerate(self.dataloader_test_seg):
            name = name[0]
            if torch.cuda.is_available():
                mi = mi.cuda()
            s_m = self.test_iterator_seg(mi)
            s_m = np.argmax(s_m.data.cpu().numpy()[0], axis=0)
            s_m = s_m.astype(np.int8)
            if not os.path.exists(join(self.results_dir, self.model_name, 'seg')):
                os.makedirs(join(self.results_dir, self.model_name, 'seg'))

            s_m = sitk.GetImageFromArray(s_m)
            sitk.WriteImage(s_m, join(self.results_dir, self.model_name, 'seg', name[:-4]+'.nii'))
            print(name[:-4]+'.nii')

        for i, (mi, ml, fi, fl, name1, name2) in enumerate(self.dataloader_test_reg):
            name1 = name1[0]
            name2 = name2[0]
            if name1 is not name2:
                if torch.cuda.is_available():
                    mi = mi.cuda()
                    fi = fi.cuda()
                    ml = ml.cuda()
                    fl = fl.cuda()

                w_m_to_f, w_label_m_to_f, flow = self.test_iterator_reg(mi, fi, ml, fl)

                flow = flow.data.cpu().numpy()[0]
                w_m_to_f = w_m_to_f.data.cpu().numpy()[0, 0]
                w_label_m_to_f = np.argmax(w_label_m_to_f.data.cpu().numpy()[0], axis=0)

                flow = flow.astype(np.float32)
                w_m_to_f = w_m_to_f.astype(np.float32)
                w_label_m_to_f = w_label_m_to_f.astype(np.int8)

                if not os.path.exists(join(self.results_dir, self.model_name, 'flow')):
                    os.makedirs(join(self.results_dir, self.model_name, 'flow'))
                if not os.path.exists(join(self.results_dir, self.model_name, 'w_m_to_f')):
                    os.makedirs(join(self.results_dir, self.model_name, 'w_m_to_f'))
                if not os.path.exists(join(self.results_dir, self.model_name, 'w_label_m_to_f')):
                    os.makedirs(join(self.results_dir, self.model_name, 'w_label_m_to_f'))

                w_m_to_f = sitk.GetImageFromArray(w_m_to_f)
                sitk.WriteImage(w_m_to_f, join(self.results_dir, self.model_name, 'w_m_to_f', name2[:-4]+'_'+name1[:-4]+'.nii'))
                w_label_m_to_f = sitk.GetImageFromArray(w_label_m_to_f)
                sitk.WriteImage(w_label_m_to_f, join(self.results_dir, self.model_name, 'w_label_m_to_f', name2[:-4]+'_'+name1[:-4]+'.nii'))
                flow = sitk.GetImageFromArray(flow)
                sitk.WriteImage(flow, join(self.results_dir, self.model_name, 'flow', name2[:-4]+'_'+name1[:-4]+'.nii'))
                print(name2[:-4]+'_'+name1[:-4]+'.nii')

    def checkpoint(self, epoch, k):
        torch.save(self.Seger.state_dict(),
                   '{0}/{1}_epoch_{2}.pth'.format(self.checkpoint_dir, 'Seger_'+self.model_name, epoch+k),
                   _use_new_zipfile_serialization=False)
        torch.save(self.Reger.state_dict(),
                   '{0}/{1}_epoch_{2}.pth'.format(self.checkpoint_dir, 'Reger_'+self.model_name, epoch+k),
                   _use_new_zipfile_serialization=False)

    def load(self):
        self.Reger.load_state_dict(
            torch.load('{0}/{1}_epoch_{2}.pth'.format(self.checkpoint_dir, 'Reger_'+self.model_name, str(self.k))))
        self.Seger.load_state_dict(
            torch.load('{0}/{1}_epoch_{2}.pth'.format(self.checkpoint_dir, 'Seger_' + self.model_name, str(self.k))))

    def train(self):
        for epoch in range(self.epoches-self.k):
            self.L_smooth_log.reset()
            self.L_sim_log.reset()
            self.L_SeC_log.reset()
            self.L_i_log.reset()

            self.L_seg_log.reset()
            self.L_mix_log.reset()
            self.train_epoch(epoch+self.k)
            if epoch % 20 == 0:
                self.checkpoint(epoch, self.k)
        self.checkpoint(self.epoches-self.k, self.k)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    RSTNet = BRBS()
    RSTNet.train()
    RSTNet.test()



