import os
from decimal import Decimal
from utils import imgproc
import utility

import torch
import torch.nn.utils as utils
import matplotlib.pyplot as plt
from utils.fid.fid import FID
from utils.is_calcu.is_calcu import IS
from utils.lpips.lpips import LPIPS
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr1 
from skimage.metrics import structural_similarity as ssim1
from utils.zmerge import save_all
class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = int(args.scale)

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8
        #self.FID_c=FID(len(loader.loader_test))
        self.LPIPS_c=LPIPS()
        #self.IS_c=IS(len(loader.loader_test))
    def testtrain(self, is_train=True):
        best_psnr_index= 0
        best_ssim_index=0
        all_PSNR=[0]
        all_SSIM=[0]
        for e in range(self.args.epochs):
            if(is_train):
                self.train()
                psnr,ssim=self.testnew(all_PSNR[best_psnr_index])
            else:
                psnr,ssim=self.testsave(all_PSNR[best_psnr_index])
                break
            if(e==0):
                all_PSNR[0]=psnr
                all_SSIM[0]=ssim
            else:
                all_PSNR.append(psnr)
                all_SSIM.append(ssim)
            if psnr>all_PSNR[best_psnr_index]:
                best_psnr_index=e
            if ssim>all_SSIM[best_ssim_index]:
                best_ssim_index=e
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(13,7))
        axes.plot(all_PSNR, 'k--')
        plt.savefig(self.args.model+'_PSNR.png')
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(13,7))
        axes.plot(all_SSIM, 'k--')
        plt.savefig(self.args.model+'_SSIM.png')
        self.ckp.write_log('Best PSNR epoch{}:PSNR:{:.3f} SSIM:{:.5f}'.format(best_psnr_index+1,all_PSNR[best_psnr_index],all_SSIM[best_psnr_index]))
        self.ckp.write_log('Best SSIM epoch{}:PSNR:{:.3f} SSIM:{:.5f}'.format(best_ssim_index+1,all_PSNR[best_ssim_index],all_SSIM[best_ssim_index]))

    def train(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        # TEMP
        for batch, (lr, hr, _,) in enumerate(self.loader_train):
            psnr = 0
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(lr)
            loss = self.loss(sr, hr)
            loss.backward()

            sr = utility.quantize(sr, self.args.rgb_range) 
            psnr = utility.calc_psnr(
                sr, hr, self.scale, self.args.rgb_range)
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                  self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s PSNR:{:.3f}'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release(),
                    psnr))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()

    def testnew(self,best):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.model.eval()
        timer_test = utility.timer()
        psnr=0
        ssim=0
        lpips=0
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, (lr, hr, filename)  in enumerate(self.loader_test):
            lr, hr = self.prepare(lr,hr)
            sr = self.model(lr)
            sr_cacu = np.round(sr[0].cpu().numpy()*255).clip(min=0, max=255).astype(np.uint8)
            hr_cacu = np.round(hr[0].cpu().numpy()*255).clip(min=0, max=255).astype(np.uint8)
            
            # self.FID_c.caculate_fid(sr_cacu,hr_cacu,idx_data)
            # self.IS_c.caculate_IS(sr_cacu,idx_data)

            lpips+=self.LPIPS_c.caculate_lpips(sr_cacu, hr_cacu)

            sr_cacu = sr_cacu.transpose(1, 2, 0)
            hr_cacu = hr_cacu.transpose(1, 2, 0)
            psnr+=psnr1(sr_cacu, hr_cacu, data_range=255)
            ss, diff =ssim1(sr_cacu, hr_cacu, full=True, channel_axis=2, data_range=255)
            ssim+=ss
            sr = utility.quantize(sr, self.args.rgb_range)
            save_list = [sr]
            if self.args.save_gt:
                save_list.extend([lr, hr])
            if(self.args.data_train=='OLI2MSI'):
                if idx_data< 10 and self.args.save_results:   #OLI2MSI
                   self.ckp.save_results(self.args.data_train, filename, save_list, self.scale)
            if(self.args.data_train=='ALSAT'):
                #if idx_data in [1,2,3,4,58,59.60,61,340,341,342,343] and self.args.save_results:  #Alsat
                if filename[0] in ['special_HR_28','urban_HR_41'] and self.args.save_results:  #Alsat
                    self.ckp.save_results(self.args.data_train, filename, save_list, self.scale)
            if(self.args.data_train=='Potsdam'):
                if idx_data <10 and self.args.save_results:
                    self.ckp.save_results(self.args.data_train, filename, save_list, self.scale)
        # print('fid:'+str(round(self.FID_c.all_fid(),5)))
        # print('is:'+str(round(self.IS_c.all_IS(),5)))
        psnr /= idx_data+1
        ssim /= idx_data+1
        lpips /= idx_data+1
        self.ckp.write_log(
            '[{} x{}]\tPSNR: {:.3f} SSIM:{:.5f} lpips:{:.5f}'.format(
                self.args.data_train,
                self.scale,
                psnr,ssim,lpips[0][0][0][0]
            )
        )
        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=psnr>best)

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)
        return float(psnr),float(ssim)
    
    def testsave(self,best):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.model.eval()
        timer_test = utility.timer()
        psnr=0
        ssim=0
        lpips=0
        all_image=[]
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, (lr, hr, filename)  in enumerate(self.loader_test):
            lr, hr = self.prepare(lr,hr)
            sr = self.model(lr)
            sr_cacu = np.round(sr[0].cpu().numpy()*255).clip(min=0, max=255).astype(np.uint8)
            hr_cacu = np.round(hr[0].cpu().numpy()*255).clip(min=0, max=255).astype(np.uint8)
            lpips+=self.LPIPS_c.caculate_lpips(sr_cacu, hr_cacu)

            sr_cacu = sr_cacu.transpose(1, 2, 0)
            hr_cacu = hr_cacu.transpose(1, 2, 0)
            all_image.append(sr_cacu)
            psnr+=psnr1(sr_cacu, hr_cacu, data_range=255)
            ss, diff =ssim1(sr_cacu, hr_cacu, full=True, channel_axis=2, data_range=255)
            ssim+=ss
            sr = utility.quantize(sr, self.args.rgb_range)
        if(self.args.data_train=='Vaihingen' and len(self.loader_test)<350):
            save_all('../experiment/'+self.args.save+'/results-Vaihingen/all.tif',all_image)
        if(self.args.data_train=='Potsdam'):
            save_all('../experiment/'+self.args.save+'/results-Potsdam/all.tif',all_image,40)
        psnr /= idx_data+1
        ssim /= idx_data+1
        lpips /= idx_data+1
        self.ckp.write_log(
            '[{} x{}]\tPSNR: {:.3f} SSIM:{:.5f} lpips:{:.5f}'.format(
                self.args.data_train,
                self.scale,
                psnr,ssim,lpips[0][0][0][0]
            )
        )
        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)
        return float(psnr),float(ssim)
    
    def test(self,best):  #废弃
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.model.eval()
        timer_test = utility.timer()
        psnr=0
        ssim=0
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, (lr, hr, filename)  in enumerate(self.loader_test):
            lr, hr = self.prepare(lr,hr)
            sr = self.model(lr)
            sr = utility.quantize(sr, self.args.rgb_range)
            psnr += utility.calc_psnr(
                sr, hr, self.scale, self.args.rgb_range
            )
            ssim += utility.calc_ssim(sr,hr,self.args.rgb_range)          
            save_list = [sr]
            if self.args.save_gt:
                save_list.extend([lr, hr])
            if(self.args.data_train=='OLI2MSI'):
                if idx_data< 10 and self.args.save_results:   #OLI2MSI
                   self.ckp.save_results(self.args.data_train, filename, save_list, self.scale)
            if(self.args.data_train=='ALSAT'):
                #if idx_data in [1,2,3,4,58,59.60,61,340,341,342,343] and self.args.save_results:  #Alsat
                if filename[0] in ['special_HR_28','urban_HR_41'] and self.args.save_results:  #Alsat
                    self.ckp.save_results(self.args.data_train, filename, save_list, self.scale)
        psnr /= idx_data+1
        ssim /= idx_data+1
        self.ckp.write_log(
            '[{} x{}]\tPSNR: {:.3f} SSIM:{:.5f}'.format(
                self.args.data_train,
                self.scale,
                psnr,ssim
            )
        )
        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=psnr>best)

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)
        return float(psnr),float(ssim)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]