from data import common
import os
from data import srdata
from utils import imgproc
import numpy as np
import torch

def handelycbcr(image):  #tensor cwh
    image = image.permute(1,2,0)*255  #to whc
    image=imgproc.ycbcr2rgb(image.cpu().numpy())
    image=torch.from_numpy(image) /255
    image = image.permute(2,0,1) #to cwh
    
    image=image.unsqueeze_(0) 
    image=image.to('cuda')
    return image  #cwh


class DataProcess():
    def __init__(self):
        super(DataProcess, self).__init__()

    def lr_process(lr):
        return lr[:,0:1,:,:]

    def sr_process(sr,lr): 
        sr = torch.cat([sr,lr[:,1:3,:,:]],dim=1)
        sr_tensor = handelycbcr(sr[0])
        lr_image = handelycbcr(lr[0])
        return sr_tensor, lr_image #cwh

class OLI2MSIY(srdata.SRData):
    def __init__(self, args, name='OLI2MSIY', train=True, benchmark=False):
        super(OLI2MSIY, self).__init__(
            args, name, train=train, benchmark=benchmark
        )

    def __getitem__(self, idx):
        lr, hr, filename = self._load_file(idx)  # 480/160 * 480/160 * 3 array未归一化
        lr,hr = self.get_patch(lr, hr)  #切片
        lr = imgproc.imresize(lr.astype(np.float32), self.scale)  #把lr放大n倍
        hr_image = imgproc.rgb2ycbcr(hr)
        lr_image = imgproc.rgb2ycbcr(lr)
        if self.split =='test':
            hr_image= imgproc.ycbcr2rgb(hr_image)
            lr_tensor,hr_tensor = common.np2Tensor(*[lr_image,hr_image], rgb_range=1) 
            return {"lr": lr_tensor, "hr": hr_tensor, "filename":filename}  #cwh
        else:
            lr_image, hr_image = imgproc.random_rotate(lr_image, hr_image, angles=[0, 90, 180, 270])
            lr_image, hr_image = imgproc.random_horizontally_flip(lr_image, hr_image, p=0.5)
            lr_image, hr_image = imgproc.random_vertically_flip(lr_image, hr_image, p=0.5)
            lr_tensor,hr_tensor = common.np2Tensor(*[lr_image, hr_image], rgb_range=self.args.rgb_range)
            return lr_tensor[0:1,:,:],hr_tensor[0:1,:,:], filename 


    def _set_filesystem(self, dir_data):
        super(OLI2MSIY, self)._set_filesystem(dir_data)
        self.dir_hr = os.path.join(self.apath, 'train_hr')
        self.dir_lr = os.path.join(self.apath, 'train_lr')
        self.dir_test_hr = os.path.join(self.apath, 'test_hr')
        self.dir_test_lr = os.path.join(self.apath, 'test_lr')
