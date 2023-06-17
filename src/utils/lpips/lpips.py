import numpy as np
import torch
import lpips


class LPIPS:
    def __init__(self):
       self.loss_fn = lpips.LPIPS(net='alex', version=0.1)

    def caculate_lpips(self,img0,img1):
        im1=np.copy(img0)
        im2=np.copy(img1)
        im1=torch.from_numpy(im1.astype(np.float32))
        im2 = torch.from_numpy(im2.astype(np.float32))
        im1.unsqueeze_(0)
        im2.unsqueeze_(0)
        current_lpips_distance  = self.loss_fn.forward(im1, im2)
        return current_lpips_distance 

