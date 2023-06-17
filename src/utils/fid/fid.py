import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from .inception import InceptionV3
from torch.nn import functional as F
from scipy import linalg

def _compute_FID(mu1, mu2, sigma1, sigma2,eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
                'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    FID = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    return FID

def compute_act_mean_std(act):
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma

class FID:
    def __init__(self, leng, net='alex'):
        n_act = 2048
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[n_act]
        self.inception_model = InceptionV3([block_idx]).to('cuda')
        self.inception_model.eval()
        self.batch_size=1
        act1 = np.zeros((leng, n_act))
        act2 = np.zeros((leng, n_act))
        self.act = [act1, act2]

    
    def get_pred(self,x):
        # if resize:
        #     x = up(x)
        x = self.inception_model2(x)
        return F.softmax(x, dim=1).data.cpu().numpy()

    def get_activations(self,x):
        x = self.inception_model(x)[0]
        return x.cpu().data.numpy().reshape(self.batch_size, -1)
    

    def caculate_fid(self,im1,im2,i):
        im1 = torch.from_numpy(im1.astype(np.float32)).to('cuda')
        im2 = torch.from_numpy(im2.astype(np.float32)).to('cuda')
        im1.unsqueeze_(0)
        im2.unsqueeze_(0)
        batch_size_i =1
        activation = self.get_activations(im1)
        self.act[0][i * self.batch_size:i * self.batch_size + batch_size_i] = activation
        activation2 = self.get_activations(im2)
        self.act[1][i * self.batch_size:i * self.batch_size + batch_size_i] = activation2

    def all_fid(self):
        mu_act1, sigma_act1 = compute_act_mean_std(self.act[0])
        mu_act2, sigma_act2 = compute_act_mean_std(self.act[1])
        FID = _compute_FID(mu_act1, mu_act2, sigma_act1, sigma_act2)
        return FID