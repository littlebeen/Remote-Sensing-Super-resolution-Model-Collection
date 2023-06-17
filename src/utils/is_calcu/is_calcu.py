import numpy as np
import torch
from torch.nn import functional as F
from torchvision.models.inception import inception_v3
from scipy.stats import entropy
class IS:
    def __init__(self, leng, net='alex'):
        self.batch_size=1

        self.inception_model2 = inception_v3(pretrained=True, transform_input=False).to('cuda')
        self.inception_model2.eval()
        self.preds = np.zeros((leng, 1000))
        self.length= leng

    def get_pred(self,x):
        # if resize:
        #     x = up(x)
        x = self.inception_model2(x)
        return F.softmax(x, dim=1).data.cpu().numpy()
    
    def caculate_IS(self,im1,i):
        im1 = torch.from_numpy(im1.astype(np.float32)).to('cuda')
        im1.unsqueeze_(0)
        batch_size_i = 1
        self.preds[i * self.batch_size:i * self.batch_size + batch_size_i] = self.get_pred(im1)

    
    def all_IS(self):
        split_scores = []
        splits=10
        for k in range(splits):
            part = self.preds[k * (self.length // splits): (k + 1) * (self.length // splits), :] # split the whole data into several parts
            py = np.mean(part, axis=0)  # marginal probability
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]  # conditional probability
                scores.append(entropy(pyx, py))  # compute divergence
            split_scores.append(np.exp(np.mean(scores)))
        return np.mean(split_scores)

