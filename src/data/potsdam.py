import os
import glob
from .srdata import SRData
from .common import np2Tensor
from utils import imgproc
import numpy as np

class Potsdam(SRData):
    def __init__(self, args,name='Potsdam', train=True, benchmark=False):
        super(Potsdam, self).__init__(
            args, name, train=train, benchmark=benchmark
        )

    def _scan(self):
        if(self.train):
            names_hr = sorted(
                glob.glob(os.path.join(self.dir_hr, '*' + '.tif'))
            )
            names_lr = []
            for f in names_hr:
                filename, _ = os.path.splitext(os.path.basename(f))
                names_lr.append(os.path.join(
                    self.dir_lr, '{}{}'.format(
                        filename, '.tif'
                    )
                ))
        else:
            names_hr = sorted(
                glob.glob(os.path.join(self.dir_test_hr, '*' + '.tif'))
            )
            names_lr = []
            for f in names_hr:
                filename, _ = os.path.splitext(os.path.basename(f))
                names_lr.append(os.path.join(
                    self.dir_test_lr, '{}{}'.format(
                        filename, '.tif'
                    )
                ))
            names_hr.sort(key=lambda x:int(x[51:-4]))
            names_lr.sort(key=lambda x:int(x[51:-4]))
        return names_hr, names_lr

    def __getitem__(self, idx):
        lr, hr, filename = self._load_file(idx)  #whc           
        lr = imgproc.imresize(lr.astype(np.float32), 1/self.scale)
        lr_up = imgproc.imresize(lr.astype(np.float32), self.scale)
        lr,hr,lr_up = np2Tensor(*[lr,hr,lr_up], rgb_range=self.args.rgb_range) #归一化外加转成cwh
        return lr, hr, filename  #cwh
    

    def _set_filesystem(self, dir_data):
        super(Potsdam, self)._set_filesystem(dir_data)
        self.dir_hr = os.path.join(self.apath, 'train')
        self.dir_lr = os.path.join(self.apath, 'train')
        self.dir_test_hr = os.path.join(self.apath, 'test')
        self.dir_test_lr = os.path.join(self.apath, 'test') 


