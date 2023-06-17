import os
import glob
from data import srdata
from data import common
from utils import imgproc
import numpy as np

class OLI2MSI(srdata.SRData):
    def __init__(self, args,name='OLI2MSI', train=True, benchmark=False):
        super(OLI2MSI, self).__init__(
            args, name, train=train, benchmark=benchmark
        )

    def _scan(self):
        if(self.train):
            names_hr = sorted(
                glob.glob(os.path.join(self.dir_hr, '*' + '.png'))
            )
            names_lr = []
            for f in names_hr:
                filename, _ = os.path.splitext(os.path.basename(f))
                names_lr.append(os.path.join(
                    self.dir_lr, '{}{}'.format(
                        filename, '.png'
                    )
                ))
            return names_hr, names_lr
        else:
            names_hr = sorted(
                glob.glob(os.path.join(self.dir_test_hr, '*' + '.png'))
            )
            names_lr = []
            for f in names_hr:
                filename, _ = os.path.splitext(os.path.basename(f))
                names_lr.append(os.path.join(
                    self.dir_test_lr, '{}{}'.format(
                        filename, '.png'
                    )
                ))
            return names_hr, names_lr

    def __getitem__(self, idx):
        lr, hr, filename = self._load_file(idx)  #whc           
        lr,hr = self.get_patch(lr, hr)  #切片
        if(self.args.model=='vdsr' or self.args.model=='edrn' or self.args.model=='mae'):
            lr = imgproc.imresize(lr.astype(np.float32), self.scale)  #把lr放大n倍
        #pair=lr,hr  #关闭切片
        #pair = common.set_channel(*pair, n_channels=self.args.n_colors)  #统一通道，目前用不到
        lr,hr = common.np2Tensor(*[lr,hr], rgb_range=self.args.rgb_range) #归一化外加转成cwh
        return lr, hr, filename  #cwh
    

    def _set_filesystem(self, dir_data):
        super(OLI2MSI, self)._set_filesystem(dir_data)
        self.dir_hr = os.path.join(self.apath, 'train_hr')
        self.dir_lr = os.path.join(self.apath, 'train_lr')
        self.dir_test_hr = os.path.join(self.apath, 'test_hr')
        self.dir_test_lr = os.path.join(self.apath, 'test_lr')

