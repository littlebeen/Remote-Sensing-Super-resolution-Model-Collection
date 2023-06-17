import os
from data import srdata
from data import common
import glob

class DIV2K(srdata.SRData):
    def __init__(self, args,name='DIV2K', train=True, benchmark=False):
        data_range = [r.split('-') for r in '1-800/801-810'.split('/')]
        if train:
            data_range = data_range[0]
        else:
            if args.test_only and len(data_range) == 1:
                data_range = data_range[0]
            else:
                data_range = data_range[1]

        self.begin, self.end = list(map(lambda x: int(x), data_range))
        super(DIV2K, self).__init__(
            args, name, train=train, benchmark=benchmark
        )

    def __getitem__(self, idx):
        lr, hr, filename = self._load_file(idx)
        pair = self.get_patch(lr, hr)
        #pair = common.set_channel(*pair, n_channels=self.args.n_colors)
        lr,hr = common.np2Tensor(*pair, rgb_range=self.args.rgb_range)
        if self.split=='test':
            return {"lr": lr, "hr": hr, "filename":filename}
        else:
            return lr, hr, filename  #cwh

    def _scan(self):
        names_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '*' + '.png'))
        )
        names_lr = []
        for f in names_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            names_lr.append(os.path.join(
                self.dir_lr, '{}x{}{}'.format(
                    filename, self.args.scale , '.png'
                )
            ))
        names_hr = names_hr[self.begin - 1:self.end]
        names_lr = names_lr[self.begin - 1:self.end]
        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        super(DIV2K, self)._set_filesystem(dir_data)
        self.dir_hr = os.path.join(self.apath, 'DIV2K_train_HR')
        self.dir_lr = os.path.join(self.apath, 'DIV2K_train_LR_bicubic/X3')
        self.dir_test_hr = os.path.join(self.apath, 'DIV2K_train_HR')
        self.dir_test_lr = os.path.join(self.apath, 'DIV2K_train_LR_bicubic/X3')
        if self.input_large: self.dir_lr += 'L'

