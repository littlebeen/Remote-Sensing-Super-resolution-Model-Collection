import torch
import os
import utility
import data
import model
import loss
from option import args
from trainer import Trainer
import glob
from thop import profile
import time
import torch.optim as optim

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args) 


#os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
def main():
    global model
    if checkpoint.ok:                  
        loader = data.Data(args)
        _model = model.Model(args, checkpoint)
        args.test_only =True
        # for name, parms in _model.named_parameters():
        #     if 'layers.0.residual_group.blocks.0.attn.relative_position_bias_table' in name:
        #         print(parms)
            # print('%-50s' % name, '%-30s' % str(parms.shape), '%-10s' % str(parms.nelement()))
        #print('Total params: %.2fM' % (sum(p.numel() for p in _model.parameters())/1000000.0))
        #input = torch.randn(16, 3, 32, 32).cuda()
        #flops, params = profile(_model, inputs=(input, ))
        #print('Complexity: %.3fM' % (flops/1000000000), end=' GFLOPs\n')
        # torch.cuda.synchronize()
        # time_start = time.time()
        # predict = _model(input)
        # torch.cuda.synchronize()
        # time_end = time.time()
        # print('Speed: %.5f FPS\n' % (1/(time_end-time_start)))
        # optimizer = optim.SGD(_model.parameters(), lr=0.9, momentum=0.9, weight_decay=0.0005)
        # for _ in range(1000):
        #     optimizer.zero_grad()
        #     _model(input)

        _loss = loss.Loss(args, checkpoint) if not args.test_only else None
        t = Trainer(args, loader, _model, _loss, checkpoint)
        t.testtrain(False)
        checkpoint.done()

if __name__ == '__main__':
    main()
    