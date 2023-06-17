import torch
import imageio
import matplotlib.pyplot as plt
import earthpy.plot as ep
from model.nlsn.nlsn import NLSN
from option import args
def save_image(img,img_path):
    #img=img.permute(2, 0, 1)
    img=img.detach().mul_(255).detach().numpy()
    ep.plot_rgb(img, rgb=[0,1,2], title=type, stretch=True)
    plt.savefig(img_path+".jpg")
    plt.close()

# model = NLSN(args)
# model.load_state_dict(torch.load('../model/model_best.pt'))

# lr = imageio.imread('../model/image/man.jpg')
# lr = torch.from_numpy(lr).permute(2,0,1)/255
# lr = lr.unsqueeze_(0)
# sr = model(lr)
# torch.save(sr,'sr.pt')
# save_image(sr[0], './srman')

sr = torch.load('sr.pt')
save_image(sr[0],'../srman')