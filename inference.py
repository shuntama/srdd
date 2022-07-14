import os
import glob
import importlib
from tqdm import tqdm
import numpy as np
import skimage.io as io
import skimage.color as color
import torch
import torch.nn.functional as F
import option
import time


def im2tensor(im):
    np_t = np.ascontiguousarray(im.transpose((2, 0, 1)))
    tensor = torch.from_numpy(np_t).float()
    return tensor


@torch.no_grad()
def main(opt):
    os.makedirs(opt.save_root, exist_ok=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = importlib.import_module("model.{}".format(opt.model.lower()))
    net = module.Net(opt).to(dev)

    state_dict = torch.load(opt.pretrain, map_location=lambda storage, loc: storage)
    net.load_state_dict(state_dict)

    net.eval()

    paths = sorted(glob.glob(os.path.join(opt.dataset_root, "*.png")))
    for path in tqdm(paths):
        name = path.split("/")[-1]

        LR = color.gray2rgb(io.imread(path))
        LR = im2tensor(LR).unsqueeze(0).to(dev)
        #LR = torch.randn(1, 3, 256, 256).to(dev)  # runtime test
        if opt.model != "SRDD":
            LR = F.interpolate(LR, scale_factor=opt.scale, mode="nearest")

        if opt.model != "SRDD":
            #torch.cuda.synchronize()  # runtime test
            #start = time.time()
            #for _ in range(100):
            SR = net(LR).detach()
            #torch.cuda.synchronize()
            #elapsed_time = time.time() - start
            #print(elapsed_time/100, 'sec.')
            #exit()
        else:
            mod = 8
            _, _, h, w = LR.size()
            w_pad, h_pad = mod - w%mod, mod - h%mod
            if w_pad == mod: w_pad = 0
            if h_pad == mod: h_pad = 0
            LR = torch.nn.functional.pad(LR, (w_pad, 0, h_pad, 0), mode='reflect')

            if path == paths[0]:
                _, stored_dict, stored_code = net(LR[:, :, :mod, :mod])
                stored_dict = stored_dict.detach().repeat(1, 1, 512, 512)
                stored_code = stored_code.detach().repeat(1, 1, 512, 512)
            #torch.cuda.synchronize()  # runtime test
            #start = time.time()
            #for _ in range(100):
            h = LR.size()[2]
            w = LR.size()[3]
            SR, _, _ = net(LR, stored_dict[:, :, :h*opt.scale, :w*opt.scale], stored_code[:, :, :h, :w])
            #torch.cuda.synchronize()
            #elapsed_time = time.time() - start
            #print(elapsed_time/100, 'sec.')
            #exit()

            SR  = SR[:, :, h_pad*opt.scale:, w_pad*opt.scale:]

        SR = SR[0].clamp(0, 255).round().cpu().byte().permute(1, 2, 0).numpy()

        save_path = os.path.join(opt.save_root, name)
        io.imsave(save_path, SR)

        if opt.model == "SRDD" and path == paths[0]:
            d = stored_dict[:, :, :opt.scale, :opt.scale]
            d = F.pad(d, (1, 1, 1, 1), "constant", 1)
            d = d.reshape(1, 1, -1, opt.scale+2)
            d = 255*(d + 1)/2
            d = d[0].clamp(0, 255).round().cpu().byte().permute(1, 2, 0).numpy()
            save_path = os.path.join(opt.save_root, 'atoms.png')
            io.imsave(save_path, d)


if __name__ == "__main__":
    opt = option.get_option()
    main(opt)
