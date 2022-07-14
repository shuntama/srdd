import os
import time
import skimage.io as io
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import augments
from data import generate_loader
from torch.utils.tensorboard import SummaryWriter
import lpips


class Solver():
    def __init__(self, module, opt):
        self.writer = SummaryWriter(log_dir="./logs/log")
        self.opt = opt

        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = module.Net(opt).to(self.dev)
        print("# params:", sum(map(lambda x: x.numel(), self.net.parameters())))

        if opt.pretrain:
            self.load(opt.pretrain)

        self.loss_fn = nn.L1Loss()
        self.loss_fn_lpips = lpips.LPIPS(net='alex').to(self.dev)

        if opt.model == "SRDD":
            self.optim = torch.optim.Adam(
                [
                    {"params": self.net.dict.parameters(), "lr": 0.005},
                    {"params": self.net.body.parameters()},
                    {"params": self.net.tail.parameters()},
                    {"params": self.net.comp.parameters()},
                    {"params": self.net.fuse.parameters()},
                    {"params": self.net.chsr.parameters()}
                ],
                opt.lr, betas=(0.9, 0.999), eps=1e-8
            )
        else:
            self.optim = torch.optim.Adam(
                self.net.parameters(), opt.lr,
                betas=(0.9, 0.999), eps=1e-8
            )
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optim, [1000*int(d) for d in opt.decay.split("-")],
            gamma=opt.gamma,
        )

        if not opt.test_only:
            self.train_loader = generate_loader("train", opt)
        self.test_loader = generate_loader("test", opt)

        self.t1, self.t2 = None, None
        self.best_psnr, self.best_step = 0, 0

    def fit(self):
        opt = self.opt

        self.t1 = time.time()
        for step in range(opt.max_steps):
            try:
                inputs = next(iters)
            except (UnboundLocalError, StopIteration):
                iters = iter(self.train_loader)
                inputs = next(iters)

            HR = inputs[0].to(self.dev)
            LR = inputs[1].to(self.dev)

            # match the resolution of (LR, HR) due to CutBlur
            if HR.size() != LR.size() and opt.model != "SRDD":
                scale = HR.size(2) // LR.size(2)
                LR = F.interpolate(LR, scale_factor=scale, mode="nearest")

            HR, LR, mask, aug = augments.apply_augment(
                HR, LR,
                opt.augs, opt.prob, opt.alpha,
                opt.aux_alpha, opt.aux_alpha, opt.mix_p
            )

            if opt.model != "SRDD":
                SR = self.net(LR)
            else:
                if step in [50000, 100000]:
                    self.optim.param_groups[0]['lr'] = self.optim.param_groups[0]['lr']/2
                    print('Reduce dictionary lr -> 0.5*lr.')

                if step == opt.max_steps - 40000:
                    self.freeze_dict(self.net)  # freeze dictionary
                    print('Freeze params in dictionary.')

                if step < 1000:
                    shuffle = True  # stabilize training
                else:
                    shuffle = False

                SR, _, _ = self.net(LR, shuffle=shuffle)

            if aug == "cutout":
                SR, HR = SR*mask, HR*mask

            loss = self.loss_fn(SR, HR)
            self.optim.zero_grad()
            loss.backward()

            if opt.gclip > 0:
                torch.nn.utils.clip_grad_value_(self.net.parameters(), opt.gclip)

            self.optim.step()
            self.scheduler.step()

            if (step+1) % opt.eval_steps == 0:
                self.summary_and_save(step)

    def summary_and_save(self, step):
        step, max_steps = (step+1)//1000, self.opt.max_steps//1000
        psnr, lpips = self.evaluate()
        self.t2 = time.time()

        if psnr >= self.best_psnr or step >= max_steps - 10:
            self.best_psnr, self.best_step = psnr, step
            self.save(step)

        curr_lr = self.scheduler.get_lr()
        eta = (self.t2-self.t1) * (max_steps-step) / 3600
        print("[{}K/{}K] {:.2f} (Best: {:.2f} @ {}K step) LR: {}, ETA: {:.1f} hours"
            .format(step, max_steps, psnr, self.best_psnr, self.best_step,
             curr_lr, eta))

        self.writer.add_scalar("valid psnr", psnr, step)
        self.writer.add_scalar("valid lpips", lpips, step)

        self.t1 = time.time()

    @torch.no_grad()
    def evaluate(self):
        opt = self.opt
        self.net.eval()

        if opt.save_result:
            save_root = os.path.join(opt.save_root, opt.dataset)
            os.makedirs(save_root, exist_ok=True)

        psnr = 0
        lpips = 0
        for i, inputs in enumerate(self.test_loader):
            HR = inputs[0].to(self.dev)
            LR = inputs[1].to(self.dev)
            eval_size = 256  # speed up evaluation

            # match the resolution of (LR, HR) due to CutBlur
            if HR.size() != LR.size() and opt.model != "SRDD":
                scale = HR.size(2) // LR.size(2)
                LR = F.interpolate(LR, scale_factor=scale, mode="nearest")
                LR = LR[:, :, :eval_size*opt.scale, :eval_size*opt.scale]
                HR = HR[:, :, :eval_size*opt.scale, :eval_size*opt.scale]

            if opt.model == "SRDD":
                LR = LR[:, :, :eval_size, :eval_size]
                HR = HR[:, :, :eval_size*opt.scale, :eval_size*opt.scale]

            if opt.model != "SRDD":
                SR = self.net(LR).detach()
            else:
                SR, _, _ = self.net(LR)
                SR = SR.detach()

            lpips_ = self.loss_fn_lpips.forward(2*(SR/255 - 0.5), 2*(HR/255 - 0.5))
            HR = HR[0].clamp(0, 255).round().cpu().byte().permute(1, 2, 0).numpy()
            SR = SR[0].clamp(0, 255).round().cpu().byte().permute(1, 2, 0).numpy()

            if opt.save_result:
                save_path = os.path.join(save_root, "{:04d}.png".format(i+1))
                io.imsave(save_path, SR)
                if i == 0 and opt.model == "SRDD":
                    _, d, _ = self.net(torch.rand(1, 3, 32, 32).to(self.dev))
                    d = F.pad(d, (1, 1, 1, 1), "constant", 1)
                    d = d.reshape(1, 1, -1, opt.scale+2)
                    d = 255*(d + 1)/2
                    d = d[0].clamp(0, 255).round().cpu().byte().permute(1, 2, 0).numpy()
                    io.imsave(os.path.join(save_root, "atoms.png"), d)

            HR = HR[opt.crop:-opt.crop, opt.crop:-opt.crop, :]
            SR = SR[opt.crop:-opt.crop, opt.crop:-opt.crop, :]
            if opt.eval_y_only:
                HR = utils.rgb2ycbcr(HR)
                SR = utils.rgb2ycbcr(SR)
            psnr += utils.calculate_psnr(HR, SR)
            lpips += lpips_

        #torch.cuda.empty_cache()
        self.net.train()

        return psnr/len(self.test_loader), lpips/len(self.test_loader)

    def load(self, path):
        state_dict = torch.load(path, map_location=lambda storage, loc: storage)

        if self.opt.strict_load:
            self.net.load_state_dict(state_dict)
            return

        # when to fine-tune the pre-trained model
        own_state = self.net.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data

                try:
                    own_state[name].copy_(param)
                except Exception:
                    # head and tail modules can be different
                    if name.find("head") == -1 and name.find("tail") == -1:
                        raise RuntimeError(
                            "While copying the parameter named {}, "
                            "whose dimensions in the model are {} and "
                            "whose dimensions in the checkpoint are {}."
                            .format(name, own_state[name].size(), param.size())
                        )
            else:
                raise RuntimeError(
                    "Missing key {} in model's state_dict".format(name)
                )

    def save(self, step):
        os.makedirs(self.opt.ckpt_root, exist_ok=True)
        save_path = os.path.join(self.opt.ckpt_root, str(step)+".pt")
        torch.save(self.net.state_dict(), save_path)

    def freeze_dict(self, model):
        for child in model.dict.children():
            for param in child.parameters():
                param.requires_grad = False
        return
