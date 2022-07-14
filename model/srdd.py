import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia


dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ESPCN(nn.Module):
    def __init__(self, scale, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, 64, 5, 1, 2),
            nn.Tanh(),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.Tanh(),
            nn.Conv2d(32, out_c*(scale**2), 3, 1, 1),
            nn.PixelShuffle(scale)
        )

    def forward(self, x):
        return self.conv(x)


class PatchConv(nn.Module):
    def __init__(self, channels, tanh=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, 1, 0)
        )

        if tanh:
            self.act = nn.Tanh()
        else:
            self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        return self.act(x)


class Dictionary(nn.Module):
    def __init__(self, n_atoms, patch_size):
        super().__init__()
        self.depth = int(math.log2(n_atoms))  # num elements = 2^depth
        for k in range(self.depth):
            for i in range(2**(k+1)):
                if k < self.depth - 1:
                    setattr(self,
                        "conv_{}_{}".format(k+1, i+1),
                        PatchConv(patch_size**2)
                    )
                else:
                    setattr(self,
                        "conv_{}_{}".format(k+1, i+1),
                        PatchConv(patch_size**2, tanh=True)
                    )
        self.d2s = nn.PixelShuffle(patch_size)

        self.code = nn.Sequential(
            nn.Conv2d(n_atoms, n_atoms, patch_size, patch_size, 0, groups=n_atoms),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_atoms, n_atoms, 1, 1, 0)
        )

    def forward(self, x, shuffle):
        d = [x]
        for k in range(self.depth):
            tmp = [None]*(len(d)*2)
            for i in range(len(d)):
                x1 = getattr(self, "conv_{}_{}".format(k+1, i*2+1))(d[i])
                x2 = getattr(self, "conv_{}_{}".format(k+1, i*2+2))(d[i])
                if k == self.depth - 1:
                    x1 = self.d2s(x1)
                    x2 = self.d2s(x2)
                tmp[i*2]   = x1
                tmp[i*2+1] = x2
            d = tmp

        if shuffle:
            random.shuffle(d)
        d = torch.cat(d, dim=1)
        return d, self.code(d)


class Bottleneck(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(nf, nf//4, 1, 1, 0),
            nn.BatchNorm2d(nf//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf//4, nf//4, 3, 1, 1),
            nn.BatchNorm2d(nf//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf//4, nf, 1, 1, 0),
            nn.BatchNorm2d(nf)
            )

    def forward(self, x):
        return x + self.conv(x)


class BottleneckGroup(nn.Module):
    def __init__(self, nf, blocks):
        super().__init__()
        group = [Bottleneck(nf) for _ in range(blocks)]
        group += [nn.Conv2d(nf, nf, 1, 1, 0)]
        self.group = nn.Sequential(*group)

    def forward(self, x):
        return x + self.group(x)


class UNetBlock(nn.Module):
    def __init__(self, in_c, mid_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, mid_c, 3, 1, 1),
            nn.BatchNorm2d(mid_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_c, out_c, 3, 1, 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class NestedUNet(nn.Module):
    def __init__(self, nfs, in_c=1):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.initial = nn.Conv2d(in_c, nfs[0]//2, 3, 1, 1)

        self.conv0_0 = UNetBlock(in_c, nfs[0], nfs[0])
        self.conv1_0 = UNetBlock(nfs[0], nfs[1], nfs[1])
        self.conv2_0 = UNetBlock(nfs[1], nfs[2], nfs[2])
        self.conv3_0 = UNetBlock(nfs[2], nfs[3], nfs[3])

        self.conv0_1 = UNetBlock(nfs[0]+nfs[1], nfs[0], nfs[0])
        self.conv1_1 = UNetBlock(nfs[1]+nfs[2], nfs[1], nfs[1])
        self.conv2_1 = UNetBlock(nfs[2]+nfs[3], nfs[2], nfs[2])

        self.conv0_2 = UNetBlock(nfs[0]*2+nfs[1], nfs[0], nfs[0])
        self.conv1_2 = UNetBlock(nfs[1]*2+nfs[2], nfs[1], nfs[1])

        self.conv0_3 = UNetBlock(nfs[0]*3+nfs[1], nfs[0], nfs[0])

        self.final = nn.Conv2d(nfs[0]+nfs[0]//2, nfs[0], 3 , 1, 1)

    def forward(self, x):
        xini = self.initial(x)

        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        return self.final(torch.cat([x0_3, xini], 1))


class Net(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.ps = opt.scale  # patch size
        n_atoms = 64  # 2^N
        nfs     = [n_atoms, n_atoms*2, n_atoms*4, n_atoms*8]

        self.chsr = ESPCN(scale=opt.scale, in_c=3, out_c=2)

        self.dict = Dictionary(n_atoms=n_atoms, patch_size=self.ps)

        self.body = NestedUNet(nfs=nfs, in_c=3)
        self.tail = nn.Sequential(
            nn.Conv2d(nfs[0] + n_atoms, nfs[0] + n_atoms, 1, 1, 0),
            BottleneckGroup(nfs[0] + n_atoms, blocks=10),
            nn.Conv2d(nfs[0] + n_atoms, n_atoms, 1, 1, 0),
            nn.Softmax(dim=1)  # more stable than relu
        )

        self.comp = nn.Sequential(
            nn.Conv2d(n_atoms, n_atoms, 2, 1, 0),  # size -1
            nn.ReLU(inplace=True),
            nn.Conv2d(n_atoms, n_atoms, 1, 1, 0),
            nn.Softmax(dim=1)
        )
        self.fuse = nn.Conv2d(2, 1, 5, 1, 0)

    def forward(self, x, stored_dict=None, stored_code=None, shuffle=False):
        x = x/255
        b, _, h, w = x.size()

        #### initial
        x  = kornia.color.rgb_to_ycbcr(x)
        xl = F.interpolate(x[:, 0:1, :, :], scale_factor=self.ps, mode="bicubic", align_corners=False)
        xc = self.chsr(x)

        #### dictionary
        if stored_dict == None or stored_code == None:
            d = torch.randn(b, self.ps**2, 1, 1, requires_grad=True).to(dev)
            d, c = self.dict(d, shuffle=shuffle)  # [B, n_atoms, ps, ps], [B, n_atoms, 1, 1]
            d = d.repeat(1, 1, h, w)  # [B, n_atoms, H*ps, W*ps]
            c = c.repeat(1, 1, h, w)  # [B, n_atoms, H, W]
        else:
            d = stored_dict
            c = stored_code

        #### prediction
        x = self.body(x)  # [B, nfs[0], H, W]
        x = self.tail(torch.cat([x, c], dim=1))
        y = self.comp(x)

        # visualize sparsity map
        #x = torch.where(x > 1e-2, torch.ones(x.size()).to("cuda"), torch.zeros(x.size()).to("cuda"))
        #x = torch.sum(x, dim=1, keepdim=True)
        #return x, d, c

        #### reconstruction
        x = F.interpolate(x, scale_factor=self.ps, mode="nearest")
        y = F.interpolate(y, scale_factor=self.ps, mode="nearest")
        x = x * d
        y = y * d[:, :, self.ps:, self.ps:]
        x = torch.sum(x, dim=1, keepdim=True)  # [B, 1, H*ps, W*ps]
        y = torch.sum(y, dim=1, keepdim=True)  # [B, 1, (H-1)*ps, (W-1)*ps]

        m = self.ps//2
        x[:, :, m+2:-m-2, m+2:-m-2] = self.fuse(torch.cat([x[:, :, m:-m, m:-m], y], dim=1))

        #### final
        x = x + xl
        x = kornia.color.ycbcr_to_rgb(torch.cat([x, xc], dim=1))
        return x*255, d[:1, :, :self.ps, :self.ps], c[:1, :, :1, :1]
