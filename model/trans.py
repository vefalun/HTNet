import sys
from einops.einops import rearrange
sys.path.append("..")
import torch
import torch.nn as nn
from model.Block import Hiremixer
from common.opt import opts
opt = opts().parse()



class HTNet(nn.Module):
    def __init__(self, args, adj):
        super().__init__()

        if args == -1:
            layers, channel, d_hid, length  = 3, 512, 1024, 27
            self.num_joints_in, self.num_joints_out = 17, 17
        else:
            layers, channel, d_hid, length  = args.layers, args.channel, args.d_hid, args.frames
            self.num_joints_in, self.num_joints_out = args.n_joints, args.out_joints

        self.patch_embed = nn.Linear(2, channel)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_joints_in, channel))
        self.Hiremixer = Hiremixer(adj, layers, channel, d_hid, length=length)
        self.fcn = nn.Linear(args.channel, 3)

    def forward(self, x):
        x = rearrange(x, 'b f j c -> (b f) j c').contiguous()
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.Hiremixer(x)
        x = self.fcn(x)
        x = x.view(x.shape[0], -1, self.num_joints_out, x.shape[2])
        return x


