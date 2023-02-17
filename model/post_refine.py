import torch
import torch.nn as nn

from torch.autograd import Variable



inter_channels = [128, 256]
fc_out = inter_channels[1]
fc_unit = 1024
class post_refine(nn.Module):


    def __init__(self, opt):
        super().__init__()

        out_seqlen = 1
        fc_in = opt.out_channels*2*out_seqlen*opt.n_joints

        fc_out = opt.in_channels * opt.n_joints
        self.post_refine = nn.Sequential(
            nn.Linear(fc_in, fc_unit),
            nn.ReLU(),
            nn.Dropout(0.5,inplace=True),
            nn.Linear(fc_unit, fc_out),
            nn.Sigmoid()

        )


    def forward(self, x, x_1):
        """

        :param x:  N*T*V*3
        :param x_1: N*T*V*2
        :return:
        """
        # data normalization
        N, T, V,_ = x.size()
        x_in = torch.cat((x, x_1), -1)  #N*T*V*5
        x_in = x_in.view(N, -1)



        score = self.post_refine(x_in).view(N,T,V,2)
        score_cm = Variable(torch.ones(score.size()), requires_grad=False).cuda() - score
        x_out = x.clone()
        x_out[:, :, :, :2] = score * x[:, :, :, :2] + score_cm * x_1[:, :, :, :2]

        return x_out