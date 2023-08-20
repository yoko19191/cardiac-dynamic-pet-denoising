import torch
from torch.autograd import Variable
import numpy as np
from torch import nn
from fastai2.callback.hook import hook_outputs
from torchvision.models import vgg16_bn
import torch.nn.functional as F
from fastai2.layers import MSELossFlat
from fastai2.vision.gan import AdaptiveLoss


# Perceptual Loss

# base loss
base_loss = F.l1_loss


def gram_matrix(x):
    n, c, h, w = x.size()
    x = x.view(n, c, -1)
    return (x @ x.transpose(1, 2)) / (c * h * w)


vgg_m = vgg16_bn(True).features.cuda().eval()

vgg_m.requires_grad = False
blocks = [i - 1 for i, o in enumerate(vgg_m.children()) if isinstance(o, nn.MaxPool2d)]


class PerceptualLoss(nn.Module):
    def __init__(self, m_feat, layer_ids, layer_wgts):
        super().__init__()
        self.m_feat = m_feat
        self.loss_features = [self.m_feat[i] for i in layer_ids]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = layer_wgts
        self.metric_names = ['pixel', ] + [f'feat_{i}' for i in range(len(layer_ids))
                                           ] + [f'gram_{i}' for i in range(len(layer_ids))]

    def make_features(self, x, clone=False):
        self.m_feat(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]

    def forward(self, input, target):

        # the pre-trained model needs 3 channels
        if input.shape[-3] == 1:
            input = input.repeat(1, 3, 1, 1)
        if target.shape[-3] == 1:
            target = target.repeat(1, 3, 1, 1)

        out_feat = self.make_features(target, clone=True)
        in_feat = self.make_features(input)
        self.feat_losses = [base_loss(input, target)]
        self.feat_losses += [base_loss(f_in, f_out) * w
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.feat_losses += [base_loss(gram_matrix(f_in), gram_matrix(f_out)) * w ** 2 * 5e3
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.metrics = dict(zip(self.metric_names, self.feat_losses))
        return sum(self.feat_losses)

    def __del__(self): self.hooks.remove()


perc_loss = PerceptualLoss(vgg_m, blocks[2:5], [5, 15, 2])

# MSE Loss
mse_loss = MSELossFlat()

# MAE Loss
mae_loss = F.l1_loss


# content Loss
def content_loss(pred, target, coeff=1):
    return coeff * perc_loss(pred, target) + mae_loss(pred, target)


# discriminator loss (GAN)
loss_critic = AdaptiveLoss(nn.BCEWithLogitsLoss())


## metrics ##

def gaussian(window_size, sigma):
    gauss = torch.Tensor([np.exp(-(x - window_size/2)**2/float(2*sigma**2)) for x in range(window_size)]).cuda()
    return gauss/gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size)).cuda()
    return window


def SSIM(img1, img2):
    """Structural similarity measure"""

    # pixel range 0-1
    PIXEL_MAX = 1

    # maxv = torch.max(torch.cat((img1, img2)))
    # minv = torch.min(torch.cat((img1, img2)))
    #
    # # pixel range 0-1
    # img1 = (img1 - minv) / (maxv - minv)
    # img2 = (img2 - minv) / (maxv - minv)


    if len(img1.size()) > 4:
        img1 = img1[:,0,:, :,:]
        img2 = img2[:,0,:, :,:]

    (_, channel, _, _) = img1.size()

    window_size = 17
    window = create_window(window_size, channel)
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel).cuda()
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel).cuda()

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel).cuda() - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel).cuda() - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel).cuda() - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean()


def SSIM_255(img1, img2):
    """Structural similarity measure"""

    # # SSIM 0-1
    # maxv = np.max(np.concatenate((img1, img2)))
    # minv = np.min(np.concatenate((img1, img2)))
    #
    # img1 = (img1 - minv) / (maxv - minv)
    # img2 = (img2 - minv) / (maxv - minv)
    #
    #
    #

    maxv = torch.max(torch.cat((img1, img2)))
    minv = torch.min(torch.cat((img1, img2)))

    # pixel range 0-1
    img1 = (img1 - minv) / (maxv - minv)
    img2 = (img2 - minv) / (maxv - minv)

    # SSIM 0-255
    img1 = img1 * 255
    img2 = img2 * 255



    if len(img1.size()) > 4:
        img1 = img1[:, 0, :, :, :]
        img2 = img2[:, 0, :, :, :]

    (_, channel, _, _) = img1.size()

    window_size = 17
    window = create_window(window_size, channel)
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel).cuda()
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel).cuda()

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel).cuda() - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel).cuda() - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel).cuda() - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()


def SSIM_255_CPU(img1, img2):
    """Structural similarity measure"""

    # SSIM 0-1
    maxv = np.max(np.concatenate((img1, img2)))
    minv = np.min(np.concatenate((img1, img2)))

    img1 = (img1 - minv) / (maxv - minv)
    img2 = (img2 - minv) / (maxv - minv)

    # SSIM 0-255
    img1 = img1 * 255
    img2 = img2 * 255

    if len(img1.size()) > 4:
        img1 = img1[:, 0, :, :, :]
        img2 = img2[:, 0, :, :, :]

    (_, channel, _, _) = img1.size()

    window_size = 17
    window = create_window(window_size, channel)
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()



def PSNR(img1, img2):
    """ peak signal-to-noise ratio """

    PIXEL_MAX = 1

    maxv = torch.max(img2)
    minv = torch.min(img2)

    # pixel range 0-1
    img1 = (img1 - minv) / (maxv - minv)
    img2 = (img2 - minv) / (maxv - minv)

    mse = torch.mean((img1 - img2) ** 2)

    if mse == 0:
        return 100

    return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))


def mae_ssim_loss(img1, img2):
    b1, b2 = 1, 1
    return (b1 * mae_loss(img1, img2)) + (b2 * SSIM(img1, img2))
