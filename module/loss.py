import torch
import torch.nn.functional as F

class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel

        return -self._ssim(img1, img2, window, self.window_size, channel, self.size_average)

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
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
        
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def create_window(self, window_size, channel):
        def gaussian(window_size, sigma):
            gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
            return gauss / gauss.sum()

        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

class SSIM_MSELoss(torch.nn.Module):
    """
    Combines the Structural Similarity Index (SSIM) loss with the Mean Square Error (MSE) loss.
    
    Args:
    - window_size (int): The size of the window to compute SSIM. Default is 11.
    - size_average (bool): If True, the losses will be averaged across each batch. 
                           If False, the losses will be summed. Default is True.
    - alpha (float): The weighting factor for the SSIM loss in the combined loss.
                     The MAE loss weight will be (1-alpha). Default is 0.5.

    Usage:
    Initialize the loss and compute it:
    ```python
    criterion = SSIMPlusMAELoss(window_size=11, size_average=True, alpha=0.5)
    loss = criterion(preds, targets)
    ```
    where `preds` are the model outputs and `targets` are the ground truth images.
    """
    def __init__(self, window_size=11, size_average=True, alpha=0.5):
        super(SSIM_MSELoss, self).__init__()
        self.ssim = SSIMLoss(window_size, size_average)
        self.mse = torch.nn.MSELoss(reduction='mean' if size_average else 'sum')
        self.alpha = alpha

    def forward(self, img1, img2):
        ssim_loss = self.ssim(img1, img2)
        mse_loss = self.mse(img1, img2)
        return self.alpha * ssim_loss + (1 - self.alpha) * mse_loss


class SSIM_MAELoss(torch.nn.Module):
    """
    Combines the Structural Similarity Index (SSIM) loss with the Mean Absolute Error (MAE) loss.
    
    Args:
    - window_size (int): The size of the window to compute SSIM. Default is 11.
    - size_average (bool): If True, the losses will be averaged across each batch. 
                           If False, the losses will be summed. Default is True.
    - alpha (float): The weighting factor for the SSIM loss in the combined loss.
                     The MAE loss weight will be (1-alpha). Default is 0.5.

    Usage:
    Initialize the loss and compute it:
    ```python
    criterion = SSIMPlusMAELoss(window_size=11, size_average=True, alpha=0.5)
    loss = criterion(preds, targets)
    ```
    where `preds` are the model outputs and `targets` are the ground truth images.
    """
    def __init__(self, window_size=11, size_average=True, alpha=0.5):
        super(SSIM_MAELoss, self).__init__()
        self.ssim = SSIMLoss(window_size, size_average)
        self.mae = torch.nn.L1Loss(reduction='mean' if size_average else 'sum')
        self.alpha = alpha

    def forward(self, img1, img2):
        ssim_loss = self.ssim(img1, img2)
        mae_loss = self.mae(img1, img2)
        return self.alpha * ssim_loss + (1 - self.alpha) * mae_loss