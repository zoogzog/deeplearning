import torch
import torch.nn as nn

# --------------------------------------------------------------------------------
# --- This file contains the collection of custom layers, used in various networks
# --------------------------------------------------------------------------------

class LayerNormalization(nn.Module):
    """This is a normalization layer"""

    def __init__(self, mean, std):
        """
        :param mean: mean vector for normalization
        :param std: standard deviation vector for normalization
        """
        super(LayerNormalization, self).__init__()

        # ---- Transform vectors so that they can work directly with tensors [Batch x Channel x H x W]
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std