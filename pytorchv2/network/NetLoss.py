import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------------
# ---- USED IN: Neural style transfer - content loss
class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()

        # ---- Detach form computing the gradients
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

# --------------------------------------------------------------------------------
# ---- USED IN: Neural style transfer - style loss
class StyleLoss(nn.Module):

    def __grammatrix__(self, input):
        a, b, c, d = input.size()
        features = input.view(a * b, c * d)
        G = torch.mm(features, features.t())

        return G.div(a * b * c * d)

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = self.__grammatrix__(target_feature).detach()

    def forward(self, input):
        G = self.__grammatrix__(input)
        self.loss = F.mse_loss(G, self.target)
        return input

# --------------------------------------------------------------------------------
# ---- USED IN: Binary classification tasks with unbalanced datasets
class WeightedBinaryCrossEntropy(torch.nn.Module):

    def __init__(self, weightPOS, weightNEG):
        super(WeightedBinaryCrossEntropy, self).__init__()

        self.wp = weightPOS
        self.wn = weightNEG

    def forward(self, output, target):
        wp = self.wp
        wn = self.wn

        loss = -(
            (wn * (target * torch.log(output + 0.00001)) + wp * ((1 - target) * torch.log(1 - output + 0.00001))).sum(
                1).mean())

        return loss

# --------------------------------------------------------------------------------
# ----- Loss: weighted binary cross entropy (multiclass version)

def multiclass_binary_cross_entropy(prediction, target, weight=None):
    # Bunch of assertions to make sure what we got is good, yay Pythonnnnn typeless poooooop
    assert torch.is_tensor(prediction)
    assert torch.is_tensor(target)
    assert torch.is_tensor(weight)
    # Checking sizes
    assert prediction.size() == target.size()
    size = target.size()
    L = torch.zeros(size[0])

    for b_idx in range(size[0]):
        L[b_idx] = F.binary_cross_entropy(prediction[b_idx], target[b_idx], weight=weight, size_average=False)

    return torch.mean(L)


class WeightedBinaryCrossEntropyMC(torch.nn.Module):

    def __init__(self, weights):

        super(WeightedBinaryCrossEntropyMC, self).__init__()

        self.weights = weights

    def forward (self, output, target):

        weights = self.weights
        wt = torch.from_numpy(np.array(weights)).type(torch.FloatTensor).cuda()


        L = multiclass_binary_cross_entropy(output, target, wt)

        return L