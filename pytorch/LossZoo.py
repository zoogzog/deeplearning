import torch

#--------------------------------------------------------------------------------
#---- Loss: weighted binary cross entropy
#---- Used to balance the unbalanced datasets (binary classification)
class WeightedBinaryCrossEntropy(torch.nn.Module):

    def __init__(self, weightPOS, weightNEG):

        super(WeightedBinaryCrossEntropy, self).__init__()

        self.wp = weightPOS
        self.wn = weightNEG

    def forward (self, output, target):
        
        wp = self.wp
        wn = self.wn
        
        loss = -((wn * (target * torch.log(output + 0.00001)) + wp * ((1 - target) * torch.log(1 - output + 0.00001))).sum(1).mean())
             
        return loss

#--------------------------------------------------------------------------------