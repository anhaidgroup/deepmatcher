import pdb

import torch
import torch.nn as nn


class SoftNLLLoss(nn.NLLLoss):

    def __init__(self, label_smoothing=0, weight=None, num_classes=2, **kwargs):
        super(SoftNLLLoss, self).__init__(**kwargs)
        self.label_smoothing = label_smoothing
        self.confidence = 1 - self.label_smoothing
        self.num_classes = num_classes
        self.register_buffer('weight', weight)

        assert label_smoothing >= 0.0 and label_smoothing <= 1.0

        if num_classes == 2:
            self.criterion1 = nn.NLLLoss(weight=weight, **kwargs)
        else:
            self.criterion2 = nn.KLDivLoss(**kwargs)

    def forward(self, input, target):
        # if self.num_classes > 2:
        one_hot = torch.zeros_like(input)
        one_hot.fill_(self.label_smoothing / (self.num_classes - 1))
        one_hot.scatter_(1, target.unsqueeze(1).long(), self.confidence)
        print(one_hot)
        pdb.set_trace()

        if self.weight:
            one_hot.mul_(self.weight)

        loss1 = self.criterion1(input, one_hot)
        # else:
        loss2 = (self.confidence * self.criterion2(input, target) +
                 self.label_smoothing * self.criterion2(input, 2 - target))

        print(loss1 - loss2)
        pdb.set_trace()
