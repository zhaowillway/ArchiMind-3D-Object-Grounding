"""Original educational minimal visual-language multi-slot regression baseline.
Not CyclopsNet, and not guaranteed compatible with VLP evaluation.
"""
import torch
from torch import nn

class TinyVLMOD(nn.Module):
    def __init__(self, vocab_size=2048, slots=2):
        super().__init__()
        self.slots = slots
        self.vision = nn.Sequential(
            nn.Conv2d(3,16,3,2,1),nn.ReLU(),
            nn.Conv2d(16,32,3,2,1),nn.ReLU(),
            nn.Conv2d(32,64,3,2,1),nn.ReLU(),
            nn.AdaptiveAvgPool2d((4,4)),nn.Flatten(),nn.Linear(1024,128),nn.ReLU())
        self.text = nn.Embedding(vocab_size,32)
        self.fusion = nn.Sequential(nn.Linear(160,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU())
        # each output slot: objectness logit + 7 normalized box fields (xyz, whd, yaw)
        self.head = nn.Linear(128,slots*8)
    def forward(self, images, token_ids):
        vis = self.vision(images)
        txt = self.text(token_ids).mean(dim=1)
        return self.head(self.fusion(torch.cat([vis,txt],dim=1))).view(-1,self.slots,8)
