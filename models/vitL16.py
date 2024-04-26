import torch
import torch.nn as nn
import torchvision.models as models

class Sigm(nn.Module):
    def __init__(self):
        super(Sigm, self).__init__()
        self.fc_out = nn.Sequential(nn.Dropout(p = 0.5), nn.Linear(1024, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.fc_out(x)
        return x

def load_model():
    layer_width = 512
    model_ft = timm.create_model('vit_large_patch16_224', pretrained=True, num_classes = 1)
    model_ft.head = Sigm()
    return model_ft
