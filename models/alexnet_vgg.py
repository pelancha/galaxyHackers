import torch
import torch.nn as nn
from torchvision import models

class AlexNet_VGG(nn.Module):
    def __init__(self, num_ftrs, num_class=1):
        super(AlexNet_VGG, self).__init__()
        self.num_ftrs = num_ftrs

        VGG_fc = nn.Sequential(
            nn.Linear(num_ftrs, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class),
            nn.Sigmoid()
        )

        self.classifier = VGG_fc

    def forward(self, x):
        x = self.classifier(x)
        return x

def load_model(num_class=1):
  model_ft = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
  num_ftrs = model_ft.classifier[1].in_features
  half_in_size = round(num_ftrs/2)

  model_ft.classifier  = AlexNet_VGG(num_ftrs)
  return model_ft
