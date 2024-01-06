import torch
import torch.nn as nn

class Identity(nn.Module):
    """An identity layer"""
    def __init__(self, d):
        super().__init__()
        self.in_features = d
        self.out_features = d

    def forward(self, x):
        return x

class DinoVisionTransformerClassifier(nn.Module):
    def __init__(self, d_out = 10, model_name = 'dinov2_vits14'):
        super(DinoVisionTransformerClassifier, self).__init__()
        checkpoint_model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.transformer = deepcopy(checkpoint_model)
        self.classifier = nn.Linear(384, d_out)

    def forward(self, x):
        x = self.transformer(x)
        #x = self.transformer.norm(x)
        x = self.classifier(x)
        return x