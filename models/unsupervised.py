import torch
import torch.nn as nn
import sys
from models.layers import Identity
from copy import deepcopy
import types
import sys
import models.models_vit as mae


def _replace_fc(model, output_dim):
    d = model.fc.in_features
    model.fc = torch.nn.Linear(d, output_dim)
    return model

def _vit_replace_fc(model, output_dim):
    model.fc = torch.nn.Identity()
    model.fc.in_features = model.head.in_features
    delattr(model, "head")

    def forward(self, x):
        x = self.forward_features(x)
        x = self.fc(x)
        return x

    forwardType = types.MethodType
    model.forward = forwardType(forward, model)
    return _replace_fc(model, output_dim)

def imagenet_resnet50_dino(output_dim):
    # workaround to avoid module name collision
    sys.modules.pop("utils")
    model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
    model.fc.in_features = 2048
    return _replace_fc(model, output_dim)

def load_mae_model(model, checkpoint_path, global_pool=True):
    from timm.models.layers import trunc_normal_
    from models.util.pos_embed import interpolate_pos_embed

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
    print("Load pre-trained checkpoint from: %s" % checkpoint_path)
    if 'model' in checkpoint:
        checkpoint_model = checkpoint['model']
    else:
        checkpoint_model = checkpoint

    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model:# and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    if global_pool:
        for k in ['fc_norm.weight', 'fc_norm.bias']:
            if k in checkpoint_model:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

    # interpolate position embedding
    interpolate_pos_embed(model, checkpoint_model)

    # load pre-trained model
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)

    trunc_normal_(model.head.weight, std=2e-5)

def imagenet_mae_base_pretrained(output_dim):
    model = mae.vit_base_patch16(num_classes=output_dim)
    load_mae_model(model, '/kaggle/input/mae-checkpoint/mae_pretrain_vit_base.pth')
    return _vit_replace_fc(model, output_dim)
