import torch
import torch.nn as nn
import sys
from models.layers import Identity, DinoVisionTransformerClassifier
from undupervised import imagenet_mae_base_pretrained, imagenet_resnet50_dino
from copy import deepcopy

def initialize_model(config, d_out, is_featurizer=False):
    """
    Initializes models according to the config
        Args:
            - config (dictionary): config dictionary
            - d_out (int): the dimensionality of the model output
            - is_featurizer (bool): whether to return a model or a (featurizer, classifier) pair that constitutes a model.
        Output:
            If is_featurizer=True:
            - featurizer: a model that outputs feature Tensors of shape (batch_size, ..., feature dimensionality)
            - classifier: a model that takes in feature Tensors and outputs predictions. In most cases, this is a linear layer.

            If is_featurizer=False:
            - model: a model that is equivalent to nn.Sequential(featurizer, classifier)

        Pretrained weights are loaded according to config.pretrained_model_path using either transformers.from_pretrained (for bert-based models)
        or our own utils.load function (for torchvision models, resnet18-ms, and gin-virtual).
        There is currently no support for loading pretrained weights from disk for other models.
    """
    # If load_featurizer_only is True,
    # then split into (featurizer, classifier) for the purposes of loading only the featurizer,
    # before recombining them at the end
    featurize = is_featurizer or config.load_featurizer_only or (config.local_norm == 'last')

    if config.model == 'resnet50':
        if featurize:
            featurizer = initialize_torchvision_model(
                name=config.model,
                d_out=None,
                **config.model_kwargs)
            classifier = nn.Linear(featurizer.d_out, d_out)
            model = (featurizer, classifier)
        else:
            model = initialize_torchvision_model(
                name=config.model,
                d_out=d_out,
                **config.model_kwargs)
            
    elif config.model == 'dino':
        if featurize:
            # Initialize DINO model without final classification layer
            featurizer = imagenet_resnet50_dino(None)
            # Create a linear classifier with the output dimension of the featurizer
            classifier = nn.Linear(featurizer.fc.in_features, d_out)
            model = (featurizer, classifier)
        else:
            # Initialize DINO model with final classification layer for the specified output dimension
            model = imagenet_resnet50_dino(d_out)
    
    elif config.model == 'dinov2':
        if featurize:
            # Initialize DINO model without final classification layer
            featurizer = DinoVisionTransformerClassifier(d_out = None, model_name = 'dinov2_vits14')
            # Create a linear classifier with the output dimension of the featurizer
            classifier = nn.Linear(featurizer.fc.in_features, d_out)
            print('featurizer')
            model = (featurizer, classifier)
        else:
            # Initialize DINO model with final classification layer for the specified output dimension
            model = DinoVisionTransformerClassifier(d_out=d_out, model_name = 'dinov2_vits14')
            
    elif 'bert' in config.model:
        if featurize:
            featurizer = initialize_bert_based_model(config, d_out, featurize)
            classifier = nn.Linear(featurizer.d_out, d_out)
            model = (featurizer, classifier)
        else:
            model = initialize_bert_based_model(config, d_out)

    else:
        raise ValueError(f'Model: {config.model} not recognized.')

    # Recombine model if we originally split it up just for loading
    if featurize and not is_featurizer:
        model = nn.Sequential(*model)

    # The `needs_y` attribute specifies whether the model's forward function
    # needs to take in both (x, y).
    # If False, Algorithm.process_batch will call model(x).
    # If True, Algorithm.process_batch() will call model(x, y) during training,
    # and model(x, None) during eval.
    if not hasattr(model, 'needs_y'):
        # Sometimes model is a tuple of (featurizer, classifier)
        if is_featurizer:
            for submodel in model:
                submodel.needs_y = False
        else:
            model.needs_y = False

    return model


def initialize_bert_based_model(config, d_out, featurize=False):
    from models.bert.bert import BertClassifier, BertFeaturizer
    from models.bert.distilbert import DistilBertClassifier, DistilBertFeaturizer

    if config.pretrained_model_path:
        print(f'Initialized model with pretrained weights from {config.pretrained_model_path}')
        config.model_kwargs['state_dict'] = torch.load(config.pretrained_model_path, map_location=config.device)

    if config.model == 'bert-base-uncased':
        if featurize:
            model = BertFeaturizer.from_pretrained(config.model, **config.model_kwargs)
        else:
            model = BertClassifier.from_pretrained(
                config.model,
                num_labels=d_out,
                **config.model_kwargs)
    elif config.model == 'distilbert-base-uncased':
        if featurize:
            model = DistilBertFeaturizer.from_pretrained(config.model, **config.model_kwargs)
        else:
            model = DistilBertClassifier.from_pretrained(
                config.model,
                num_labels=d_out,
                **config.model_kwargs)
    else:
        raise ValueError(f'Model: {config.model} not recognized.')
    return model

def initialize_torchvision_model(name, d_out, **kwargs):
    import torchvision

    # get constructor and last layer names
    if name == 'resnet50':
        constructor_name = name
        last_layer_name = 'fc'
    else:
        raise ValueError(f'Torchvision model {name} not recognized')
    # construct the default model, which has the default last layer
    constructor = getattr(torchvision.models, constructor_name)
    model = constructor(**kwargs)
    # adjust the last layer
    d_features = getattr(model, last_layer_name).in_features
    if d_out is None:  # want to initialize a featurizer model
        last_layer = Identity(d_features)
        model.d_out = d_features
    else: # want to initialize a classifier for a particular num_classes
        last_layer = nn.Linear(d_features, d_out)
        model.d_out = d_out
    setattr(model, last_layer_name, last_layer)

    return model