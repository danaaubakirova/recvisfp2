model_defaults = {
    'bert-base-uncased': {
        'optimizer': 'AdamW',
        'max_grad_norm': 1.0,
        'scheduler': 'linear_schedule_with_warmup',
    },
    'distilbert-base-uncased': {
        'optimizer': 'AdamW',
        'max_grad_norm': 1.0,
        'scheduler': 'linear_schedule_with_warmup',
    },
    'resnet50': {
        'model_kwargs': {
            'pretrained': True,
        },
        'target_resolution': (224, 224),
    },
    'dino': {
        'model_kwargs': {
            'pretrained': True,
        },
        'target_resolution': (224, 224),
    },
    'dinov2': {
        'model_kwargs': {
            'pretrained': True,
        },
        'target_resolution': (224, 224),
    },
    'mae': {
        'model_kwargs': {
            'pretrained': True,
        },
        'target_resolution': (224, 224),
    },
}
