from wilds.common.metrics.all_metrics import (
    Accuracy,
    MultiTaskAccuracy,
    MSE,
    multiclass_logits_to_pred,
    binary_logits_to_pred,
    pseudolabel_binary_logits,
    pseudolabel_multiclass_logits,
    pseudolabel_identity,
    pseudolabel_detection,
    pseudolabel_detection_discard_empty,
    MultiTaskAveragePrecision
)

algo_log_metrics = {
    'accuracy': Accuracy(prediction_fn=multiclass_logits_to_pred),
    'mse': MSE(),
    'multitask_accuracy': MultiTaskAccuracy(prediction_fn=multiclass_logits_to_pred),
    'multitask_binary_accuracy': MultiTaskAccuracy(prediction_fn=binary_logits_to_pred),
    'multitask_avgprec': MultiTaskAveragePrecision(prediction_fn=None),
    None: None,
}

process_outputs_functions = {
    'binary_logits_to_pred': binary_logits_to_pred,
    'multiclass_logits_to_pred': multiclass_logits_to_pred,
    None: None,
}

process_pseudolabels_functions = {
    'pseudolabel_binary_logits': pseudolabel_binary_logits,
    'pseudolabel_multiclass_logits': pseudolabel_multiclass_logits,
    'pseudolabel_identity': pseudolabel_identity,
    'pseudolabel_detection': pseudolabel_detection,
    'pseudolabel_detection_discard_empty': pseudolabel_detection_discard_empty,
}

# see initialize_*() functions for correspondence=
# See algorithms/initializer.py
algorithms = ['ERM', 'groupDRO']

# See transforms.py
transforms = ['bert', 'image_base', 'image_resize', 'image_resize_and_center_crop']
additional_transforms = ['randaugment', 'weak']

# See models/initializer.py
models = ['resnet50', 'bert-base-uncased', 'distilbert-base-uncased', 'dino', 'dinov2','mae']

# See optimizer.py
optimizers = ['SGD', 'Adam', 'AdamW']

# See scheduler.py
schedulers = ['linear_schedule_with_warmup', 'cosine_schedule_with_warmup', 'ReduceLROnPlateau', 'StepLR', 'MultiStepLR']

# See losses.py
losses = ['cross_entropy', 'lm_cross_entropy', 'MSE', 'multitask_bce', 'cross_entropy_logits']
