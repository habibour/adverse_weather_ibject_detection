Cuda            = True
seed            = 114514
distributed     = False
sync_bn         = False
fp16            = False
classes_path    = 'model_data/rtts_classes.txt'
anchors_path    = 'model_data/yolo_anchors.txt'
anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
model_path      = 'model_data/yolov7_tiny_weights.pth'
input_shape     = [640, 640]
pretrained      = False
Init_Epoch          = 0
Freeze_Epoch        = 100
Freeze_batch_size   = 16
UnFreeze_Epoch      = 300
Unfreeze_batch_size = 16
Freeze_Train        = True
Init_lr             = 1e-2
Min_lr              = Init_lr * 0.01
optimizer_type      = "sgd"
momentum            = 0.937
weight_decay        = 5e-4
lr_decay_type       = "cos"
save_period         = 10
save_dir            = 'logs'
eval_flag           = True
eval_period         = 10
num_workers         = 0
train_annotation_path   = '2007_train_fog.txt'
val_annotation_path     = '2007_val_fog.txt'
clear_annotation_path = '2007_train.txt'

# --- Thesis modifications over the base paper, all default OFF so the ---
# --- committed repo reproduces the base paper's exact behavior unless   ---
# --- explicitly enabled (e.g. by the fine-tuning notebook).             ---
# DRM: zero-init residual Detail Recovery Module at the P3 (finest) scale,
#      targeting the small/thin-object weakness (bicycle, motorbike).
USE_DRM         = False
# Wise-IoU: dynamic non-monotonic focusing applied on top of the existing
#           CIoU box-regression loss.
USE_WISE_IOU    = False
# DWA: Dynamic Weight Averaging (Liu et al., CVPR 2019) for the detection /
#      dehaze task balance, replacing the fixed lambda=0.1 below.
USE_DWA         = False