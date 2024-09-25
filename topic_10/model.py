import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
)

def create_model(num_classes,checkpoint=None,device='cpu'):

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=True,
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
        pretrained_backbone=True
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    if checkpoint:
        checkpoint = torch.load(checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model = model.to(device)
    return model