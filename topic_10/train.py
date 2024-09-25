# https://github.com/GirinChutia/FasterRCNN-Torchvision-FineTuning

import os
import tqdm
import time
import json

from torch import nn
import torch
from torchvision import datasets
from torchvision import io, utils
# from torchvision import datapoints
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as F
# from pycocotools import COCO, COCOeval

from model import create_model
from model_utils import InferFasterRCNN
from engine import train_one_epoch, evaluate
DATA_PATH = r"C:\Users\pzaka\Documents\datasets\COCO"


def save_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file)


def evaluate_model(image_dir,
                   gt_ann_file,
                   model_weight):
    _ds = CocoDataset(
        image_folder=image_dir,
        annotations_file=gt_ann_file,
        height=640,
        width=640,
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    IF_C = InferFasterRCNN(num_classes=_ds.get_total_classes_count() + 1,
                           classnames=_ds.get_classnames())

    IF_C.load_model(checkpoint=model_weight,
                    device=device)

    image_dir = image_dir

    cocoGt = COCO(annotation_file=gt_ann_file)
    imgIds = cocoGt.getImgIds()  # all image ids

    res_id = 1
    res_all = []

    for id in tqdm(imgIds, total=len(imgIds)):
        id = id
        img_info = cocoGt.loadImgs(imgIds[id])[0]
        annIds = cocoGt.getAnnIds(imgIds=img_info['id'])
        ann_info = cocoGt.loadAnns(annIds)
        image_path = os.path.join(image_dir,
                                  img_info['file_name'])
        transform_info = CocoDataset.transform_image_for_inference(image_path, width=640, height=640)
        result = IF_C.infer_image(transform_info=transform_info,
                                  visualize=False)

        if len(result) > 0:
            pred_boxes_xyxy = result['unscaled_boxes']
            pred_boxes_xywh = [[i[0], i[1], i[2] - i[0], i[3] - i[1]] for i in pred_boxes_xyxy]
            pred_classes = result['pred_classes']
            pred_scores = result['scores']
            pred_labels = result['labels']

            for i in range(len(pred_boxes_xywh)):
                res_temp = {"id": res_id,
                            "image_id": id,
                            "bbox": pred_boxes_xywh[i],
                            "segmentation": [],
                            "iscrowd": 0,
                            "category_id": int(pred_labels[i]),
                            "area": pred_boxes_xywh[i][2] * pred_boxes_xywh[i][3],
                            "score": float(pred_scores[i])}
                res_all.append(res_temp)
                res_id += 1

    save_json_path = 'test_dect.json'
    save_json(res_all, save_json_path)

    cocoGt = COCO(gt_ann_file)
    cocoDt = cocoGt.loadRes(save_json_path)

    cocoEval = COCOeval(cocoGt, cocoDt, iouType='bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    AP_50_95 = cocoEval.stats.tolist()[0]
    AP_50 = cocoEval.stats.tolist()[1]

    del IF_C, _ds
    os.remove(save_json_path)

    torch.cuda.empty_cache()
    gc.collect()

    return {'AP_50_95': AP_50_95,
            'AP_50': AP_50}

class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's
    validation loss is less than the previous least less, then save the
    model state.
    """

    def __init__(
            self, best_valid_loss=float('inf'), output_dir='weight_outputs',
    ):
        self.best_valid_loss = best_valid_loss

        os.makedirs(output_dir, exist_ok=True)

        self.output_dir = output_dir

    def __call__(
            self, current_valid_loss,
            epoch, model, optimizer
    ):
        self.model_save_path = f'{self.output_dir}/best_model.pth'
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch + 1}\n")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, self.model_save_path)

def train(
        train_dataset,
        val_dataset,
        epochs=2,
        batch_size=8,
        exp_folder="exp",
        val_eval_freq=1,
):
    date_format = "%d-%m-%Y-%H-%M-%S"
    date_string = time.strftime(date_format)

    exp_folder = os.path.join("exp", "summary", date_string)

    def custom_collate(data):
        return data

    # Dataloaders --
    train_dl = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_dl = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # Device --
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Model --
    faster_rcnn_model = create_model(train_dataset.get_total_classes_count() + 1)
    faster_rcnn_model = faster_rcnn_model.to(device)

    # Optimizer --
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups

    for k, v in faster_rcnn_model.named_modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d) or "bn" in k:
            pg0.append(v.weight)  # no decay
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay

    optimizer = torch.optim.SGD(
        pg0, lr=0.001, momentum=0.9, nesterov=True
    )  # BN

    optimizer.add_param_group(
        {"params": pg1, "weight_decay": 5e-4}
    )  # add pg1 with weight_decay # Weights

    optimizer.add_param_group({"params": pg2})  # Biases

    num_epochs = epochs
    save_best_model = SaveBestModel(output_dir=exp_folder)

    for epoch in range(num_epochs):

        faster_rcnn_model, optimizer, writer, epoch_loss = train_one_epoch(
            faster_rcnn_model,
            train_dl,
            optimizer,
            writer,
            epoch + 1,
            num_epochs,
            device,
        )

        if (epoch % val_eval_freq == 0) and epoch != 0:  # Do evaluation of validation set
            eval_result = evaluate_model(image_dir=val_dataset.image_folder,
                                         gt_ann_file=val_dataset.annotations_file,
                                         model_weight=save_best_model.model_save_path)

            writer.add_scalar("Val/AP_50_95", eval_result['AP_50_95'], epoch + 1)
            writer.add_scalar("Val/AP_50", eval_result['AP_50'], epoch + 1)

        else:
            writer, val_epoch_loss = evaluate(
                faster_rcnn_model,
                val_dl,
                writer,
                epoch + 1,
                num_epochs,
                device,
                log=True,
            )


            save_best_model(val_epoch_loss,
                            epoch,
                            faster_rcnn_model,
                            optimizer)

    _, _ = evaluate(
        faster_rcnn_model, val_dl, writer, epoch + 1, num_epochs, device, log=False
    )

    writer.add_hparams(
        {"epochs": epochs, "batch_size": batch_size},
        {"Train/total_loss": epoch_loss, "Val/total_loss": val_epoch_loss},
    )


if __name__ == '__main__':
    val_data = datasets.CocoDetection(os.path.join(DATA_PATH, "val"),
                                      os.path.join(DATA_PATH, "annotations\instances_val2017.json"))
    train_data = datasets.CocoDetection(os.path.join(DATA_PATH, "train"),
                                        os.path.join(DATA_PATH, "annotations\instances_train2017.json"))

    train(train_data, val_data)