import os
import numpy as np
import cv2

import torch
from torchvision import datasets
from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot as plt

DATA_PATH = r"C:\Users\pzaka\Documents\datasets\COCO"

def IoU(boxA, boxB):

	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])

	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

	iou = interArea / float(boxAArea + boxBArea - interArea)

	return iou

if __name__ == '__main__':

    data = datasets.CocoDetection(os.path.join(DATA_PATH, "val"), os.path.join(DATA_PATH, "annotations\instances_val2017.json"))

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    res = []
    gt = []
    scores = []
    prediction_class_id = []
    gt_class_id = []
    for j, (img, info) in enumerate(data):

        r = model(img)
        res.append(r)

        IoUs = []
        for pred_box in r.xyxy[0].cpu():
            row = []
            for gt_box in info:
                row.append(IoU((float(pred_box[0]), float(pred_box[1]), float(pred_box[2]), float(pred_box[3])),
                               (gt_box["bbox"][0], gt_box["bbox"][1], gt_box["bbox"][0] + gt_box["bbox"][2], gt_box["bbox"][1] + gt_box["bbox"][3])))
            IoUs.append(row)

        if len(IoUs) == 0 or len(IoUs[0]) == 0:
            continue

        for i, box_id in enumerate(np.argmax(IoUs, axis=1)):
            gt_box = info[box_id]
            gt.append((gt_box["bbox"][0], gt_box["bbox"][1], gt_box["bbox"][0] + gt_box["bbox"][2], gt_box["bbox"][1] + gt_box["bbox"][3]))
            pred_box = r.xyxy[0].cpu()[i]
            res.append((float(pred_box[0]), float(pred_box[1]), float(pred_box[2]), float(pred_box[3])))

            # prediction_class_id.append(gt_box["category_id"])
            # gt_class_id.append(int(pred_box[5]))

            scores.append(IoUs[i][box_id])

        # res_image = np.array(res.ims[0])
        # for x_min, y_min, x_max, y_max, _, _ in res.xyxy[0].cpu():
        #     res_image = cv2.rectangle(res_image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 1)
        #
        # cv2.imshow("", res_image)
        # cv2.waitKey()

    precision, recall, thresholds = precision_recall_curve(np.array(scores), scores)

    plt.plot(thresholds, recall[:-1])
    plt.savefig('recall.png')
    plt.clf()

    plt.plot(thresholds, precision[:-1])
    plt.savefig('precision.png')
    plt.clf()

    plt.plot(precision, recall, 'o')
    plt.savefig('pr.png')
    plt.clf()