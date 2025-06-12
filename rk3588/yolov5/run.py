import json
import cv2
import numpy as np
from rknn.api import RKNN
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

def parse_outputs(outputs, conf_thres=0.3, iou_thres=0.6):
    if outputs is None:
        raise ValueError("Inference output is None")
        
    boxes = []
    scores = []
    classes = []

    for output in outputs:
        output = output.reshape(-1, 85)
        box = output[:, :4]
        score = output[:, 4]
        cls = np.argmax(output[:, 5:], axis=-1)
        score *= np.max(output[:, 5:], axis=-1)

        boxes.append(box)
        scores.append(score)
        classes.append(cls)

    boxes = np.concatenate(boxes, axis=0)
    scores = np.concatenate(scores, axis=0)
    classes = np.concatenate(classes, axis=0)

    keep = scores >= conf_thres
    boxes = boxes[keep]
    scores = scores[keep]
    classes = classes[keep]

    return nms(boxes, scores, classes, iou_thres)

def nms(boxes, scores, classes, iou_thres):
    boxes = boxes.tolist()
    scores = scores.tolist()
    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes,
        scores=scores,
        score_threshold=0.0,
        nms_threshold=iou_thres
    )
    
    if len(indices) > 0:
        indices = indices.flatten()
        boxes = [boxes[i] for i in indices]
        scores = [scores[i] for i in indices]
        classes = [classes[i] for i in indices]

    return np.array(boxes), np.array(scores), np.array(classes)

# Load RKNN model
rknn = RKNN()
ret = rknn.load_rknn('./yolov5s.rknn')
if ret != 0:
    print('Load yolov5s.rknn failed!')
    exit(ret)

# Initialize RKNN model
# Set target parameter based on your device type, e.g., 'RK3399Pro', 'RK1808', 'RV1126', 'RV1109'
ret = rknn.init_runtime(target='rk3588')
if ret != 0:
    print('Init runtime environment failed!')
    exit(ret)

# Load COCO validation set
coco = COCO('./instances_val2017.json')
img_ids = coco.getImgIds()
results = []

# Prediction and evaluation
for img_id in tqdm(img_ids):
    img_info = coco.loadImgs(img_id)[0]
    img_path = f'./val2017/{img_info["file_name"]}'
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Preprocess image
    img_resized = cv2.resize(img, (640, 640))
    img_resized = img_resized.astype(np.float32) / 255.0
    img_resized = np.expand_dims(img_resized, 0)

    # Run model
    outputs = rknn.inference(inputs=[img_resized])
    if outputs is None:
        print(f"Inference failed for image id: {img_id}")
        continue

    # Parse outputs
    boxes, scores, classes = parse_outputs(outputs)

    for box, score, cls in zip(boxes, scores, classes):
        result = {
            'image_id': img_id,
            'category_id': int(cls),
            'bbox': [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])],
            'score': float(score)
        }
        results.append(result)

# Evaluate results
coco_gt = coco
coco_dt = coco.loadRes(results)
coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

# Release RKNN resources
rknn.release()

