import math
from abc import abstractmethod, ABC

import acl
import cv2
import numpy as np
import json

ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2


dtype_map=[
    np.dtype('float32'),
    np.dtype('float16'),
    np.dtype('int8'),
    np.dtype('int32'),
    np.dtype('uint8'),
    np.dtype('int16'),
    np.dtype('uint16'),
    np.dtype('uint32'),
    np.dtype('int64'),
    np.dtype('uint64'),
    np.dtype('float64'),
    np.dtype('bool'),
]


CLASSES=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush'] #coco80类别


def xywh2xyxy(x):
    # [x,y,w,h] to [x1,y1,x2,y2]
    y = np.copy(x)
    y[:,0] = x[:,0] - x[:,2] / 2
    y[:,1] = x[:,1] - x[:,3] / 2
    y[:,2] = x[:,0] + x[:,2] / 2
    y[:,3] = x[:,1] + x[:,3] / 2
    return y


def nms(dets,thresh):
    x1 = dets[:,0]
    y1 = dets[:,1]
    x2 = dets[:,2]
    y2 = dets[:,3]
    #-------------------------------
    # 计算框的面积
    # 置信度从大到小排序
    #-------------------------------
    areas = (y2 -y1 + 1) * (x2 -x1 + 1)
    scores = dets[:,4]
    keep = []
    index = scores.argsort()[::-1]

    while index.size > 0:
        i = index[0]
        keep.append(i)
        #---------------------------
        # 计算相交面积
        # 1.相交
        # 2.不相交
        #---------------------------
        x11 = np.maximum(x1[i], x1[index[1:]])
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        w = np.maximum(0,x22 - x11 + 1)
        h = np.maximum(0,y22 - y11 + 1)

        overlaps = w *h
        #-----------------------------------------------------
        # 计算该框与其它框的IOU，去除掉重复的框，即IOU值大的框
        # IOU小于thresh的框保留下来
        #-----------------------------------------------------
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
        idx = np.where(ious <= thresh)[0]
        index = index[idx + 1]
    return keep



def load_imagenet_labels(path):
    with open(path) as f:
        label_list = json.load(f)
    return label_list



class Model(ABC):
    def __init__(self, model_path, device_id=0):
        self.model_id = None
        self.output_data = None
        self.load_output_dataset = None
        self.input_data = None
        self.load_input_dataset = None
        self.device_id = device_id
        self.context = None
        self.model_desc = None

        self.model_height = None
        self.model_width = None
        self.model_channel = None
        self.output_shapes = []
        self.output_dtypes = []
        ret = acl.init()
        ret = acl.rt.set_device(self.device_id)
        self.init_acl(model_path)

    def init_acl(self, model_path):
        self.model_id, ret = acl.mdl.load_from_file(model_path)
        if ret:
            raise RuntimeError(ret)

        model_desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(model_desc, self.model_id)
        if ret:
            raise RuntimeError(ret)

        dims, ret = acl.mdl.get_input_dims_v2(model_desc, 0)
        if ret:
            raise RuntimeError(ret)
        dims = dims['dims']

        self.model_channel = dims[1]
        self.model_height = dims[2]
        self.model_width = dims[3]

        self.model_desc = model_desc
        # prepare input data resources for model infer
        self.load_input_dataset = acl.mdl.create_dataset()
        input_size = acl.mdl.get_num_inputs(self.model_desc)
        self.input_data = []
        for i in range(input_size):
            buffer_size = acl.mdl.get_input_size_by_index(self.model_desc, i)
            buffer, ret = acl.rt.malloc(buffer_size, ACL_MEM_MALLOC_HUGE_FIRST)
            data = acl.create_data_buffer(buffer, buffer_size)
            _, ret = acl.mdl.add_dataset_buffer(self.load_input_dataset, data)
            self.input_data.append({"buffer": buffer, "size": buffer_size})

        # prepare output data resources for model infer
        self.load_output_dataset = acl.mdl.create_dataset()
        output_size = acl.mdl.get_num_outputs(self.model_desc)
        self.output_data = []
        for i in range(output_size):
            dims, ret = acl.mdl.get_output_dims(model_desc, i)
            self.output_shapes.append(tuple(dims['dims']))
            data_type = acl.mdl.get_output_data_type(model_desc, i)
            self.output_dtypes.append(dtype_map[data_type])
            buffer_size = acl.mdl.get_output_size_by_index(self.model_desc, i)
            buffer, ret = acl.rt.malloc(buffer_size, ACL_MEM_MALLOC_HUGE_FIRST)
            data = acl.create_data_buffer(buffer, buffer_size)
            _, ret = acl.mdl.add_dataset_buffer(self.load_output_dataset, data)
            self.output_data.append({"buffer": buffer, "size": buffer_size})

        print(f'{self.__class__.__name__} Resources init successfully.')

    @abstractmethod
    def preprocess(self, img):
        pass

    def infer(self, tensor):
        np_ptr = acl.util.bytes_to_ptr(tensor.tobytes())
        # copy input data from host to device
        ret = acl.rt.memcpy(self.input_data[0]["buffer"], self.input_data[0]["size"], np_ptr,
                            self.input_data[0]["size"], ACL_MEMCPY_HOST_TO_DEVICE)

        # infer exec
        ret = acl.mdl.execute(self.model_id, self.load_input_dataset, self.load_output_dataset)

        inference_result = []

        for i, item in enumerate(self.output_data):
            buffer_host, ret = acl.rt.malloc_host(self.output_data[i]["size"])
            # copy output data from device to host
            ret = acl.rt.memcpy(buffer_host, self.output_data[i]["size"], self.output_data[i]["buffer"],
                                self.output_data[i]["size"], ACL_MEMCPY_DEVICE_TO_HOST)

            data = acl.util.ptr_to_bytes(buffer_host, self.output_data[i]['size'])
            data = np.frombuffer(data, self.output_dtypes[i]).reshape(self.output_shapes[i])
            inference_result.append(data)

        return inference_result

    @abstractmethod
    def postprocess(self, output):
        pass

    def deinit(self):
        ret = acl.mdl.unload(self.model_id)
        if ret:
            raise RuntimeError(ret)
        if self.model_desc:
            acl.mdl.destroy_desc(self.model_desc)
            self.model_desc = None

        while self.input_data:
            item = self.input_data.pop()
            ret = acl.rt.free(item["buffer"])
            if ret:
                raise RuntimeError(ret)

        while self.output_data:
            item = self.output_data.pop()
            ret = acl.rt.free(item["buffer"])
            if ret:
                raise RuntimeError(ret)
        print(f'{self.__class__.__name__} Resources released successfully.')




class YOLO(Model):
	def __init__(self, model_path):
		super().__init__(model_path, 0)
		#self.idx2label_list=label_list


	def preprocess(self, img):
		img = cv2.imread(img_path)
		img = cv2.resize(img,(640,640))#,interpolation=cv2.INTER_LINEAR
		#img = cv2.copyMakeBorder(img, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(114, 114, 114))
		img = img[:,:,::-1].transpose(2,0,1) # BGR2RGB和HWC2CHW
		img = img.astype(dtype=np.float32)
		img /=255.0
		img = np.expand_dims(img,axis=0)
		# img = img.unsqueeze(0)
		# img = np.array(img)
		img = np.ascontiguousarray(img).astype(np.float32)
		result = np.frombuffer(img.tobytes(), np.float16)
		return result

	def postprocess(self, outputs, conf_thres=0.4, iou_thres=0.35):
		org_box = outputs[0]
		#------------------------------
		# 删除为1的维度
    # 删除置信度小于conf_thres的BOX
    #------------------------------
		#org_box = np.squeeze(org_box)
		conf = org_box[...,4] > conf_thres
		results = []
		for index ,data in enumerate(org_box):
			box = data[conf[index]]

      #-------------------------------
      # 通过argmax获取置信度最大的类别
      #-------------------------------
			cls_cinf = box[...,5:]
			cls = []
			for i in range(len(cls_cinf)):
				cls.append(int(np.argmax(cls_cinf[i])))
			all_cls = list(set(cls))
      #--------------------------------
      #  分别对每个类别进行过滤
      #  1.将第6列元素替换为类别下标
      #  2.xywh2xyxy 坐标变换
      #  3.经过非极大抑制后输出的BOX下标
      #  4.利用下标取出非极大抑制后的BOX
      #---------------------------------
			output = []
				#print(all_cls)
			for i in range(len(all_cls)):
				curr_cls = all_cls[i]
					
				curr_cls_box = []
				curr_out_box = []
				for j in range(len(cls)):
					if cls[j] == curr_cls:
						box[j][5] = curr_cls
						curr_cls_box.append(box[j][:6])
				curr_cls_box = np.array(curr_cls_box)
          # curr_cls_box_old = np.copy(curr_cls_box)
				curr_cls_box = xywh2xyxy(curr_cls_box)
				curr_out_box = nms(curr_cls_box,iou_thres)
				for k in curr_out_box:
					output.append(curr_cls_box[k])
			output = np.array(output)
			results.append(output)
		return results

def draw(image,box_data):
    #-------------------------
    # 取整，仿编画框
    #-------------------------
    image = cv2.resize(image,(640,640))#,interpolation=cv2.INTER_LINEAR
    boxes = box_data[...,:4].astype(np.int32)
    print("box_data.shape:",box_data.shape)
    scores = box_data[...,4]
    classes = box_data[...,5].astype(np.int32)

    for box,score,cl in zip(boxes,scores,classes):
        top,left,right,bottom = box
        print("class:{},score:{}".format(CLASSES[cl],score))
        print("box coordinate left,top,right,down:[{},{},{},{}]".format(top,left,right,bottom))

        cv2.rectangle(image,(top,left),(right,bottom),(255,0,0),2)
        cv2.putText(image,"{0} {1:2f}".format(CLASSES[cl],score),(top,left),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
    cv2.imwrite("./out.jpg",image)


import time

if __name__ == '__main__':
	img_path = './world_cup.jpg'
	model_path = './yolov7.om'
	label_path = './imagenet-simple-labels.json'
	#idx2label_list = load_imagenet_labels(label_path)
	model = YOLO(model_path)
	img=model.preprocess(img_path)
	outputs = model.infer(img)
	outbox = model.postprocess(outputs)
	draw(cv2.imread(img_path),outbox[0])
#	result = model.postprocess(outputs)
	model.deinit()


