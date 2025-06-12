import math
from abc import abstractmethod, ABC

import acl
import cv2
import numpy as np
import json
from PIL import Image

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


def load_imagenet_labels(path):
    with open(path) as f:
        label_list = json.load(f)
    return label_list

def softmax(data,dim):
	pass
	if dim >= data.ndim or dim<-data.ndim:
		raise IndexError(f'Dimension out of range (expected to be in range of [{-data.ndim}, {data.ndim-1}], but got {dim})')
	max_data = (data.max(axis=dim))[...,np.newaxis]
	data = np.exp(data-max_data)
	x = np.sum(data, axis = dim, keepdims=True)
	return data / x



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




class MobileNet(Model):
	def __init__(self, model_path,label_list):
		super().__init__(model_path, 0)
		self.idx2label_list=label_list


	def preprocess(self, img):
#		preprocess = transforms.Compose([
#		    transforms.Resize((224,224)),             # 调整图像短边为256像素
#		    transforms.ToTensor(),              # 转换为张量，并缩放到[0, 1]范围
#			transforms.Normalize(
#				mean=[0.485, 0.456, 0.406], \
#				std=[0.229, 0.224,0.225])  # 标准化
#		])
		img = Image.open(img)
		img = img.resize((224,224),2)
		#img = cv2.imread(img)
		#img = cv2.resize(img,(224,224))
		img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img = img.transpose(2,0,1)/255
		img = (img-np.array([[[0.485]], [[0.456]], [[0.406]]]))/np.array([[[0.229]], [[0.224]],[[0.225]]])
		img = img.astype(dtype=np.float32)
		img = np.expand_dims(img,axis=0)
		# img = img.unsqueeze(0)
		# img = np.array(img)
		img = np.ascontiguousarray(img).astype(np.float32)
		result = np.frombuffer(img.tobytes(), np.float16)
		return result

	def postprocess(self, outputs):
		"""打印推理结果"""
		results = np.array(outputs)
		#print(results)
		results=softmax(results,-1)
		results = results.squeeze(0)
		topk_s = results.argsort(axis =1)[...,-1:-6:-1]
		topk_s = topk_s[-1:-6:-1]
		print(topk_s.shape)
		for result, top_k in zip(results,topk_s):
			print("======== top5 inference results: =============")

			pred_dict = {}
			for j in top_k:
				print(f'{self.idx2label_list[j]}: {result[j]}')
				pred_dict[self.idx2label_list[j]] = result[j]
		return pred_dict





if __name__ == '__main__':

	img_path = './v2-c96b88ba953b84936cfeed0bb613a822_r.jpg'
	model_path = './mobilenet.om'
	label_path = './imagenet-simple-labels.json'
	idx2label_list = load_imagenet_labels(label_path)
	model = MobileNet(model_path, idx2label_list)
	img=model.preprocess(img_path)
	print(img)
	outputs = model.infer(img)
	result = model.postprocess(outputs)
	model.deinit()


