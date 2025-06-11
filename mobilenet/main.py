# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import argparse
import numpy as np
import acl
import os
import cv2                   
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from constant import ACL_MEM_MALLOC_HUGE_FIRST, \
    ACL_MEMCPY_HOST_TO_DEVICE, ACL_MEMCPY_DEVICE_TO_HOST, \
    ACL_SUCCESS, IMG_EXT, NPY_FLOAT32

buffer_method = {
    "in": acl.mdl.get_input_size_by_index,
    "out": acl.mdl.get_output_size_by_index
    }

def softmax(data,dim):
	pass
	if dim >= data.ndim or dim<-data.ndim:
		raise IndexError(f'Dimension out of range (expected to be in range of [{-data.ndim}, {data.ndim-1}], but got {dim})')
	max_data = (data.max(axis=dim))[...,np.newaxis]
	data = np.exp(data-max_data)
	x = np.sum(data, axis = dim, keepdims=True)
	return data / x

def check_ret(message, ret):
    if ret != ACL_SUCCESS:
        raise Exception("{} failed ret={}"
                        .format(message, ret))


class Net(object):
    def __init__(self, device_id, model_path, idx2label_list):
        self.device_id = device_id      # int
        self.model_path = model_path    # string
        self.model_id = None            # pointer
        self.context = None             # pointer

        self.input_data = []
        self.output_data = []
        self.model_desc = None          # pointer when using
        self.load_input_dataset = None
        self.load_output_dataset = None
        self.idx2label_list = idx2label_list

        self.init_resource()

    def release_resource(self):
        print("Releasing resources stage:")
        ret = acl.mdl.unload(self.model_id)
        check_ret("acl.mdl.unload", ret)
        if self.model_desc:
            acl.mdl.destroy_desc(self.model_desc)
            self.model_desc = None

        while self.input_data:
            item = self.input_data.pop()
            ret = acl.rt.free(item["buffer"])
            check_ret("acl.rt.free", ret)

        while self.output_data:
            item = self.output_data.pop()
            ret = acl.rt.free(item["buffer"])
            check_ret("acl.rt.free", ret)

        if self.context:
            ret = acl.rt.destroy_context(self.context)
            check_ret("acl.rt.destroy_context", ret)
            self.context = None

        ret = acl.rt.reset_device(self.device_id)
        check_ret("acl.rt.reset_device", ret)
        ret = acl.finalize()
        check_ret("acl.finalize", ret)
        print('Resources released successfully.')

    def init_resource(self):
        print("init resource stage:")
        ret = acl.init()
        check_ret("acl.init", ret)
        ret = acl.rt.set_device(self.device_id)
        check_ret("acl.rt.set_device", ret)

        self.context, ret = acl.rt.create_context(self.device_id)
        check_ret("acl.rt.create_context", ret)

        # load_model
        self.model_id, ret = acl.mdl.load_from_file(self.model_path)
        check_ret("acl.mdl.load_from_file", ret)
        print("model_id:{}".format(self.model_id))

        self.model_desc = acl.mdl.create_desc()
        self._get_model_info()
        print("init resource success")

    def _get_model_info(self,):
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)
        check_ret("acl.mdl.get_desc", ret)
        input_size = acl.mdl.get_num_inputs(self.model_desc)
        output_size = acl.mdl.get_num_outputs(self.model_desc)
        self._gen_data_buffer(input_size, des="in")
        self._gen_data_buffer(output_size, des="out")

    def _gen_data_buffer(self, size, des):
        func = buffer_method[des]
        for i in range(size):
            # check temp_buffer dtype
            temp_buffer_size = func(self.model_desc, i)
            temp_buffer, ret = acl.rt.malloc(temp_buffer_size,
                                             ACL_MEM_MALLOC_HUGE_FIRST)
            check_ret("acl.rt.malloc", ret)

            if des == "in":
                self.input_data.append({"buffer": temp_buffer,
                                        "size": temp_buffer_size})
            elif des == "out":
                self.output_data.append({"buffer": temp_buffer,
                                         "size": temp_buffer_size})

    def _data_interaction(self, dataset, policy=ACL_MEMCPY_HOST_TO_DEVICE):
        temp_data_buffer = self.input_data \
            if policy == ACL_MEMCPY_HOST_TO_DEVICE \
            else self.output_data
        if len(dataset) == 0 and policy == ACL_MEMCPY_DEVICE_TO_HOST:
            for item in self.output_data:
                temp, ret = acl.rt.malloc_host(item["size"])
                if ret != 0:
                    raise Exception("can't malloc_host ret={}".format(ret))
                dataset.append({"size": item["size"], "buffer": temp})

        for i, item in enumerate(temp_data_buffer):
            if policy == ACL_MEMCPY_HOST_TO_DEVICE:
                if "bytes_to_ptr" in dir(acl.util):
                    bytes_data = dataset[i].tobytes()
                    ptr = acl.util.bytes_to_ptr(bytes_data)
                else:
                    ptr = acl.util.numpy_to_ptr(dataset[i])
                ret = acl.rt.memcpy(item["buffer"],
                                    item["size"],
                                    ptr,
                                    item["size"],
                                    policy)
                check_ret("acl.rt.memcpy", ret)

            else:
                ptr = dataset[i]["buffer"]
                ret = acl.rt.memcpy(ptr,
                                    item["size"],
                                    item["buffer"],
                                    item["size"],
                                    policy)
                check_ret("acl.rt.memcpy", ret)

    def _gen_dataset(self, type_str="input"):
        dataset = acl.mdl.create_dataset()

        temp_dataset = None
        if type_str == "in":
            self.load_input_dataset = dataset
            temp_dataset = self.input_data
        else:
            self.load_output_dataset = dataset
            temp_dataset = self.output_data

        for item in temp_dataset:
            data = acl.create_data_buffer(item["buffer"], item["size"])
            _, ret = acl.mdl.add_dataset_buffer(dataset, data)

            if ret != ACL_SUCCESS:
                ret = acl.destroy_data_buffer(data)
                check_ret("acl.destroy_data_buffer", ret)

    def _data_from_host_to_device(self, images):
        print("data interaction from host to device")
        # copy images to device
        self._data_interaction(images, ACL_MEMCPY_HOST_TO_DEVICE)
        # load input data into model
        self._gen_dataset("in")
        # load output data into model
        self._gen_dataset("out")
        print("data interaction from host to device success")

    def _data_from_device_to_host(self):
        print("data interaction from device to host")
        res = []
        # copy device to host
        self._data_interaction(res, ACL_MEMCPY_DEVICE_TO_HOST)
        print("data interaction from device to host success")
        result = self.get_result(res)
        pred_dict = self._print_result(result)
        # free host memory
        for item in res:
            ptr = item['buffer']
            ret = acl.rt.free_host(ptr)
            check_ret('acl.rt.free_host', ret)
        return pred_dict

    def run(self, images):
        self._data_from_host_to_device(images)
        self.forward()
        pred_dict = self._data_from_device_to_host()
        return pred_dict

    def forward(self):
        print('execute stage:')
        ret = acl.mdl.execute(self.model_id,
                              self.load_input_dataset,
                              self.load_output_dataset)
        check_ret("acl.mdl.execute", ret)
        self._destroy_databuffer()
        print('execute stage success')

    def _print_result(self, result):
        vals = np.array(result).flatten()
        vals = softmax(vals,-1)
        top_k = vals.argsort()[-1:-6:-1]
        print("======== top5 inference results: =============")

        for j in top_k:
            print("[%d]: %f" % (j, vals[j]))

        pred_dict = {}
        for j in top_k:
            print(f'{self.idx2label_list[j]}: {vals[j]}')
            pred_dict[self.idx2label_list[j]] = vals[j]
        return pred_dict

    def _destroy_databuffer(self):
        for dataset in [self.load_input_dataset, self.load_output_dataset]:
            if not dataset:
                continue
            number = acl.mdl.get_dataset_num_buffers(dataset)
            for i in range(number):
                data_buf = acl.mdl.get_dataset_buffer(dataset, i)
                if data_buf:
                    ret = acl.destroy_data_buffer(data_buf)
                    check_ret("acl.destroy_data_buffer", ret)
            ret = acl.mdl.destroy_dataset(dataset)
            check_ret("acl.mdl.destroy_dataset", ret)

    def get_result(self, output_data):
        result = []
        dims, ret = acl.mdl.get_cur_output_dims(self.model_desc, 0)
        check_ret("acl.mdl.get_cur_output_dims", ret)
        out_dim = dims['dims']
        for temp in output_data:
            ptr = temp["buffer"]
            # 转化为float32类型的数据
            if "ptr_to_bytes" in dir(acl.util):
                bytes_data = acl.util.ptr_to_bytes(ptr, temp["size"])
                data = np.frombuffer(bytes_data, dtype=np.float32).reshape(tuple(out_dim))
            else:
                data = acl.util.ptr_to_numpy(ptr, tuple(out_dim), NPY_FLOAT32)
            result.append(data)
        return result


def preprocess(img):
#       preprocess = transforms.Compose([
#           transforms.Resize((224,224)),             # 调整图像短边为256像素
#           transforms.ToTensor(),              # 转换为张量，并缩放到[0, 1]范围
#           transforms.Normalize(
#               mean=[0.485, 0.456, 0.406], \
#               std=[0.229, 0.224,0.225])  # 标准化
#       ])
    img = Image.open(img).convert('RGB')
    img = img.resize((224,224))
    #img = cv2.imread(img)
    #img = cv2.resize(img,(224,224))
    img = np.array(img)
    img = img.transpose(2,0,1)
    img = img.astype(np.float32)/255.0
    mean = np.array([0.485, 0.456, 0.406]).reshape(3,1,1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3,1,1)
    img = (img - mean) / std
    img = np.expand_dims(img,axis=0)
    # img = img.unsqueeze(0)
    # img = np.array(img)
    img = np.ascontiguousarray(img).astype(np.float32)
    return img

def load_imagenet_labels(path):
    with open(path) as f:
        label_list = json.load(f)
    return label_list

def display_image(path, pred_dict):
    
    setFont = ImageFont.truetype('font.ttf', 20)
    fillColor = "#fff"
    im = Image.open(path)
    im = im.resize((800, 500))
    draw = ImageDraw.Draw(im)
    
    start_y = 20
    for label, pred in pred_dict.items():
        draw.text(xy = (20, start_y), text=f'{label}: {pred:.2f}', font=setFont, fill=fillColor)
        start_y += 30

    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.imshow(im)
    plt.savefig('result.png')

def append_results_to_file(image_name, predictions, output_file):
    """将推理结果以追加方式写入结果文件"""
    top_k_predictions = ",".join(predictions)
    with open(output_file, "a") as f:
        f.write(f"{image_name}: {top_k_predictions}\n")

def process_and_infer(image_path, net, idx2label_list, output_file):
    """处理单张图片并推理"""
    img = preprocess(image_path)  # 调用 preprocess 函数预处理图像
    pred_dict = net.run([img])    # 推理结果为字典形式，键是类别，值是置信度

    # 获取 Top-K 预测类别
    sorted_preds = sorted(pred_dict.items(), key=lambda x: x[1], reverse=True)
    top_k_predictions = [item[0] for item in sorted_preds[:5]]

    # 保存结果
    append_results_to_file(os.path.basename(image_path), top_k_predictions, output_file)


if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--model_path', type=str,
                        default=os.path.join(current_dir, "mobilenet.om"))
    parser.add_argument('--images_path', type=str,
                        default=os.path.join(current_dir, "./data"))
    args = parser.parse_args()
    print("Using device id:{}\nmodel path:{}\nimages path:{}"
          .format(args.device, args.model_path, args.images_path))

    label_path = './imagenet-simple-labels.json'
    idx2label_list = load_imagenet_labels(label_path)
    net = Net(args.device, args.model_path, idx2label_list)
    images_list = [os.path.join(args.images_path, img)
                   for img in os.listdir(args.images_path)
                   if os.path.splitext(img)[1] in IMG_EXT]
    
    output_file = os.path.join(current_dir,"results.txt")  # 结果文件的路径
    
    for image in images_list:
        print("images:{}".format(image))
        img = preprocess(image)
        pred_dict = net.run([img])
        display_image(image, pred_dict)
        # process_and_infer(image,net,idx2label_list,output_file)
        # os.remove(image)

    print("*****run finish******")
    net.release_resource()
