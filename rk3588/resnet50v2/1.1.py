import os
import cv2
import numpy as np
from rknnlite.api import RKNNLite

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)  # only difference

def show_outputs(outputs):
    output = outputs[0][0]
    probabilities = softmax(output)
    top_indices = np.argsort(-probabilities)[:5] 
    top5_str = 'resnet50v2\n-----TOP 5-----\n'
    for i in top_indices:
        value = probabilities[i]
        topi = f'{i}: {value:.4f}\n'  
        top5_str += topi
    print(top5_str)

def process_images(folder_path):
    # Load model
    rknn = RKNNLite(verbose=False)
    rknn.load_rknn('./1.1.rknn')
    
    # Initialize runtime environment
    print('--> Init runtime environment')
    if rknn.init_runtime() != 0:
        print('Init runtime environment failed!')
        exit(1)
    print('done')

    # Process each image in the folder
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to load image {image_name}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0)
        
        # Inference
        print(f'--> Running model on {image_name}')
        outputs = rknn.inference(inputs=[img])
        if outputs:
            print(f'Results for {image_name}:')
            show_outputs(outputs)
        else:
            print(f'Failed to run inference on {image_name}')
        print('done')
    
    # Release resources
    rknn.release()

if __name__ == '__main__':
    images_folder = './images'
    process_images(images_folder)

