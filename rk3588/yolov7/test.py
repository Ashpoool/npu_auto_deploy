from rknn.api import RKNN

ONNX_MODEL = 'yolov7.onnx'
RKNN_MODEL = 'yolov7.rknn'

DATASET = './dataset.txt'

QUANTIZE_ON = True


if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN(verbose=True)

    # pre-process config
    print('--> Config model')
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],target_platform='rk3588')
    print('done')

    # Load ONNX model
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL, outputs=['output', '262', '263'])
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.hybrid_quantization_step1(dataset='./dataset.txt') 
    #ret = rknn.build(do_quantization=QUANTIZE_ON, dataset=DATASET)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    ret = rknn.hybrid_quantization_step2(model_input='./yolov7.model',
    data_input='./yolov7.data',
    model_quantization_cfg='./yolov7.quantization.cfg')    
    print('done')

    # Export RKNN model
    print('--> Export rknn model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

 
    rknn.release()
