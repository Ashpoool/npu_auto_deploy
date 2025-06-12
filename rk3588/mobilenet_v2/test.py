from rknn.api import RKNN

ONNX_MODEL = 'mobilenet_v2.onnx'
RKNN_MODEL = 'mobilenet.rknn'


if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN(verbose=True)

    # pre-process config
    print('--> config model')
    rknn.config(mean_values=[103.94, 116.78, 123.68], std_values=[58.82, 58.82, 58.82],target_platform='rk3588')
    print('done')   # mean_values=[123.675, 116.28, 103.53], std_values=[58.82, 58.82, 58.82]

    # Load model
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=False, dataset='./dataset.txt')
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')


    rknn.release()
