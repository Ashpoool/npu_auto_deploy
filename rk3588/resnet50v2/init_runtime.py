from rknn.api import RKNN

if __name__ == "__main__":
    rknn = RKNN()

    # 使用load_rknn接口导入RKNN模型
    rknn.load_rknn(path="./resnet50v2.rknn")

    # 使用init_runtime接口初始化运行环境
    rknn.init_runtime(
        # target = None,
        target="rk3588",
        
        perf_debug = True,          # 表示是否开启性能评估的Debug模式
        eval_mem = True,            # 表示是否是内存评估
    )

    # 使用eval_memory接口进行内存评估
    rknn.eval_memory(
        is_print=True,   # 表示使能打印内存评估信息
    )

    rknn.release()

