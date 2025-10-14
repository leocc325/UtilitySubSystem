#ifndef AWGDEFINES_H
#define AWGDEFINES_H

namespace Awg
{
    ///最终发送给FPGA时波形数据类型,FPGA中每个点12bit,所以这里用short保存即可
    /// 代码使用了AVX2指令对数据处理进行加速,但是对于不同的数据类型,所用到的AVX2指令以及流程会不一样
    /// 现在所有使用AVX2指令的函数都是按处理short数据的方式书写的,如果更改了DT的数据类型,需要检查 全部后缀带AVX2指令函数,避免程序执行出错
    using DT = short;

    ///发送给FPGA的数据位数
    constexpr unsigned FPGAbits = 12;

    ///线程池大小,建议将这个值设置为核心数-2,预留两个核心给操作系统
    constexpr unsigned PoolSize = 4;

    ///数组字节对齐数
    constexpr int ArrayAlignment = 32;

    ///pi
    constexpr double PI = 3.14159265358979323846;

    ///波形的最大幅度
    constexpr int Amplitude = 2 << (FPGAbits - 1);

    ///波形图最大绘制点数,当波形点数超过MaxPlotPoints之后抽值将点数压缩到MaxPlotPoints*2的点数显示
    constexpr unsigned MaxPlotPoints = 10 * 1000;

    ///加载文件时分块加载的最小尺寸 100M
    constexpr unsigned MinFileChunk = 100 * 1024 *1024;

    ///多线程处理数组时最小分块长度
    constexpr unsigned MinArrayLength = ArrayAlignment * 100;

}

#endif // AWGDEFINES_H
