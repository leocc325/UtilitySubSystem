#ifndef AWGDEFINES_H
#define AWGDEFINES_H

namespace Awg
{
    ///线程池大小,建议将这个值设置为核心数-2,预留两个核心给操作系统
    constexpr unsigned PoolSize = 16;

    ///数组字节对齐数
    constexpr int ArrayAlignment = 32;

    ///pi
    constexpr double PI = 3.14159265358979323846;

    ///波形图最大绘制点数,当波形点数超过MaxPlotPoints之后抽值将点数压缩到MaxPlotPoints*2的点数显示
    constexpr unsigned MaxPlotPoints = 10 * 1000;

    ///加载文件时分块加载的最小尺寸 100M
    constexpr unsigned MinFileChunk = 100 * 1024 *1024;

    ///多线程处理数组时最小分块长度
    constexpr unsigned MinArrayLength = ArrayAlignment * 100;

    ///波形略缩图文件格式
    static const char* PixFileFormat = ".png";

    ///任意波采样率[FPGA下面跟频率相关的都是1000进制]
    constexpr double FpgaClockMax = 20e9;

    ///fpga内部可以保存的波形数据点的点数(暂时假定为1G)[FPGA下面跟点数相关的都是1024进制]
    constexpr double FpgaPointsSize = 1*1024*1024*1024;

    ///发送给FPGA的数据位数
    constexpr unsigned FPGAbits = 12;

    ///波形的最大幅度
    constexpr int Amplitude = 2 << (FPGAbits - 1);

    ///DMA地址
    constexpr unsigned long long DMAaddress = 0xC0000000;

    ///DMA数据缓冲区大小(字节)
    constexpr unsigned long long DMAsize = 32 * 1024;

    ///读取MCU的数据缓冲区大小
    constexpr unsigned long long McuBufSize = 32 * 1024;

    ///Mcu上传的数据帧大小
    constexpr unsigned long long McuFrameSize = 8;

    ///根据波形设置频率和fpga存储深度计算时钟频率
    inline double calculateFpgaClock(const double Fs)
    {
        ///假如计算出来的点数超过了FGPA能保存的最大点数,就重新调整点数,最后根据调整的点数重新设置FPGA时钟
        ///这里的调整点数的含义是:调整每个点在FPGA中的输出次数
        ///假如FPGA时钟就是1G,FPGA能保存的最大点数为1M,所以满点数的情况下只能输出1KHz信号,假如需要输出100Hz信号,则需要10M个点
        ///等价于:将1M个点每个点重复10次输出,在点数和输出频率不变的情况下,将FPGA时钟调整为100M
        ///Fc:fpga时钟频率  Fs:用户设置的波形输出频率  P:波形总长度点数
        ///三者的关系满足:Fs = P * Fc
        double repeat = 1;//每个点的输出次数
        double P = Awg::FpgaClockMax / Fs;
        while( (P / repeat) > Awg::FpgaPointsSize)
        {
            ++repeat;
        }

        double Fc = Fs * (P / repeat);
        return Fc;
    }

}

#endif // AWGDEFINES_H
