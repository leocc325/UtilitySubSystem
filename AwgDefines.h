#ifndef AWGDEFINES_H
#define AWGDEFINES_H

namespace Awg
{
    ///底层数据类型
    using DT = float;

    ///线程池大小,建议将这个值设置为核心数-2,预留两个核心给操作系统
    constexpr unsigned PoolSize = 10;

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
    constexpr int Amplitude = (2 << (FPGAbits - 1)) - 1;

    ///DMA地址
    constexpr unsigned long long DMAaddress = 0xC0000000;

    ///DMA数据缓冲区大小(字节)
    constexpr unsigned long long DMAsize = 32 * 1024;

    ///读取MCU的数据缓冲区大小
    constexpr unsigned long long McuBufSize = 32 * 1024;

    ///Mcu上传的数据帧大小
    constexpr unsigned long long McuFrameSize = 8;

    ///根据波形设置频率和fpga时钟频率计算实际生成的点数N和每个点的重复次数,适用于软件自身生成的波形
    inline bool calculateFpgaClock_G(const double Fs,double& Re,double& Fc,double& N)
    {
        ///假如计算出来的点数超过了FGPA能保存的最大点数,就重新调整点数,最后根据调整的点数重新设置FPGA时钟
        ///这里的调整点数的含义是:调整每个点在FPGA中的输出次数
        ///假如FPGA时钟就是1G,FPGA能保存的最大点数为1M,所以满点数的情况下只能输出1KHz信号,假如需要输出100Hz信号,则需要10M个点
        ///等价于:将1M个点每个点重复10次输出,在点数和输出频率不变的情况下,将FPGA时钟调整为100M
        ///N:波形长度  Fc:Fpga时钟频率 Fs:用户设置的波形输出频率 Re:每一个点重复次数
        ///Fs = Fc / (Re * N)  =>  Fc = Fs * N * Re

        if(Fs > Awg::FpgaClockMax/2)
            return false;

        Re = 1;
        N = Awg::FpgaClockMax / Fs;
        while( N  > Awg::FpgaPointsSize)
        {
            Re = Re * 2;//repeat值只能是2的N次幂
            N = N / 2;
        }

        Fc = Fs * N * Re;
        return true;
    }

    ///根据波形设置频率和数据长度计算fpga时钟频率N和每个点的重复次数,适用于从文件加载的波形
    inline bool calculateFpgaClock_F(const double Fs,const double N,double& Re,double& Fc)
    {
        /// N=1Gpts Fs=1Hz > 1*2 > 1*4 > 1*8 > 1*16 === 停止,每个点重复16次将时钟设为16G,此时输出频率为1Hz
        /// N=3Gpts Fs=0.5Hz > 3*2*0.5 > 3*4*0.5 > 3*8*0.5 === 停止,每个点重复8次将时钟设置为12G,此时输出频率为0.5Hz
        /// N=2.5Gpts Fs=0.1Hz > 2.5*1*0.1 > 2.5*2*0.1 > 2.5*4*0.1 > 2.5*8*0.1 > 2.5*16*0.1 > 2.5*32*0.1  > 2.5*64*0.1 ===停止,每个点重复64次并且将时钟频率设置为16G,此时输出频率为0.1Hz
        /// 先调整每个点的重复次数,直到其对应的输出频率落在[20G,10G]区间,然后将这个频率设置为时钟频率即可

        //如果目标频率和点数已经大于fpga时钟最高频率则直接返回false
        if(Fs* N > Awg::FpgaClockMax)
            return false;

        Re = 1;

        auto freq = [&]()->double{
            return N * Fs * Re;
        };

        //调整重复次数使输出频率能落在10G-20G之间
        while( freq() < Awg::FpgaClockMax/2)
        {
            Re = Re * 2;//repeat值只能是2的N次幂
        }

        //如果最后一次重复次数增加导致输出频率超出了20G,则说明没有对应的设置能让信号按预定频率输出
        if(freq() > Awg::FpgaClockMax)
            return false;
        else
        {
            Fc = freq();
            return true;
        }
    }

}

#endif // AWGDEFINES_H
