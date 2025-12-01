#ifndef AWGALGORITHM_H
#define AWGALGORITHM_H

#ifndef AWGARRAY_H
#include "AwgArray.hpp"
#endif

namespace Awg {
    ///函数后缀Avx2表示这个函数内部使用了Avx2指令进行硬件计算加速,后缀Tp表示函数内部使用了多线程进行软件计算加速,Fast表示同时使用进行了软件和硬件加速
    ///更新:所有带后缀的函数都移动到cpp文件中,函数在编译器允许的条件下始终会使用更快的算法,对于函数调用者来说不需要再自行区分需要调用哪种算法
    /// 函数内部都做了向量计算和标量计算两种实现以及线程池并行计算(如果可行的话),所以大部分函数的执行效率都是很高的,未使用软件和硬件加速的算法也会在后期补充

    /// 从给定short数组中查找最小值并返回最小值指针,在极端情况下(数组已经按从大到小排序)这个函数的效率可能会略低于std::min_element
    const short* min(const short* begin,const short* end);

    /// 从给定short数组中查找最大值并返回最大值指针,在极端情况下(数组已经按从小到大排序)这个函数的效率可能会略低于std::max_element
    const short* max(const short* begin,const short* end);

    ///从给定的数组中查找最小值和最大值,返回最小值和最大值指针
    std::pair<const short*,const short*> minmax(const short *begin, const short*end);

    ///将short数组中的每一个值取12bit压缩写入到二进制内存中,返回写入的长度,需要保证output长度足够写入全部数据,否则会导致程序崩溃
    void compressShort12Bit(const short* begin,const short* end,char* output);//1024bit,前768bit存放，后256比特不适用

    ///将给定的double数组归一化,并返回一个字节对齐的数据,rangeLow和rangeHigh表示原始数据的范围,min和max表示归一化之后的数据范围
    AwgDoubleArray normalization(const AwgDoubleArray& input,const double inputMin,const double inputMax,const double outputMin,const double outputMax);

    ///数据点超过最大像素点之后按包络生成数据的略缩图
    AwgShortArray generateOverview(const Awg::DT* data,const std::size_t length);

    ///生成正弦波波形
    AwgShortArray generateSin(double sampleRate,double frequency,double phase);
    
    ///生成方波波形
    AwgShortArray generateSquare(double sampleRate,double frequency,double duty);

    ///生成三角波形
    AwgShortArray generateTriangle(double sampleRate,double frequency,double symmetry);

    ///生成噪声波形
    AwgShortArray generateNoise(double sampleRate,double bandWidth);
}

#endif // AWGALGORITHM_H
