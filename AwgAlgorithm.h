#ifndef AWGALGORITHM_H
#define AWGALGORITHM_H

#ifndef _GLIBCXX_CXX_CONFIG_H
#include "bits/c++config.h"
#endif

#ifndef AWGDEFINES_H
#include "AwgDefines.h"
#endif

#ifndef AWGARRAY_H
#include "AwgArray.hpp"
#endif

namespace Awg {
    ///函数后缀Avx2表示这个函数内部使用了Avx2指令进行硬件计算加速,后缀Tp表示函数内部使用了多线程进行软件计算加速,Fast表示同时使用进行了软件和硬件加速

    /// 从给定short数组中查找最小值并返回最小值指针
    const short* minAvx2(const short* begin,const short* end);

    /// 从给定short数组中查找最大值并返回最大值指针
    const short* maxAvx2(const short* begin,const short* end);

    ///从给定的数组中查找最小值和最大值,返回最小值和最大值指针
    std::pair<const short*,const short*> minmaxAvx2(const short *begin, const short*end);

    ///从给定的数组中查找最小值和最大值,返回最小值和最大值指针
    std::pair<const short *, const short *> minmaxFast(const AwgShortArray& input);

    ///将给定的double数组归一化,并返回一个字节对齐的数据,rangeLow和rangeHigh表示原始数据的范围,min和max表示归一化之后的数据范围
    AwgDoubleArray normalizationFast(const AwgDoubleArray& input,const double inputMin,const double inputMax,const double outputMin,const double outputMax);

    ///数据点超过最大像素点之后按包络生成数据的略缩图
    AwgShortArray generateOverview(const Awg::DT* data,const std::size_t length);

    ///数据点超过最大像素点之后按包络生成数据的略缩图
    AwgShortArray generateOverviewTp(const Awg::DT* data,const std::size_t length);

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
