#include "AwgAlgorithm.h"
#include "UtilitySubSystem/AwgUtility.h"
#include "UtilitySubSystem/ThreadPool.hpp"
#include "AwgDefines.h"

#include <immintrin.h>
#include <iostream>
#include <algorithm>
#include <cmath>

namespace Awg{

    #if __AVX2__
    // 更精确的向量化sin函数,自定义的avx函数必须添加__attribute__,否则在有时候会崩溃,哪怕这个函数什么都不干
    __inline __m256d __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    sinAvx2(const __m256d x)
    {
        const __m256d pi = _mm256_set1_pd(Awg::PI);
        const __m256d half_pi = _mm256_set1_pd(0.5 * Awg::PI);
        const __m256d inv_pi = _mm256_set1_pd(1.0 / Awg::PI);

        // 范围缩减到 [-π/2, π/2]
        __m256d x_mod = _mm256_sub_pd(x, _mm256_mul_pd(_mm256_round_pd(_mm256_mul_pd(x, inv_pi),
                                                                       _MM_FROUND_TO_NEAREST_INT), pi));

        // 使用更精确的多项式系数
        const __m256d s1 = _mm256_set1_pd(-1.6666654611e-1);
        const __m256d s2 = _mm256_set1_pd(8.3321608736e-3);
        const __m256d s3 = _mm256_set1_pd(-1.9515295891e-4);

        __m256d x2 = _mm256_mul_pd(x_mod, x_mod);
        __m256d result = _mm256_mul_pd(x2, s3);
        result = _mm256_add_pd(result, s2);
        result = _mm256_mul_pd(result, x2);
        result = _mm256_add_pd(result, s1);
        result = _mm256_mul_pd(result, x2);
        result = _mm256_mul_pd(result, x_mod);
        result = _mm256_add_pd(result, x_mod);

        return result;
    }
    #endif

    const short* minAvx2(const short *begin, const short *end)
    {
        const std::size_t chunkLeng = 32 / sizeof (short);//32字节除以每一个数据的长度
        const short* min = begin;

        __m256i minMask = _mm256_set1_epi16(*min);
        __m256i minVec = _mm256_set1_epi16(*min);
        __m256i cmpVec = _mm256_set1_epi16(*min);
        __m256i dataVec = _mm256_set1_epi16(*min);
        while (begin+chunkLeng <= end)
        {
            dataVec  = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(begin));//加载数据到寄存器
            cmpVec = _mm256_min_epi16(dataVec,minVec);//将加载到寄存器中的值和最小值寄存器中的值作比较
            minMask = _mm256_cmpgt_epi8(minVec,cmpVec);//将比较结果和最小值寄存器做比较,判断是否产生了新的最小值

            //如果有新的最小值产生,则从这一组数据中找到最小值所在的索引
            if(_mm256_movemask_epi8(minMask))
                min = std::min_element(begin,begin+chunkLeng);
            begin += chunkLeng;
        }

        if(begin == end)
            return min;
        else
        {
            const short* tmpMin = std::min_element(begin,end);
            return *min < *tmpMin ? min : tmpMin;
        }
    }

    const short* maxAvx2(const short *begin, const short *end)
    {
        const std::size_t chunkLeng = 32 / sizeof (short);//32字节除以每一个数据的长度
        const short* max = begin;

        __m256i maxMask = _mm256_set1_epi16(*max);
        __m256i maxVec = _mm256_set1_epi16(*max);
        __m256i cmpVec = _mm256_set1_epi16(*max);
        __m256i dataVec = _mm256_set1_epi16(*max);
        while (begin+chunkLeng <= end)
        {
            dataVec  = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(begin));//加载数据到寄存器
            cmpVec = _mm256_max_epi16(dataVec,maxVec);//将加载到寄存器中的值和最大值寄存器中的值作比较
            maxMask = _mm256_cmpgt_epi8(cmpVec,maxVec);//将比较结果和最大值寄存器做比较,判断是否产生了新的最大值

            //如果有新的最小值产生,则从这一组数据中找到最小值所在的索引
            if(_mm256_movemask_epi8(maxMask))
                max = std::max_element(begin,begin+chunkLeng);
            begin += chunkLeng;
        }

        if(begin == end)
            return max;
        else
        {
            const short* tmpMax = std::max_element(begin,end);
            return *max > *tmpMax ? max : tmpMax;
        }
    }

    std::pair<const short *, const short *> minmaxAvx2(const short *begin, const short *end)
    {
        const std::size_t chunkLeng = 32 / sizeof (short);
        const short* max = begin;
        const short* min = begin;
        __m256i dataVec = _mm256_set1_epi16(*min);
        __m256i minMask = _mm256_set1_epi16(*min);
        __m256i minVec = _mm256_set1_epi16(*min);
        __m256i cmpMinVec = _mm256_set1_epi16(*min);
        __m256i maxMask = _mm256_set1_epi16(*max);
        __m256i maxVec = _mm256_set1_epi16(*max);
        __m256i cmpMaxVec = _mm256_set1_epi16(*max);
        while (begin+chunkLeng <= end)
        {
            dataVec  = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(begin));
            cmpMinVec = _mm256_min_epi16(dataVec,minVec);
            cmpMaxVec = _mm256_max_epi16(dataVec,maxVec);
            minMask = _mm256_cmpgt_epi8(minVec,cmpMinVec);
            maxMask = _mm256_cmpgt_epi8(cmpMaxVec,maxVec);

            if(_mm256_movemask_epi8(minMask))
                min = std::min_element(begin,begin+chunkLeng);
            if(_mm256_movemask_epi8(maxMask))
                max = std::max_element(begin,begin+chunkLeng);
            begin += chunkLeng;
        }

        if(begin == end)
            return std::make_pair(min,max);
        else
        {
            std::pair<const short *, const short *> ret = std::minmax_element(begin,end);
            ret.first = (*ret.first) < (*min) ? ret.first : min;
            ret.second = (*ret.second) > (*max) ? ret.second : max;
            return ret;
        }
    }

    void compressShort12BitScalar(const short *begin, const short *end, char *output)
    {
        //每2个short最终占用3字节内存
        unsigned short first = 0,second = 0;
        while (begin + 2 <= end)
        {
            first = begin[0] & 0xFFF;
            second = begin[1] & 0xFFF;

            output[0] = static_cast<char>(first >> 4);;//static_cast<char>(first << 8);
            output[1] = static_cast<char>((first & 0xF) << 4) |  static_cast<char>(second >> 8);;//static_cast<char>(first << 12) |  static_cast<char>(seconde << 4) ;
            output[2] = static_cast<char>(second & 0xFF); ;//static_cast<char>(second << 8);

            begin += 2;
            output += 3;
        }

        //处理剩下的最后一个数据
        if(begin != end)
        {
            first = (*begin) & 0xFFF;
            output[0] = static_cast<char>(first >> 4);
            output[1] = static_cast<char>((first & 0xF) << 4);
        }
    }

    void compressShort12BitAvx2(const short *begin, const short *end, char *output)
    {
        const std::size_t chunkLeng = 32 / sizeof (short);
        __m256i dataVec = _mm256_setzero_si256();
        __m256i orVec = _mm256_setzero_si256();
        const  __m256i shiftMask = _mm256_setr_epi16(16,1,16,1,16,1,16,1,16,1,16,1,16,1,16,1);
        const __m256i orMask = _mm256_setr_epi8(0,0,0,0xFF,0,0,0,0xFF,0,0,0,0xFF,0,0,0,0xFF,
                                                0,0,0,0xFF,0,0,0,0xFF,0,0,0,0xFF,0,0,0,0xFF);
        const __m256i orShuffle = _mm256_setr_epi8(3,0,1,2,7,4,5,6,11,8,9,10,15,12,13,14,
                                                   3,0,1,2,7,4,5,6,11,8,9,10,15,12,13,14);
        const __m256i bigEndianMask = _mm256_setr_epi8(1,0,2,5,4,6,9,8,10,13,12,14,3,7,11,15,
                                                       1,0,2,5,4,6,9,8,10,13,12,14,3,7,11,15);//按大端模式重排字节顺序
        const __m256i littleEndianMask = _mm256_setr_epi8(0,1,5,2,6,4,8,9,13,10,14,12,3,7,11,15,
                                                          0,1,5,2,6,4,8,9,13,10,14,12,3,7,11,15);//按小端模式重排字节顺序

        __m256i permutexMask = _mm256_setr_epi32(0,1,2,4,5,6,3,7);
        while (begin + chunkLeng <= end)
        {
            dataVec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(begin));
            dataVec = _mm256_mullo_epi16(dataVec,shiftMask);//用乘法代替左移运算
            orVec = _mm256_setzero_si256();//将或运算寄存器置零
            orVec = _mm256_blendv_epi8(orVec,dataVec,orMask);//提取指定位上的8bit数据
            orVec = _mm256_shuffle_epi8(orVec,orShuffle);//洗牌之后需要合并的bit位已经在目标位置上了,直接跟源寄存器做或运算
            dataVec = _mm256_or_si256(dataVec,orVec);//求或运算完成数据拼接
            dataVec = _mm256_shuffle_epi8(dataVec,bigEndianMask);//对拼接好的数据重新排序,可以自行选择大端排序或者小端排序
            dataVec = _mm256_permutevar8x32_epi32(dataVec,permutexMask);//将拼接好的数据放到寄存器前面,无效数据放到寄存器最后
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(output),dataVec);

            //实际上这里写入了12字节的无效数据,所以这里output指针只跳过24
            output += std::size_t(chunkLeng*1.5);//写入到output数组中的数据是24字节
            begin += chunkLeng;//输入数组跳过16个
        }

        //最后处理无法使用AVX指令处理的数据
        compressShort12BitScalar(begin,end,output);
    }

    void normalizationScalar(const double* inputBegin,const double* inputEnd,double* outputIterator,const double inputMin,const double inputMax,const double outputMin,const double outputMax)
    {
        while (inputBegin < inputEnd)
        {
            *outputIterator = (*inputBegin - inputMin)/(inputMax - inputMin)*(outputMax-outputMin) + outputMin;
            ++inputBegin;
            ++outputIterator;
        };
    }

    ///这个函数在仅在当前文件中被调用,因此可以保证inputBegin地址是按32字节对齐的,所以这里不需要额外处理未对齐的数据
    void normalizationAvx2(const double* inputBegin,const double* inputEnd,double* outputIterator,const double inputMin,const double inputMax,const double outputMin,const double outputMax)
    {
        const std::size_t step = Awg::ArrayAlignment / sizeof (double);
        //输出 = (输入 - 输入范围下限) / 输入范围 * 输出范围 + 输出范围下限
        __m256d inputMinVec = _mm256_set1_pd(inputMin);
        __m256d inputRangeVec = _mm256_set1_pd(inputMax - inputMin);
        __m256d outputRangeVec = _mm256_set1_pd(outputMax - outputMin);
        __m256d outputMinVec = _mm256_set1_pd(outputMin);

        while (inputBegin + step <= inputEnd)
        {
            __m256d ret = _mm256_load_pd(inputBegin);
            ret = _mm256_sub_pd(ret,inputMinVec);
            ret = _mm256_div_pd(ret,inputRangeVec);
            ret = _mm256_mul_pd(ret,outputRangeVec);
            ret = _mm256_add_pd(ret,outputMinVec);

            _mm256_store_pd(outputIterator,ret);
            inputBegin+= step;
            outputIterator += step;
        }

        //处理剩下的数据
        normalizationScalar(inputBegin,inputEnd,outputIterator,inputMin,inputMax,outputMin,outputMax);
    }

    ///计算正弦波形数据,数组为output,计算出来的数据写到[beg,end)区间
    void outputSinScalar(std::size_t totalPoints,double phaseRad,const Awg::DT* output,Awg::DT* beg,Awg::DT* end)
    {
        while (beg < end)
        {
            double rad = 2.0 * PI * static_cast<double>(beg - output) / static_cast<double>(totalPoints) + phaseRad;
            *beg = std::round(Amplitude * std::sin(rad));
            ++beg;
        }
    }

    void outputSinAvx2(std::size_t totalPoints,double phaseRad,const Awg::DT* output,Awg::DT* beg,Awg::DT* end)
    {
        constexpr int chunk = 4;
        typename std::aligned_storage<chunk*sizeof (double),32>::type Storage;
        double *buf = reinterpret_cast<double*>(&Storage);
        const  __m256d piMul2 = _mm256_set1_pd(2.0*PI);
        const __m256d phaseRadVec = _mm256_set1_pd(phaseRad);
        const __m256d totalPointsVec = _mm256_set1_pd(totalPoints);
        const __m256d amplVec = _mm256_set1_pd(Awg::Amplitude);
        __m256d retVec = _mm256_set1_pd(0);
        __m256d dataVec = _mm256_set1_pd(0);
        while (beg + chunk <= end)
        {
            //这里不使用load加载数据,因为数组是short类型,参与计算时需要隐式转换为double
            dataVec = _mm256_set_pd(*(beg+0),*(beg+1),*(beg+2),*(beg+3));
            retVec = _mm256_mul_pd(dataVec,piMul2);
            retVec = _mm256_div_pd(retVec,totalPointsVec);
            retVec = _mm256_add_pd(retVec,phaseRadVec);
            retVec = sinAvx2(retVec);
            retVec = _mm256_mul_pd(retVec,amplVec);
            retVec = _mm256_round_pd(retVec, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

            //保存结果,保存结果不使用store的原因同加载
            _mm256_store_pd(buf,retVec);
            *(beg+0)= buf[0];
            *(beg+1) = buf[1];
            *(beg+2) = buf[2];
            *(beg+3) = buf[3];

            beg += chunk;
        }

        // 处理剩余的元素(不足4个的情况)
        outputSinScalar(totalPoints,phaseRad,output,beg,end);
    }

    void outputSquareScalar(const Awg::DT* edge,Awg::DT* beg,Awg::DT* end)
    {
        while (beg < end)
        {
            *beg = (beg < edge) * Awg::Amplitude;//小于占空比索引为高电平,大于为低电平
            ++beg;
        }
    }

    void outputSquareAvx2(const Awg::DT* edge,Awg::DT* beg,Awg::DT* end)
    {
        constexpr int chunk = 32 / sizeof (Awg::DT);
        //Awg::Amplitude的位宽不超过16bit,所以可以直接使用_mm256_set1_epi16加载,否则这样做是不正确的
        const __m256i amplVec = _mm256_set1_epi16(Awg::Amplitude);
        const __m256i zeroVec = _mm256_setzero_si256();

        while (beg + chunk <= end)
        {
            Awg::DT* thunkEnd = beg + chunk;
            if(beg < edge && edge < thunkEnd)// 跨越边界的情况，逐个处理
            {
                while (beg < thunkEnd)
                {
                    *beg = (beg < edge) * Awg::Amplitude;
                    ++beg;
                }
            }
            else
            {
                if(thunkEnd <= edge)// 整个chunk都在edgeIndex左侧
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(beg), amplVec);
                else// 整个chunk都在edgeIndex右侧
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(beg), zeroVec);
                beg += chunk;
            }
        }

        // 处理剩余元素
        outputSquareScalar(edge,beg,end);
    }

    void outputTriangleScalar(double raiseK,double raiseB,double fallK,double fallB,const Awg::DT* output,const Awg::DT* peak,Awg::DT* beg,Awg::DT* end)
    {
        while (beg < end)
        {
            if(beg < peak)
                *beg = raiseK * (beg - output) + raiseB;
            else
                *beg = fallK * (beg - output) + fallB;
            ++beg;
        }
    }

    void outputTriangleAvx2(double raiseK,double raiseB,double fallK,double fallB,const Awg::DT* output,const Awg::DT* peak,Awg::DT* beg,Awg::DT* end)
    {
        constexpr int chunk = 4;
        typename std::aligned_storage<chunk*sizeof (double),32>::type Storage;
        double *buf = reinterpret_cast<double*>(&Storage);
        const __m256d raiseBvec = _mm256_set1_pd(raiseB);
        const __m256d fallBvec = _mm256_set1_pd(fallB);
        const __m256d raiseKvec = _mm256_set1_pd(raiseK);
        const __m256d fallKvec = _mm256_set1_pd(fallK);
        __m256d indexVec = _mm256_setzero_pd();
        __m256d dataVec = _mm256_setzero_pd();
        while (beg + chunk <= end)
        {
            Awg::DT* thunkEnd = beg + chunk;
            if(beg < peak && peak < thunkEnd)
            {
                if(beg < peak)
                    *beg = raiseK * (beg - output) + raiseB;
                else
                    *beg = fallK * (beg - output) + fallB;
                ++beg;
            }
            else
            {
                std::size_t index = beg - output;
                indexVec = _mm256_set_pd(index,index+1,index+2,index+3);
                if(thunkEnd <= peak)
                {
                    dataVec = _mm256_mul_pd(raiseKvec,indexVec);
                    dataVec = _mm256_add_pd(dataVec,raiseBvec);
                }
                else
                {
                    dataVec = _mm256_mul_pd(fallKvec,indexVec);
                    dataVec = _mm256_add_pd(dataVec,fallBvec);
                }

                //保存结果
                _mm256_store_pd(buf,dataVec);
                *(beg+0)= buf[0];
                *(beg+1) = buf[1];
                *(beg+2) = buf[2];
                *(beg+3) = buf[3];

                beg += chunk;
            }

            //处理剩余元素
            outputTriangleScalar(raiseK,raiseB,fallK,fallB,output,peak,beg,end);
        }
    }
}

const short *Awg::min(const short *begin, const short *end)
{
#ifdef __AVX2__
    return Awg::minAvx2(begin,end);
#else
    return std::min_element(begin,end);
#endif
}

const short *max(const short *begin, const short *end)
{
#ifdef __AVX2__
    return Awg::maxAvx2(begin,end);
#else
    return std::max_element(begin,end);
#endif
}

std::pair<const short *, const short *> minmax(const short *begin, const short *end)
{
    std::size_t dataLen = end - begin + 1;
    std::vector<std::size_t> threadChunks = Awg::cutArrayAligned(dataLen,Awg::MinArrayLength,Awg::ArrayAlignment/sizeof (short));
    std::vector< std::future<std::pair<const short *, const short *>> > futures;
    futures.reserve(threadChunks.size());

    ThreadPool* pool = Awg::globalThreadPool();
    for(std::size_t i = 0; i < threadChunks.size(); i++)
    {
#ifdef __AVX2__
        futures[i] = pool->run(Awg::minmaxAvx2,begin,begin+threadChunks[i]);
#else
        futures[i] = pool->run(std::minmax_element<const short*,const short*>,begin,begin+threadChunks[i]);
#endif
        begin += threadChunks[i];
    }
    pool->waitforDone();

    std::pair<const short *, const short *> ret = futures.at(0).get();
    for(std::size_t i = 1; i < futures.size(); i++)
    {
        std::pair<const short *, const short *> p = futures.at(i).get();
        ret.first = (*ret.first) < (*p.first) ? ret.first : p.first;
        ret.second = (*ret.second) > (*p.second) ? ret.second : p.second;
    }

    return ret;
}

void Awg::compressShort12Bit(const short *begin, const short *end, char *output)
{
#ifdef __AVX2__
    Awg::compressShort12BitAvx2(begin,end,output);
#else
    Awg::compressShort12BitScalar(begin,end,output);
#endif
}

AwgDoubleArray Awg::normalization(const AwgDoubleArray &input, const double inputMin, const double inputMax, const double outputMin, const double outputMax)
{
    if(input.empty())
        return AwgDoubleArray{};

    AwgDoubleArray output(input.size());
    std::vector<std::size_t> threadChunks = Awg::cutArrayAligned(input.size(),Awg::MinArrayLength,Awg::ArrayAlignment/sizeof (double));

    ThreadPool* pool = Awg::globalThreadPool();
    double* inputBeg = input.data();
    double* outputBeg = output.data();
    for(std::size_t i = 0; i < threadChunks.size(); i++)
    {
#ifdef __AVX2__
        pool->run(normalizationAvx2,inputBeg,inputBeg+threadChunks[i],outputBeg,inputMin,inputMax,outputMin,outputMax);
#else
        pool->run(normalizationScalar,inputBeg,inputBeg+threadChunks[i],outputBeg,inputMin,inputMax,outputMin,outputMax);
#endif
        inputBeg += threadChunks[i];
        outputBeg += threadChunks[i];
    }
    pool->waitforDone();

    return output;
}

AwgShortArray Awg::generateOverview(const Awg::DT *data, const std::size_t length)
{
    if(length <= Awg::MaxPlotPoints)
    {
        //数据点数不超过最大限制时直接返回传入的数据
        AwgShortArray buf(length);
        memcpy(buf,data,sizeof (Awg::DT)*length);
        return buf;
    }
    else
    {
        //当数据点数超过最大限制时将数据点数压缩到最大限制值的两倍,将数据分为Awg::MaxPlotPoints分别处理
        const std::size_t groupLength = length / Awg::MaxPlotPoints;
        const std::size_t remainLength = groupLength % Awg::MaxPlotPoints;
        AwgShortArray buf(Awg::MaxPlotPoints*2);

        //计算要压缩的每一段数据长度
        std::vector<std::size_t> groupVec;
        groupVec.reserve(Awg::MaxPlotPoints);
        for(std::size_t i = 0; i < Awg::MaxPlotPoints; i++)
        {
            //前面N组每一组多分一个点,确保多余出来的点被均匀分配
            if(i < remainLength)
                groupVec.push_back(groupLength + 1);
            else
                groupVec.push_back(groupLength);
        }

        //计算每一组数据的起始和结束指针
        std::vector<std::pair<const Awg::DT*,const Awg::DT*>> iteratorGroups;
        iteratorGroups.reserve(Awg::MaxPlotPoints);
        const Awg::DT* begin = data;
        for(std::size_t i = 0; i < Awg::MaxPlotPoints; i++)
        {
            iteratorGroups.emplace_back(begin,begin+groupVec.at(i));
            begin += groupVec.at(i);
        }

        //线程任务
        auto task = [&iteratorGroups,&buf](std::size_t startIndex,std::size_t endIndex)
        {
            for(std::size_t i = startIndex ; i < endIndex; i++)
            {
                //开始计算每一组数据,找出每一个分组数据中的最大值和最小值,注意最大值最小值出现顺序,不可交换这两个值的顺序
                std::pair<const Awg::DT*,const Awg::DT*> pair = iteratorGroups[i];
#ifdef __AVX2__
                std::pair<const Awg::DT*,const Awg::DT*> result = Awg::minmaxAvx2(pair.first,pair.second);
#else
                std::pair<const Awg::DT*,const Awg::DT*> result = std::minmax_element(pair.first,pair.second);
#endif
                if(result.first < result.second)
                {
                    buf[2*i] = (*result.first);
                    buf[2*i+1] = (*result.second);
                }
                else
                {
                    buf[2*i] = (*result.second);
                    buf[2*i+1] =(*result.first);
                }
            }
        };

        //按分出来的组groupVec划分到线程池中,计算每一个线程处理多少组数据
        std::vector<std::size_t> threadGroups = Awg::cutArrayMin(Awg::MaxPlotPoints,500);
        ThreadPool* pool = Awg::globalThreadPool();
        std::size_t startIndex = 0;
        for(std::size_t i = 0; i < threadGroups.size(); i++)
        {
            pool->run(task,startIndex,startIndex+threadGroups[i]);
            startIndex += threadGroups[i];
        }

        pool->waitforDone();

        return buf;
    }
}

AwgShortArray Awg::generateSin(double sampleRate, double frequency, double phase)
{
    // 计算一个完整周期内的采样点数
    const std::size_t totalPoints = std::round(sampleRate / frequency);
    const double phaseRad = phase * PI / 180.0;
    
    if (totalPoints == 0)
    {
        std::cerr << "Error: sampleRate must be greater than frequency" << std::endl;
        return AwgShortArray{};
    }
    
    // 分配内存存储波形数据
    AwgShortArray waveform(totalPoints);
    
    if (waveform == nullptr)
    {
        std::cerr << "Error: Memory allocation failed" << std::endl;
        return AwgShortArray{};
    }
    
    //每一个线程处理的点数为总点数/线程数再向上取整
    std::size_t index = 0;
    std::vector<std::size_t> chunks = Awg::cutArrayMin(totalPoints,Awg::MinArrayLength);
    ThreadPool* pool = Awg::globalThreadPool();
    for(std::size_t i = 0; i < chunks.size(); i++)
    {
        Awg::DT* output = waveform.data();
        Awg::DT* beg = output + index;
        Awg::DT* end = beg + chunks[i];
#ifdef __AVX2__
        pool->run<ThreadPool::Ordered>(Awg::outputSinAvx2,totalPoints,phaseRad,output,beg,end);
#else
        pool->run<ThreadPool::Ordered>(Awg::outputSinScalar,totalPoints,phaseRad,output,beg,end);
#endif
        index += chunks[i];
    }
    
    pool->waitforDone();

    return waveform;
}

AwgShortArray Awg::generateSquare(double sampleRate, double frequency, double duty)
{
    //每个周期最少100个点,这样可以将占空比的精度控制到1%
    const unsigned minPointsPerPeriod = 100;
    const std::size_t totalPoints = std::round(sampleRate / frequency);
    if(totalPoints < minPointsPerPeriod)
        return AwgShortArray{};

    // 分配内存存储波形数据
    AwgShortArray waveform(totalPoints);

    //将占空比限制到0和1之间
    duty = std::max(duty,0.0);
    duty = std::min(duty,1.0);
    std::size_t edgeIndex = std::round(totalPoints * duty);
    Awg::DT* output = waveform.data();
    Awg::DT* edge = output + edgeIndex;

    //每一个线程处理的点数为总点数/线程数再向上取整
    std::size_t index = 0;
    std::vector<std::size_t> chunks = Awg::cutArrayMin(totalPoints,Awg::MinArrayLength);
    ThreadPool* pool = Awg::globalThreadPool();
    for(std::size_t i = 0; i < chunks.size(); i++)
    {
        Awg::DT* beg = output + index;
        Awg::DT* end = beg + chunks[i];
#ifdef __AVX2__
        pool->run<ThreadPool::Ordered>(outputSquareAvx2,edge,beg,end);
#else
        pool->run<ThreadPool::Ordered>(outputSquareScalar,edge,beg,end);
#endif
        index += chunks[i];
    }

    pool->waitforDone();

    return waveform;
}

AwgShortArray Awg::generateTriangle(double sampleRate, double frequency, double symmetry)
{
    //每个周期最少100个点,这样可以将对称性的精度控制到1%
    const unsigned minPointsPerPeriod = 100;
    const std::size_t totalPoints = std::round(sampleRate / frequency);
    if(totalPoints < minPointsPerPeriod)
        return AwgShortArray{};

    // 分配内存存储波形数据
    AwgShortArray waveform(totalPoints);

    //将占空比限制到0和1之间
    symmetry = std::max(symmetry,0.0);
    symmetry = std::min(symmetry,1.0);

    struct Points
    {
        double x = 0;
        double y = 0;
    };
    std::size_t peakIndex = std::round(totalPoints * symmetry);

    //这里需要处理对称性为0或者1的情况
    double raiseK = 1 ,raiseB = 0,fallK = 1,fallB = 0;

    if(symmetry == 0)
    {
        fallK = double(0 - Awg::Amplitude) / (totalPoints - 0);
        fallB = Awg::Amplitude;
    }
    else if (symmetry == 1)
    {
        raiseK = double(Awg::Amplitude - 0) / (totalPoints - 0);
        raiseB = 0;
    }
    else
    {
        raiseK =double(Awg::Amplitude - 0)/ (peakIndex - 0);
        raiseB = Awg::Amplitude - raiseK * peakIndex;
        fallK = double(0 - Awg::Amplitude) / (totalPoints - peakIndex);
        fallB =  Awg::Amplitude - fallK*peakIndex;
    }
    Awg::DT* output = waveform.data();
    Awg::DT* peak = output + peakIndex;

    //每一个线程处理的点数为总点数/线程数再向上取整
    std::size_t index = 0;
    std::vector<std::size_t> chunks = Awg::cutArrayMin(totalPoints,Awg::MinArrayLength);
    ThreadPool* pool = Awg::globalThreadPool();
    for(std::size_t i = 0; i < chunks.size(); i++)
    {
        Awg::DT* beg = output + index;
        Awg::DT* end = beg + chunks[i];
#ifdef __AVX2__
        pool->run<ThreadPool::Ordered>(Awg::outputTriangleAvx2,raiseK,raiseB,fallK,fallB,output,peak,beg,end);
#else
        pool->run<ThreadPool::Ordered>(Awg::outputTriangleScalar,raiseK,raiseB,fallK,fallB,output,peak,beg,end);
#endif
        index += chunks[i];
    }

    pool->waitforDone();
    return waveform;
}

