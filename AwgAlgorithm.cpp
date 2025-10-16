#include "AwgAlgorithm.h"
#include "UtilitySubSystem/AwgUtility.h"
#include "UtilitySubSystem/ThreadPool.hpp"

#include <immintrin.h>
#include <iostream>
#include <algorithm>
#include <cmath>

const short *Awg::minAvx2(const short *begin, const short *end)
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

const short *Awg::maxAvx2(const short *begin, const short *end)
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

std::pair<const short *, const short *> Awg::minmaxAvx2(const short *begin, const short *end)
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

std::pair<const short *, const short *> Awg::minmaxFast(const AwgShortArray &input)
{
    if(input.empty())
        return std::pair<const short *, const short *>{nullptr,nullptr};

    const short* begin = input.data();
    std::vector<std::size_t> threadChunks = Awg::cutArrayAligned(input.size(),Awg::MinArrayLength,Awg::ArrayAlignment/sizeof (short));
    std::vector< std::future<std::pair<const short *, const short *>> > futures;
    futures.reserve(threadChunks.size());

    ThreadPool* pool = Awg::globalThreadPool();
    for(std::size_t i = 0; i < threadChunks.size(); i++)
    {
        futures[i] = pool->run(minmaxAvx2,begin,begin+threadChunks[i]);
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

void Awg::compressShort12BitAvx2(const short *begin, const short *end, char *output)
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
    compressShort12Bit(begin,end,output);
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
    while (inputBegin < inputEnd)
    {
        *outputIterator = (*inputBegin - inputMin)/(inputMax - inputMin)*(outputMax-outputMin) + outputMin;
        ++inputBegin;
        ++outputIterator;
    };
}

AwgDoubleArray Awg::normalizationFast(const AwgDoubleArray &input, const double inputMin, const double inputMax, const double outputMin, const double outputMax)
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
        pool->run(normalizationAvx2,inputBeg,inputBeg+threadChunks[i],outputBeg,inputMin,inputMax,outputMin,outputMax);
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

        //计算每一组数据长度
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

        //开始计算每一组数据,找出每一个分组数据中的最大值和最小值,注意最大值最小值出现顺序,不可交换这两个值的顺序
        const Awg::DT* begin = data;
        const Awg::DT* end = data;
        for(std::size_t i = 0; i < Awg::MaxPlotPoints; i++)
        {
            end += groupVec.at(i);

            std::pair<const Awg::DT*,const Awg::DT*> result = std::minmax_element(begin,end);
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

            begin += groupVec.at(i);
        }

        return buf;
    }
}

AwgShortArray Awg::generateOverviewTp(const Awg::DT *data, const std::size_t length)
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
                std::pair<const Awg::DT*,const Awg::DT*> result = std::minmax_element(pair.first,pair.second);
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
        std::vector<std::size_t> threadGroups = Awg::cutArray(Awg::MaxPlotPoints,500);
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
    
    auto task = [totalPoints,phaseRad](Awg::DT* data,std::size_t start,std::size_t end)
    {
        for(std::size_t i = start; i < end; ++i)
        {
            double rad = 2.0 * PI * static_cast<double>(i) / static_cast<double>(totalPoints) + phaseRad;
            data[i] = static_cast<Awg::DT>(Amplitude * std::sin(rad));
        }
    };
    
    //每一个线程处理的点数为总点数/线程数再向上取整
    std::size_t index = 0;
    std::vector<std::size_t> chunks = Awg::cutArray(totalPoints,Awg::MinArrayLength);
    ThreadPool* pool = Awg::globalThreadPool();
    for(std::size_t i = 0; i < chunks.size(); i++)
    {
        pool->run<ThreadPool::Ordered>(task,waveform,index,index+chunks[i]);
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

    auto task = [edgeIndex](Awg::DT* data,std::size_t start,std::size_t end)
    {
        for(std::size_t i = start; i < end; ++i)
        {
            data[i] = (i < edgeIndex) * Awg::Amplitude;//小于占空比索引为高电平,大于为低电平
        }
    };

    //每一个线程处理的点数为总点数/线程数再向上取整
    std::size_t index = 0;
    std::vector<std::size_t> chunks = Awg::cutArray(totalPoints,Awg::MinArrayLength);
    ThreadPool* pool = Awg::globalThreadPool();
    for(std::size_t i = 0; i < chunks.size(); i++)
    {
        pool->run<ThreadPool::Ordered>(task,waveform,index,index+chunks[i]);
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

    auto task = [peakIndex,raiseK,raiseB,fallK,fallB](Awg::DT* data,std::size_t start,std::size_t end)
    {
        for(std::size_t i = start; i < end; ++i)
        {
            if(i < peakIndex)
                data[i] = raiseK * i + raiseB;
            else
                data[i] = fallK * i + fallB;
        }
    };

    //每一个线程处理的点数为总点数/线程数再向上取整
    std::size_t index = 0;
    std::vector<std::size_t> chunks = Awg::cutArray(totalPoints,Awg::MinArrayLength);
    ThreadPool* pool = Awg::globalThreadPool();
    for(std::size_t i = 0; i < chunks.size(); i++)
    {
        pool->run<ThreadPool::Ordered>(task,waveform,index,index+chunks[i]);
        index += chunks[i];
    }

    pool->waitforDone();

    return waveform;
}
