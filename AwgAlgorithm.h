#ifndef AWGALGORITHM_H
#define AWGALGORITHM_H

#include "AwgArray.hpp"
#include "UtilitySubSystem/xsimd/xsimd.hpp"
#include "UtilitySubSystem/ThreadPool.hpp"
#include "UtilitySubSystem/AwgUtility.h"

class QFile;
namespace Awg
{
    ///2026.1.23使用xsimd库重写了算法接口,并且将底层数据类型换位float而非double,旧的接口在AwgAlgorithm.old.h文件中

    ///判断字符c是否是浮点数的开头
    bool isFloatBegin(char c) noexcept;

    ///判断字符c是否是整数的开头
    bool isIntegerBegin(char c) noexcept;

    ///计算字符串中出现字符target的次数
    std::size_t countChar(const char* beg, const char* end, char target) noexcept;
    std::size_t countCharParallel(const char* beg, const char* end, char target) noexcept;

    ///查找字符串中第一次出现字符target的位置,如果未找到匹配的字符则返回空指针,否则返回字符指针
    const char* findChar(const char* beg,const char* end, char target) noexcept;
    const char* findCharParallel(const char* beg,const char* end, char target) noexcept;

    ///将给定的char数组按字节反序
    void reverse(char* beg,char* end);

    ///将short数组中的每一个值取12bit压缩写入到二进制内存中,返回写入的长度,需要保证output长度足够写入全部数据,否则会导致程序崩溃
    void compressShort12Bit(const short* beg,const short* end,char* output);//1024bit,前768bit存放，后256比特不适用

    ///将给定的float数组归一化,并返回一个字节对齐的数据,rangeLow和rangeHigh表示原始数据的范围,min和max表示归一化之后的数据范围
    AwgFloatArray normalization(const AwgFloatArray& input,const float inputMin,const float inputMax,const float outputMin,const float outputMax);
    AwgFloatArray normalizationParallel(const AwgFloatArray& input,const float inputMin,const float inputMax,const float outputMin,const float outputMax);

    ///根据线程池线程数将文本文件切割成若干个大小至少为minCunk的小内存块,同时保证里面的数据不被切割到不同的块中,返回每一个块的大小
    std::vector<std::size_t> cutTextFile(QFile& file, std::size_t minChunk, const std::vector<char>& spliters);

    ///根据线程池线程数将二进制文件切割成若干个大小至少为minCunk的小内存块,同时保证里面的数据不被切割到不同的块中,不同的块的大小为dataBytes的整倍数,返回每一个块的大小
    std::vector<std::size_t> cutBinaryFile(const std::size_t fileSize,const std::size_t minChunk,const unsigned dataBytes) noexcept;

    ///根据线程池线程数将一个长度为length块切割成若干个大小至少为minChunk的小数组,返回这些数组的长度
    std::vector<std::size_t> splitLengthMin(std::size_t length,std::size_t minChunk) noexcept;

    ///将一个长度为length块切割成若干个大小至多为maxChunk的小数组,返回这些数组的长度
    std::vector<std::size_t> splitLengthMax(std::size_t length,std::size_t maxChunk) noexcept;

    ///根据线程池线程数将一个长度为length块切割成若干个大小至少为minChun而且为aligned整倍数的小数组,返回这些数组的长度
    std::vector<std::size_t> splitLengthAligned(std::size_t length,std::size_t minChunk,std::size_t aligned) noexcept;

    ///生成正弦波波形
    AwgFloatArray generateSin(float sampleRate,float frequency,float phase);

    ///生成方波波形
    AwgFloatArray generateSquare(float sampleRate,float frequency,float duty);

    ///生成三角波形
    AwgFloatArray generateTriangle(float sampleRate,float frequency,float symmetry);

    ///生成噪声波形
    AwgFloatArray generateNoise(float sampleRate,float bandWidth);

    template<typename T>
    inline const T* min(const T* beg,const T* end)
    {
#ifdef __AVX2__
        using Reg = xsimd::batch<T,xsimd::avx>;
        using Mask = xsimd::batch_bool<T, xsimd::avx>;
        constexpr int chunk = Awg::ArrayAlignment / sizeof(T);
        const T* minElement = beg;

        Mask minMask = 0;
        Reg minVec = xsimd::broadcast<T,xsimd::avx>(*beg);
        Reg dataVec = xsimd::broadcast<T,xsimd::avx>(*beg);

        while (beg + chunk <= end)
        {
            dataVec  = xsimd::load_unaligned<xsimd::avx,T>(beg);//加载数据到寄存器
            minMask = xsimd::lt(dataVec,minVec);//将数据和最小值寄存器做比较,判断是否有更小的值

            //如果有新的最小值产生,则从这一组数据中找到最小值所在的索引
            if(minMask.mask())
            {
                minElement = std::min_element(beg,beg+chunk);
                minVec = xsimd::broadcast<T,xsimd::avx>(*minElement);
            }
            beg += chunk;
        }

        if(beg == end)
            return minElement;
        else
        {
            const T* tmpMin = std::min_element(beg,end);
            return *minElement <= *tmpMin ? minElement : tmpMin;
        }
#else
        return std::min_element(begin,end);
#endif
    }


    template<typename T>
    inline const T* minParallel(const T* beg,const T* end)
    {
        std::size_t dataLen = end - beg + 1;
        std::vector<std::size_t> threadChunks = Awg::splitLengthAligned(dataLen,Awg::MinArrayLength,Awg::ArrayAlignment/sizeof (T));
        std::vector< std::future<T*> > futures;
        futures.reserve(threadChunks.size());

        ThreadPool* pool = Awg::globalThreadPool();
        for(std::size_t i = 0; i < threadChunks.size(); i++)
        {
            futures[i] = pool->run(static_cast<T* (*)(const T*, const T*)>(Awg::min),beg,beg+threadChunks[i]);
            beg += threadChunks[i];
        }
        pool->waitforDone();

        T* ret = futures.at(0).get();
        for(std::size_t i = 1; i < futures.size(); i++)
        {
            T* p = futures.at(i).get();
            ret = (*ret) < (*p) ? ret : p;
        }

        return ret;
    }

    template<typename T>
    inline const T* max(const T* beg,const T* end)
    {
#ifdef __AVX2__
        using Reg = xsimd::batch<T,xsimd::avx>;
        using Mask = xsimd::batch_bool<T, xsimd::avx>;
        constexpr int chunk = Awg::ArrayAlignment / sizeof(T);
        const T* maxElement = beg;

        Mask maxMask = 0;
        Reg maxVec = xsimd::broadcast<T,xsimd::avx>(*beg);
        Reg dataVec = xsimd::broadcast<T,xsimd::avx>(*beg);

        while (beg + chunk <= end)
        {
            dataVec  = xsimd::load_unaligned<xsimd::avx,T>(beg);//加载数据到寄存器
            maxMask = xsimd::gt(dataVec,maxVec);//将数据和最大值寄存器做比较,判断是否有更大的值

            //如果有新的最大值产生,则从这一组数据中找到最大值所在的索引
            if(maxMask.mask())
            {
                maxElement = std::max_element(beg,beg+chunk);
                maxVec = xsimd::broadcast<T,xsimd::avx>(*maxElement);
            }
            beg += chunk;
        }

        if(beg == end)
            return maxElement;
        else
        {
            const T* tmpMax = std::max_element(beg,end);
            return *maxElement >= *tmpMax ? maxElement : tmpMax;
        }
#else
        return std::max_element(begin,end);
#endif
    }


    template<typename T>
    inline const T* maxParallel(const T* beg,const T* end)
    {
        std::size_t dataLen = end - beg + 1;
        std::vector<std::size_t> threadChunks = Awg::splitLengthAligned(dataLen,Awg::MinArrayLength,Awg::ArrayAlignment/sizeof (T));
        std::vector< std::future<T*> > futures;
        futures.reserve(threadChunks.size());

        ThreadPool* pool = Awg::globalThreadPool();
        for(std::size_t i = 0; i < threadChunks.size(); i++)
        {
            futures[i] = pool->run(static_cast<T* (*)(const T*, const T*)>(Awg::max),beg,beg+threadChunks[i]);
            beg += threadChunks[i];
        }
        pool->waitforDone();

        T* ret = futures.at(0).get();
        for(std::size_t i = 1; i < futures.size(); i++)
        {
            T* p = futures.at(i).get();
            ret = (*ret) > (*p) ? ret : p;
        }

        return ret;
    }

    template<typename T>
    inline std::pair<const T*,const T*> minmax(const T* beg,const T* end)
    {
#ifdef __AVX2__
        using Reg = xsimd::batch<T,xsimd::avx>;
        using Mask = xsimd::batch_bool<T, xsimd::avx>;
        constexpr int chunk = Awg::ArrayAlignment / sizeof(T);
        const T* maxElement = beg;
        const T* minElement = beg;

        Mask maxMask = 0;
        Mask minMask = 0;
        Reg maxVec = xsimd::broadcast<T,xsimd::avx>(*beg);
        Reg minVec = xsimd::broadcast<T,xsimd::avx>(*beg);
        Reg dataVec = xsimd::broadcast<T,xsimd::avx>(*beg);
        while (beg + chunk <= end)
        {
            dataVec  = xsimd::load_unaligned<xsimd::avx,T>(beg);//加载数据到寄存器
            maxMask = xsimd::gt(dataVec,maxVec);//将数据和最大值寄存器做比较,判断是否有更大的值
            minMask = xsimd::lt(dataVec,minVec);//将数据和最小值寄存器做比较,判断是否有更小的值

            //如果有新的最小值产生,则从这一组数据中找到最小值所在的索引
            if(minMask.mask())
            {
                minElement = std::min_element(beg,beg+chunk);
                minVec = xsimd::broadcast<T,xsimd::avx>(*minElement);
            }
            //如果有新的最大值产生,则从这一组数据中找到最大值所在的索引
            if(maxMask.mask())
            {
                maxElement = std::max_element(beg,beg+chunk);
                maxVec = xsimd::broadcast<T,xsimd::avx>(*maxElement);
            }
            beg += chunk;
        }

        if(beg == end)
            return std::make_pair(minElement,maxElement);
        else
        {
            std::pair<const T *, const T *> ret = std::minmax_element(beg,end);
            ret.first = (*minElement) <= (*ret.first) ? minElement : ret.first;
            ret.second = (*maxElement) >= (*ret.second) ? maxElement : ret.second;
            return ret;
        }
#else
        return std::minmax_element(begin,end);
#endif
    }

    template<typename T>
    inline std::pair<const T*,const T*> minmaxParallel(const T* beg,const T* end)
    {
        std::size_t dataLen = end - beg + 1;
        std::vector<std::size_t> threadChunks = Awg::splitLengthAligned(dataLen,Awg::MinArrayLength,Awg::ArrayAlignment/sizeof (T));
        std::vector< std::future<std::pair<const T *, const T *>> > futures;
        futures.reserve(threadChunks.size());

        ThreadPool* pool = Awg::globalThreadPool();
        for(std::size_t i = 0; i < threadChunks.size(); i++)
        {
            futures[i] = pool->run(static_cast<std::pair<const T*, const T*> (*)(const T*, const T*)>(Awg::minmax),beg,beg+threadChunks[i]);
            beg += threadChunks[i];
        }
        pool->waitforDone();

        std::pair<const T *, const T *> ret = futures.at(0).get();
        for(std::size_t i = 1; i < futures.size(); i++)
        {
            std::pair<const T *, const T *> p = futures.at(i).get();
            ret.first = (*ret.first) < (*p.first) ? ret.first : p.first;
            ret.second = (*ret.second) > (*p.second) ? ret.second : p.second;
        }

        return ret;
    }

    ///数据点超过最大像素点之后按包络生成数据的略缩图
    template<typename T>
    inline AlignedSharedArray<T,Awg::ArrayAlignment> generateOverview(const T* data,const std::size_t length)
    {
        if(length <= Awg::MaxPlotPoints)
        {
            //数据点数不超过最大限制时直接返回传入的数据
            AlignedSharedArray<T,Awg::ArrayAlignment> buf(length);
            memcpy(buf,data,sizeof (T)*length);
            return buf;
        }
        else
        {
            //当数据点数超过最大限制时将数据点数压缩到最大限制值的两倍,将数据分为Awg::MaxPlotPoints分别处理
            const std::size_t groupLength = length / Awg::MaxPlotPoints;
            const std::size_t remainLength = length - groupLength * Awg::MaxPlotPoints;
            AlignedSharedArray<T,Awg::ArrayAlignment> buf(Awg::MaxPlotPoints*2);

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
            std::vector<std::pair<const T*,const T*>> iteratorGroups;
            iteratorGroups.reserve(Awg::MaxPlotPoints);
            const T* begin = data;
            for(std::size_t i = 0; i < Awg::MaxPlotPoints; i++)
            {
                iteratorGroups.emplace_back(begin,begin+groupVec.at(i));
                begin += groupVec.at(i);
            }

            std::size_t index = 0;
            auto groupBeg = iteratorGroups.begin();
            auto groupEnd = iteratorGroups.end();
            while(groupBeg < groupEnd)
            {
                //开始计算每一组数据,找出每一个分组数据中的最大值和最小值,注意最大值最小值出现顺序,不可交换这两个值的顺序
                std::pair<const T*,const T*> result = Awg::minmax((*groupBeg).first,(*groupBeg).second);

                buf[2*index] = (result.first < result.second) ? (*result.first) : (*result.second);
                buf[2*index+1] = (result.first < result.second) ? (*result.second) : (*result.first);

                ++groupBeg;
                ++index;
            }

            return buf;
        }
    }

    ///数据点超过最大像素点之后按包络生成数据的略缩图(多线程版本)
    template<typename T>
    inline AlignedSharedArray<T,Awg::ArrayAlignment> generateOverviewParallel(const T* data,const std::size_t length)
    {
        if(length <= Awg::MaxPlotPoints)
        {
            //数据点数不超过最大限制时直接返回传入的数据
            AlignedSharedArray<T,Awg::ArrayAlignment> buf(length);
            memcpy(buf,data,sizeof (T)*length);
            return buf;
        }
        else
        {
            //当数据点数超过最大限制时将数据点数压缩到最大限制值的两倍,将数据分为Awg::MaxPlotPoints分别处理
            const std::size_t groupLength = length / Awg::MaxPlotPoints;
            const std::size_t remainLength = length - groupLength * Awg::MaxPlotPoints;
            AlignedSharedArray<T,Awg::ArrayAlignment> buf(Awg::MaxPlotPoints*2);

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
            std::vector<std::pair<const T*,const T*>> iteratorGroups;
            iteratorGroups.reserve(Awg::MaxPlotPoints);
            const T* begin = data;
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
                    std::pair<const T*,const T*> pair = iteratorGroups[i];
                    std::pair<const T*,const T*> result = Awg::minmax(pair.first,pair.second);

                    buf[2*i] = (result.first < result.second) ? (*result.first) : (*result.second);
                    buf[2*i+1] = (result.first < result.second) ? (*result.second) : (*result.first);
                }
            };

            //按分出来的组groupVec划分到线程池中,计算每一个线程处理多少组数据
            std::vector<std::size_t> threadGroups = Awg::splitLengthMin(Awg::MaxPlotPoints,500);
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

    ///数组类型转换的标量版本,当输入输出数组类型From和To所占字节数长度不一致时会使用这个版本
    template<typename From,typename To>
    inline typename std::enable_if<sizeof(From) != sizeof(To)>::type
    arrayCastImpl(To* output,const From* beg,const From* end)
    {
        constexpr To MAX = std::numeric_limits<To>::max();
        constexpr To MIN = std::numeric_limits<To>::min();
        while (beg < end)
        {
            *output = std::min(std::max( To(std::round(*beg)), MIN), MAX);
            ++output;
            ++beg;
        }
    }

    ///数组类型转换的标量版本,当输入输出数组类型From和To所占字节数长度一致时会使用这个版本,这是硬件加速版本,效率更高
    ///如果From和To类型长度不一致又需要硬件加速则需要手动实现
    template<typename From,typename To>
    inline typename std::enable_if<sizeof(From) == sizeof(To)>::type
    arrayCastImpl(To* output,const From* beg,const From* end)
    {
        using BatchFrom = xsimd::batch<From>;
        using BatchTo = xsimd::batch<To>;
        constexpr std::size_t chunk = BatchFrom::size;
        while (beg + chunk <= end)
        {
            BatchFrom fb = xsimd::load_unaligned(beg);
            BatchTo tb = xsimd::batch_cast<To>(fb);
            tb.store_unaligned(output);

            beg += chunk;
            output += chunk;
        }

        arrayCastScalar(output,beg,end);
    }

    inline void arrayCastImpl(short* output,const float* beg,const float* end)
    {
        const int chunk = Awg::ArrayAlignment / sizeof (float);
        const __m256 short_max = _mm256_set1_ps(32767.0);
        const __m256 short_min = _mm256_set1_ps(-32768.0);
        while (beg + chunk*2 <= end)
        {
            // 加载两个 float 向量
            __m256 val1 = _mm256_loadu_ps(beg);              // 前 8 个 float
            __m256 val2 = _mm256_loadu_ps(beg + chunk);  // 后 8 个 float

            // 四舍五入（使用最近舍入）
            __m256 rounded1 = _mm256_round_ps(val1, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m256 rounded2 = _mm256_round_ps(val2, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

            // 饱和处理：限制在 [-32768.0, 32767.0] 范围内
            rounded1 = _mm256_min_ps(_mm256_max_ps(rounded1, short_min), short_max);
            rounded2 = _mm256_min_ps(_mm256_max_ps(rounded2, short_min), short_max);

            // 转换为 int32_t
            __m256i int32_1 = _mm256_cvtps_epi32(rounded1);
            __m256i int32_2 = _mm256_cvtps_epi32(rounded2);

            __m256i packed = _mm256_packs_epi32(int32_1, int32_2);

            packed = _mm256_permute4x64_epi64(packed, 0b11011000);  // 0xD8

            // 存储结果
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(output), packed);

            // 更新指针
            beg += chunk*2;// 处理了 16 个 float
            output += chunk*2; // 产生了 16 个 short
        }

        constexpr short MAX = std::numeric_limits<short>::max();
        constexpr short MIN = std::numeric_limits<short>::min();
        while (beg < end)
        {
            *output = std::min(std::max( short(std::round(*beg)), MIN), MAX);
            ++output;
            ++beg;
        }
    }

    ///数组数据类型转换
    template<typename From,typename To>
    inline AlignedSharedArray<To,Awg::ArrayAlignment> arrayCast(const AlignedSharedArray<From,Awg::ArrayAlignment>& array)
    {
        AlignedSharedArray<To,Awg::ArrayAlignment> output(array.size());
        if(output == nullptr)
            return output;

        std::size_t index = 0;
        std::vector<std::size_t> chunks = Awg::splitLengthAligned(array.size(),Awg::MinArrayLength,Awg::ArrayAlignment/sizeof(To));//这里应该以输出数组大小倍数为准
        ThreadPool* pool = Awg::globalThreadPool();
        for(std::size_t i = 0; i < chunks.size(); i++)
        {
            To* optBeg = output.data() + index;
            const From* beg = array.data() + index;
            const From* end = beg + chunks[i];
            pool->run<ThreadPool::Ordered>(Awg::arrayCastImpl<From,To>,optBeg,beg,end);

            index += chunks[i];
        }
        pool->waitforDone();
        return output;
    }
}

#endif // AWGALGORITHM_H
