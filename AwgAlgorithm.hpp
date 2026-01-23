#ifndef AWGALGORITHM_HPP
#define AWGALGORITHM_HPP

#ifndef AWGARRAY_H
#include "AwgArray.hpp"
#endif

#ifndef XSIMD_HPP
#include "UtilitySubSystem/xsimd/xsimd.hpp"
#endif

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

    template<typename T>
    const T* min(const T* beg,const T* end)
    {
        using Reg = xsimd::batch<T,xsimd::avx>;
    }

    template<typename T>
    const T* minParallel(const T* beg,const T* end)
    {
        using Reg = xsimd::batch<T,xsimd::avx>;
    }

    template<typename T>
    const T* max(const T* beg,const T* end)
    {

    }

    template<typename T>
    const T* maxParallel(const T* beg,const T* end)
    {

    }

    template<typename T>
    std::pair<const T*,const T*> minmax(const T* beg,const T* end)
    {

    }

    template<typename T>
    std::pair<const T*,const T*> minmaxParallel(const T* beg,const T* end)
    {

    }

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

}

#endif // AWGALGORITHM_HPP
