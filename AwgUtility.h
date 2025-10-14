#ifndef AWGUTILITY_H
#define AWGUTILITY_H

#ifndef AWGDEFINES_H
#include "AwgDefines.h"
#endif

#ifndef _GLIBCXX_VECTOR
#include <vector>
#endif

class QFile;
class ThreadPool;
namespace Awg
{
    ///一个全局线程池,全局线程池主要用于并行计算和读写数据 !!!不要给这个线程池传入计算和读写以外的函数,否则可能会导致计算结果获取被延后!!!
    ThreadPool* globalThreadPool() noexcept;

    ///查询系统当前可用内存
    std::size_t getFreeMemoryWindows();

    ///判断字符c是否是浮点数的开头
    bool isFloatBegin(char c) noexcept;

    ///判断字符c是否是整数的开头
    bool isIntegerBegin(char c) noexcept;

    ///查找字符串中第一次出现字符target的位置
    std::size_t findChar(const char* data,std::size_t leng,char target) noexcept;

    ///使用SIMD指令查找字符串中第一次出现字符target的位置
    std::size_t findCharAvx2(const char* data,std::size_t leng,char target) noexcept;

    ///计算字符串中出现字符target的次数
    std::size_t countChar(const char* data, std::size_t leng, char target) noexcept;

     ///使用AVX2指令集计算字符串中出现字符target的次数
    std::size_t countCharAvx2(const char* data,std::size_t leng,char target) noexcept;

    ///根据线程池线程数将文本文件切割成若干个大小至少为minCunk的小内存块,同时保证里面的数据不被切割到不同的块中,返回每一个块的大小
    std::vector<std::size_t> cutTextFile(QFile& file, std::size_t minChunk, const std::vector<char>& spliters);

    ///根据线程池线程数将二进制文件切割成若干个大小至少为minCunk的小内存块,同时保证里面的数据不被切割到不同的块中,不同的块的大小为dataBytes的整倍数,返回每一个块的大小
    std::vector<std::size_t> cutBinaryFile(const std::size_t fileSize,const std::size_t minChunk,const unsigned dataBytes) noexcept;

    ///根据线程池线程数将一个长度为length块切割成若干个大小至少为minChunk的小数组,返回这些数组的长度
    std::vector<std::size_t> cutArray(std::size_t length,std::size_t minChunk) noexcept;

    ///根据线程池线程数将一个长度为length块切割成若干个大小至少为minChun而且为aligned整倍数的小数组,返回这些数组的长度
    std::vector<std::size_t> cutArrayAligned(std::size_t length,std::size_t minChunk,std::size_t aligned) noexcept;

    ///将给定数组array中[start,end)索引上的值转换为char*数据写入到frame数组中,数组中每个数据占用bitSize位
    bool converteToBitFrame(const unsigned bitSize,char* frame,const Awg::DT* array,unsigned start,unsigned end);  //1024bit,前768bit存放，后256比特不适用

    void convertIntTo12BitCharArray(const int* intArray, char* charArray, size_t numInts);
}

#endif // AWGUTILITY_H
