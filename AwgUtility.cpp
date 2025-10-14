#include "AwgUtility.h"
#include <immintrin.h>
#include <string>
#include <QFile>
#include <iostream>
#include <windows.h>
#include "AwgSignals.h"
#include "ThreadPool.hpp"
#include "fastfloat/fast_float.h"

ThreadPool *Awg::globalThreadPool() noexcept
{
    static ThreadPool pool(PoolSize);
    return &pool;
}

std::size_t Awg::getFreeMemoryWindows()
{
    MEMORYSTATUSEX memoryStatus;
    memoryStatus.dwLength = sizeof(memoryStatus);

    if (GlobalMemoryStatusEx(&memoryStatus))
    {
        return static_cast<std::size_t>(memoryStatus.ullAvailPhys);
    }

    return -1; // 错误
}

bool Awg::isFloatBegin(char c) noexcept
{
    // 数字、小数点、正负号都可能是浮点数的开头
    return ( (c >= '0' && c <= '9') || c == '-' || c == '.' );
}

bool Awg::isIntegerBegin(char c) noexcept
{
        // 数字、正负号都可能是浮点数的开头
        return ( (c >= '0' && c <= '9') || c == '-'  );
}

std::size_t Awg::findChar(const char *data, std::size_t leng, char target) noexcept
{
    for(std::size_t  i = 0; i < leng; ++i)
    {
        if(data[i] == target)
            return i;
    }
    return 0;
}

std::size_t Awg::findCharAvx2(const char *data, std::size_t leng, char target) noexcept
{
    std::size_t start = 0;
    std::size_t step = 32;
    std::size_t index = 0;

    __m256i mask = _mm256_set1_epi8(target);
    while (start + step <= leng)
    {
        __m256i chunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data+start));
        __m256i cmpRet = _mm256_cmpeq_epi8(chunk,mask);
        int maskRet = _mm256_movemask_epi8(cmpRet);

        if(maskRet == 0)
        {
            start += step;
            index  += 32;
        }
        else
            return __builtin_ctz(maskRet) + index ;
    }

    index += findChar(data+start,leng - start,target);
    return index;
}

std::size_t Awg::countChar(const char *data, std::size_t leng , char target) noexcept
{
    std::size_t count = 0;
    for(std::size_t  i = 0; i < leng; ++i)
    {
        if(data[i] == target)
            ++count;
    }
    return count;
}

std::size_t Awg::countCharAvx2(const char *data, std::size_t leng, char target) noexcept
{
    std::size_t count = 0;
    const char* start = data;
    const char* end   = data + leng;

    // 处理不对齐的部分,_mm256_load_si256 需要32字节对齐地址,否则会引发错误。
    while (reinterpret_cast<uintptr_t>(start) % 32 != 0 && start < end)
    {
        count += (*start == target);
        ++start;
    }
    __m256i targetChar = _mm256_set1_epi8(target);

    while (start + 32 <= end)
    {
        __m256i chunk = _mm256_load_si256(reinterpret_cast<const __m256i*>(start));
        __m256i cmp = _mm256_cmpeq_epi8(chunk,targetChar);
        int mask = _mm256_movemask_epi8(cmp);

        count += __builtin_popcount(mask);
        start += 32;
    }

    // 处理剩余的不够32字节的部分
    while (start < end)
    {
        count += (*start == target);
        ++start;
    }

    return count;
}

std::vector<std::size_t> Awg::cutTextFile(QFile &file, std::size_t minChunk, const std::vector<char> &spliters)
{
    std::vector<std::size_t> vec;
    if(file.isOpen())
    {
        //判断目标size均摊到各个线程中的大小和minChunk谁更大
        std::size_t fileSize = file.size();
        std::size_t chunkFile = fileSize / Awg::PoolSize;//文件均摊之后每一个线程需要处理的Size
        std::size_t chunkThread = Awg::getFreeMemoryWindows()*0.9 / Awg::PoolSize / 2;//剩余内存能支持每一个线程处理文件的Size,这是为了避免多个线程同时加载文件导致内存耗尽,文件和数据加载之后需要占用两份内存,所以除以2
        std::size_t chunkTemp = std::min(chunkFile,chunkThread);//初步计算得到的每个线程能处理的Size,文件均摊之后的Size大于线程支持的Size就以线程支持的Size为准,否则以文件均摊的Size为准
        std::size_t chunkSize = std::max(minChunk,chunkTemp);//初步计算的到的Size如果小于最低的Size要求就以最低Size要求为准,这是为了避免体积太小的文件被切割成很多块并发处理反而会更慢

        //按上方计算出来的chunkSize和分隔符分割文本文件
        const std::size_t preReadLeng  = 64;//预读取长度
        std::size_t index = 0;//映射起始地址

        while (true)
        {
            if(fileSize - index < chunkSize * 1.5)
            {
                //允许最后一块的大小大于给定的chunkSize
                vec.push_back(fileSize - index);
                break;
            }
            else
            {
                std::size_t extraLeng = preReadLeng+1;//还需要读取额外extraLeng才能保证文本没有被切割到两个块中
                unsigned char* data = file.map(index+chunkSize,preReadLeng);

                //分隔符索引中非0的最小值就是需要实际额外读取的长度
                for(std::size_t i = 0; i < spliters.size(); i++)
                {
                    std::size_t pos = Awg::findChar(reinterpret_cast<const char*>(data),preReadLeng,spliters[i]);
                    extraLeng = (pos != 0) ? std::min(extraLeng,pos) : extraLeng;
                }

                //如果最终的额外长度还是等于预读长度+1就说明在第一个字符处匹配成功或者全部字符都没有匹配成功
                //这种情况下就不预读额外的字节数
                extraLeng = (extraLeng == preReadLeng + 1) ? 0 : extraLeng;
                vec.push_back(chunkSize+extraLeng);
                index = index + chunkSize + extraLeng;

                file.unmap(data);
            }
        }
    }
    return vec;
}

std::vector<std::size_t> Awg::cutBinaryFile(const std::size_t fileSize, const std::size_t minChunk, const unsigned dataBytes) noexcept
{
    std::size_t index = 0;//映射起始地址
    std::size_t chunkFile = fileSize / Awg::PoolSize;//文件均摊之后每一个线程需要处理的Size
    std::size_t chunkThread = Awg::getFreeMemoryWindows()*0.9 / Awg::PoolSize / 2;//剩余内存能支持每一个线程处理文件的Size,这是为了避免多个线程同时加载文件导致内存耗尽,文件和数据加载之后需要占用两份内存,所以除以2
    std::size_t chunkTemp = std::min(chunkFile,chunkThread);//初步计算得到的每个线程能处理的Size,文件均摊之后的Size大于线程支持的Size就以线程支持的Size为准,否则以文件均摊的Size为准
    std::size_t chunkSize = std::max(minChunk,chunkTemp);//初步计算的到的Size如果小于最低的Size要求就以最低Size要求为准,这是为了避免体积太小的文件被切割成很多块并发处理反而会更慢

    std::vector<std::size_t> vec;
    while (true)
    {
        if(fileSize - index < chunkSize * 1.5)
        {
            //允许最后一块的大小大于给定的chunkSize
            vec.push_back(fileSize - index);
            break;
        }
        else
        {
            std::size_t length = chunkSize + dataBytes -  chunkSize % dataBytes;
            vec.push_back(length);
            index += length;
        }
    }
    return vec;
}

std::vector<std::size_t> Awg::cutArray(std::size_t length, std::size_t minChunk) noexcept
{
    //判断目标size均摊到各个线程中的大小和minChunk谁更大
    std::size_t averageSize = length / Awg::PoolSize;
    std::size_t chunkSize = std::max(minChunk , averageSize);

    //以上方计算出来的块大小分割
    std::vector<std::size_t> vec;
    std::size_t vecSize = std::ceil( double(length) / chunkSize);
    vec.reserve(vecSize);

    std::size_t index = 0;
    while (index <= length)
    {
        if(length - index < chunkSize * 1.5)
        {
            //允许最后一块的大小大于给定的最小尺寸,避免最后一块剩下很少字节数
            vec.push_back(length - index);
            break;
        }
        else
        {
            vec.push_back(chunkSize);
            index += chunkSize;
        }
    }
    return vec;
}

std::vector<std::size_t> Awg::cutArrayAligned(std::size_t length,std::size_t minChunk,std::size_t aligned)  noexcept
{
    //判断目标size均摊到各个线程中的大小和minChunk谁更大
    std::size_t averageSize = length / Awg::PoolSize;
    std::size_t chunkSize = std::max(minChunk , averageSize);
    chunkSize = chunkSize + aligned - chunkSize % aligned;

    //以上方计算出来的块大小分割
    std::vector<std::size_t> vec;
    std::size_t vecSize = std::ceil( double(length) / chunkSize);
    vec.reserve(vecSize);

    std::size_t index = 0;
    while (index <= length)
    {
        if(length - index < chunkSize * 1.5)
        {
            //允许最后一块的大小大于给定的最小尺寸,避免最后一块剩下很少字节数
            vec.push_back(length - index);
            break;
        }
        else
        {
            vec.push_back(chunkSize);
            index += chunkSize;
        }
    }
    return vec;
}
