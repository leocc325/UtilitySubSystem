#include "AwgAlgorithm.h"
#include "UtilitySubSystem/AwgUtility.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <random>
#include <QFile>

namespace Awg
{
    std::size_t countCharScalar(const char* beg, const char* end, char target) noexcept
    {
        std::size_t count = 0;
        while (beg < end)
        {
            count += ( (*beg) == target );
            ++beg;
        }
        return count;
    }

#ifdef __AVX2__
    std::size_t countCharAvx2(const char* beg, const char* end, char target) noexcept
    {
        const int step = Awg::ArrayAlignment;
        std::size_t count = 0;
        // 处理不对齐的部分,_mm256_load_si256 需要32字节对齐地址,否则会引发错误。
        while (reinterpret_cast<uintptr_t>(beg) % step != 0 && beg < end)
        {
            count += (*beg == target);
            ++beg;
        }

        __m256i targetChar = _mm256_set1_epi8(target);
        while (beg + step <= end)
        {
            __m256i chunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(beg));
            __m256i cmp = _mm256_cmpeq_epi8(chunk,targetChar);
            int mask = _mm256_movemask_epi8(cmp);

            count += __builtin_popcount(mask);
            beg += step;
        }

        // 处理剩余的不够32字节的部分
        return count + countCharScalar(beg,end,target);
    }
#endif

    const char* findCharScalar(const char *beg, const char *end, char target) noexcept
    {
        while (beg < end)
        {
            if(*beg == target)
                return beg;
            ++beg;
        }
        return nullptr;
    }

#ifdef __AVX2__
    const char* findCharAvx2(const char *beg, const char *end, char target) noexcept
    {
        const std::size_t step = Awg::ArrayAlignment;

        __m256i mask = _mm256_set1_epi8(target);
        while (beg + step <= end)
        {
            __m256i chunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(beg));
            __m256i cmpRet = _mm256_cmpeq_epi8(chunk,mask);
            int maskRet = _mm256_movemask_epi8(cmpRet);

            if(maskRet == 0)
                beg += step;
            else
                return __builtin_ctz(maskRet) + beg ;
        }

        return findCharScalar(beg,end,target);
    }
#endif

    void compressShort12BitScalar(bool bigEndian,const short *begin, const short *end, char *output)
    {
        //每2个short最终占用3字节内存
        unsigned short first = 0,second = 0;
        while (begin + 2 <= end)
        {
            first = begin[0] & 0xFFF;
            second = begin[1] & 0xFFF;
            if(bigEndian)
            {
                output[0] = static_cast<char>(first >> 4);
                output[1] = static_cast<char>((first & 0xF) << 4) |  static_cast<char>(second >> 8);
                output[2] = static_cast<char>(second & 0xFF);
            }
            else
            {
                output[0] = static_cast<char>(((second & 0xF) << 4) | ((first >> 8) & 0xF));
                output[1] = static_cast<char>(first & 0xFF);
                output[2] = static_cast<char>(second >> 4);
            }

            begin += 2;
            output += 3;
        }

        //处理剩下的最后一个数据
        if(begin != end)
        {
            first = (*begin) & 0xFFF;
            if(bigEndian)
            {
                output[0] = static_cast<char>(first >> 4);
                output[1] = static_cast<char>((first & 0xF) << 4);
            }
            else
            {
                output[0] = static_cast<char>(first & 0xFF);        // 低8位
                output[1] = static_cast<char>((first >> 8) & 0xF);  // 高4位在低4位位置
            }
        }
    }

#ifdef __AVX2__
    void compressShort12BitAvx2(bool bigEndian,const short *begin, const short *end, char *output)
    {
        const std::size_t chunkLeng = Awg::ArrayAlignment / sizeof (short);
        __m256i dataVec = _mm256_setzero_si256();
        __m256i orVec = _mm256_setzero_si256();
        const __m256i andMask = _mm256_set1_epi16(0xFFF);
        const  __m256i shiftMask = _mm256_setr_epi16(16,1,16,1,16,1,16,1,16,1,16,1,16,1,16,1);
        const __m256i orMask = _mm256_setr_epi8(0,0,0,0xFF,0,0,0,0xFF,0,0,0,0xFF,0,0,0,0xFF,
                                                0,0,0,0xFF,0,0,0,0xFF,0,0,0,0xFF,0,0,0,0xFF);
        const __m256i orShuffle = _mm256_setr_epi8(3,0,1,2,7,4,5,6,11,8,9,10,15,12,13,14,
                                                   3,0,1,2,7,4,5,6,11,8,9,10,15,12,13,14);
        const __m256i bigEndianMask = _mm256_setr_epi8(1,0,2,5,4,6,9,8,10,13,12,14,3,7,11,15,
                                                       1,0,2,5,4,6,9,8,10,13,12,14,3,7,11,15);//按大端模式重排字节顺序
        const __m256i littleEndianMask = _mm256_setr_epi8(0,1,5,2,6,4,8,9,13,10,14,12,7,3,15,11,
                                                          0,1,5,2,6,4,8,9,13,10,14,12,7,3,15,11);//按小端模式重排字节顺序
        __m256i setLow16bitToZero = _mm256_setr_epi64x(0xFFFFFFFFFFFFFFFF,0xFFFFFFFFFFFFFFFF,0xFFFFFFFFFFFFFFFF,0x0000000000000000);
        __m256i permutexMask = _mm256_setr_epi32(0,1,2,4,5,6,3,7);
        while (begin + chunkLeng <= end)
        {
            dataVec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(begin));
            dataVec = _mm256_and_si256( dataVec,andMask);//所有数据都只取低12bit,高4bit数据置零
            dataVec = _mm256_mullo_epi16(dataVec,shiftMask);//用乘法代替左移运算
            orVec = _mm256_setzero_si256();//将或运算寄存器置零
            orVec = _mm256_blendv_epi8(orVec,dataVec,orMask);//提取指定位上的8bit数据
            orVec = _mm256_shuffle_epi8(orVec,orShuffle);//洗牌之后需要合并的bit位已经在目标位置上了,直接跟源寄存器做或运算
            dataVec = _mm256_or_si256(dataVec,orVec);//求或运算完成数据拼接
            dataVec = bigEndian ? _mm256_shuffle_epi8(dataVec,bigEndianMask) : _mm256_shuffle_epi8(dataVec,littleEndianMask);//对拼接好的数据重新排序,可以自行选择大端排序或者小端排序
            dataVec = _mm256_permutevar8x32_epi32(dataVec,permutexMask);//将拼接好的数据放到寄存器前面,无效数据放到寄存器最后
            dataVec = _mm256_and_si256(dataVec,setLow16bitToZero);//将最后8字节清零,因为最后8字节残留了无效的数据,不清零会污染波形数据
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(output),dataVec);

            //实际上这里写入了12字节的无效数据,所以这里output指针只跳过24
            output += std::size_t(chunkLeng*1.5);//写入到output数组中的数据是24字节
            begin += chunkLeng;//输入数组跳过16个
        }

        //最后处理无法使用AVX指令处理的数据
        compressShort12BitScalar(bigEndian,begin,end,output);
    }
#endif

    void normalizationScalar(const float* inputBegin,const float* inputEnd,float* outputIterator,const float inputMin,const float inputMax,const float outputMin,const float outputMax)
    {
        while (inputBegin < inputEnd)
        {
            *outputIterator = (*inputBegin - inputMin)/(inputMax - inputMin)*(outputMax-outputMin) + outputMin;
            ++inputBegin;
            ++outputIterator;
        };
    }

    ///这个函数在仅在当前文件中被调用,因此可以保证inputBegin地址是按32字节对齐的,所以这里不需要额外处理未对齐的数据
#ifdef __AVX2__
    void normalizationAvx2(const float* inputBegin,const float* inputEnd,float* outputIterator,const float inputMin,const float inputMax,const float outputMin,const float outputMax)
    {
        const std::size_t step = Awg::ArrayAlignment / sizeof (float);
        //输出 = (输入 - 输入范围下限) / 输入范围 * 输出范围 + 输出范围下限
        __m256 inputMinVec = _mm256_set1_ps(inputMin);
        __m256 inputRangeVec = _mm256_set1_ps(inputMax - inputMin);
        __m256 outputRangeVec = _mm256_set1_ps(outputMax - outputMin);
        __m256 outputMinVec = _mm256_set1_ps(outputMin);

        while (inputBegin + step <= inputEnd)
        {
            __m256 ret = _mm256_load_ps(inputBegin);
            ret = _mm256_sub_ps(ret,inputMinVec);
            ret = _mm256_div_ps(ret,inputRangeVec);
            ret = _mm256_mul_ps(ret,outputRangeVec);
            ret = _mm256_add_ps(ret,outputMinVec);

            _mm256_store_ps(outputIterator,ret);
            inputBegin+= step;
            outputIterator += step;
        }

        //处理剩下的数据
        normalizationScalar(inputBegin,inputEnd,outputIterator,inputMin,inputMax,outputMin,outputMax);
    }
#endif

    ///计算正弦波形数据,数组为output,计算出来的数据写到[beg,end)区间
    void outputSinScalar(std::size_t totalPoints,float phaseRad,const float* output,float* beg,float* end)
    {
        while (beg < end)
        {
            float rad = 2.0 * Awg::PI * (beg - output) / totalPoints + phaseRad;
            *beg = Awg::Amplitude * (std::sin(rad)+1)/2;
            ++beg;
        }
    }

#ifdef __AVX2__
    void outputSinAvx2(std::size_t totalPoints,float phaseRad,const float* output,float* beg,float* end)
    {
        constexpr int chunk = Awg::ArrayAlignment / sizeof (float);
        const  __m256 piMul2 = _mm256_set1_ps(2.0*Awg::PI);
        const __m256 phaseRadVec = _mm256_set1_ps(phaseRad);
        const __m256 totalPointsVec = _mm256_set1_ps(totalPoints);
        const __m256 amplVec = _mm256_set1_ps(Awg::Amplitude);
        __m256 retVec = _mm256_set1_ps(0);
        __m256 indexVc = _mm256_set1_ps(0);
        while (beg + chunk <= end)
        {
            float index = beg - output;
            indexVc = _mm256_setr_ps(index,index+1,index+2,index+3,index+4,index+5,index+6,index+7);
            retVec = _mm256_mul_ps(indexVc,piMul2);
            retVec = _mm256_div_ps(retVec,totalPointsVec);
            retVec = _mm256_add_ps(retVec,phaseRadVec);
            //-o0优化下xsimd::sin的效率显著低于AvxSin::sinAvx2HighPrecision,但是在-o2优化下又于AvxSin::sinAvx2HighPrecision
            retVec = xsimd::sin(xsimd::batch<float,xsimd::avx>(retVec));//这里-o0优化并且直接指定axv2反而会导致段错误,很奇怪。
            retVec = _mm256_add_ps(retVec,_mm256_set1_ps(1.0));
            retVec = _mm256_div_ps(retVec,_mm256_set1_ps(2.0));
            retVec = _mm256_mul_ps(retVec,amplVec);
            _mm256_storeu_ps(beg,retVec);

            beg += chunk;
        }
        // 处理剩余的元素(不足4个的情况)
        outputSinScalar(totalPoints,phaseRad,output,beg,end);
    }
#endif

    void outputSquareScalar(const float* edge,float* beg,float* end)
    {
        while (beg < end)
        {
            *beg = (beg < edge) * Awg::Amplitude;//小于占空比索引为高电平,大于为低电平
            ++beg;
        }
    }

#ifdef __AVX2__
    void outputSquareAvx2(const float* edge,float* beg,float* end)
    {
        constexpr int chunk = Awg::ArrayAlignment / sizeof (float);
        const __m256 amplVec = _mm256_set1_ps(Awg::Amplitude);
        const __m256 zeroVec = _mm256_setzero_ps();

        while (beg + chunk <= end)
        {
            float* thunkEnd = beg + chunk;
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
                    _mm256_storeu_ps(beg, amplVec);
                else// 整个chunk都在edgeIndex右侧
                    _mm256_storeu_ps(beg, zeroVec);
                beg += chunk;
            }
        }

        // 处理剩余元素
        outputSquareScalar(edge,beg,end);
    }
#endif

    void outputTriangleScalar(float raiseK,float raiseB,float fallK,float fallB,const float* output,const float* peak,float* beg,float* end)
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

#ifdef __AVX2__
    void outputTriangleAvx2(float raiseK,float raiseB,float fallK,float fallB,const float* output,const float* peak,float* beg,float* end)
    {
        constexpr int chunk = Awg::ArrayAlignment / sizeof (float);
        const __m256 raiseBvec = _mm256_set1_ps(raiseB);
        const __m256 fallBvec = _mm256_set1_ps(fallB);
        const __m256 raiseKvec = _mm256_set1_ps(raiseK);
        const __m256 fallKvec = _mm256_set1_ps(fallK);
        __m256 indexVec = _mm256_setzero_ps();
        __m256 dataVec = _mm256_setzero_ps();
        while (beg + chunk <= end)
        {
            float* thunkEnd = beg + chunk;
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
                indexVec = _mm256_setr_ps(index,index+1,index+2,index+3,index+4,index+5,index+6,index+7);//索引反向加载
                if(thunkEnd <= peak)
                {
                    dataVec = _mm256_mul_ps(raiseKvec,indexVec);
                    dataVec = _mm256_add_ps(dataVec,raiseBvec);
                }
                else
                {
                    dataVec = _mm256_mul_ps(fallKvec,indexVec);
                    dataVec = _mm256_add_ps(dataVec,fallBvec);
                }
                _mm256_storeu_ps(beg,dataVec);

                beg += chunk;
            }
        }
        //处理剩余元素
        outputTriangleScalar(raiseK,raiseB,fallK,fallB,output,peak,beg,end);
    }
#endif
}
#include <QDebug>
    void Awg::test()
    {
        short original[20] = {0x123,0x456,0x789,0xabc,0xdef
            ,0x123,0x456,0x789,0xabc,0xdef
        ,0x123,0x456,0x789,0xabc,0xdef
        ,0x123,0x456,0x789,0xabc,0xdef};

        char opt1[30];
        char opt2[30];
        char opt3[30];
        char opt4[30];
        Awg::compressShort12BitAvx2(true,original,original+20,opt1);
        Awg::compressShort12BitScalar(true,original,original+20,opt2);
        Awg::compressShort12BitAvx2(false,original,original+20,opt3);
        Awg::compressShort12BitScalar(false,original,original+20,opt4);
        for(int i = 0; i < 30; i++)
        {
            qDebug()<<opt1[i]<<opt2[i]<<opt3[i]<<opt4[i];
        }
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

std::size_t Awg::countChar( const char *beg, const char *end, char target) noexcept
{
#ifdef __AVX2__
    return Awg::countCharAvx2(beg,end,target);
#else
    return Awg::countCharScalar(beg,end,target);
#endif
}

std::size_t Awg::countCharParallel(const char *beg, const char *end, char target) noexcept
{
    const std::size_t length = end - beg;
    std::vector<std::size_t> chunks = Awg::splitLengthMin(length,Awg::MinArrayLength);
    std::vector< std::future<std::size_t> > futures;
    futures.reserve(chunks.size());

    ThreadPool* pool = Awg::globalThreadPool();
    for(std::size_t i = 0; i < chunks.size(); i++)
    {
        futures.push_back(pool->run(Awg::countChar,beg,beg+chunks[i],target));
        beg += chunks[i];
    }
    pool->waitforDone();

    std::size_t count = 0;
    for(std::size_t i = 0; i < futures.size(); i++)
    {
        count += futures[i].get();
    }
    return count;
}

const char* Awg::findChar(const char *beg, const char *end, char target) noexcept
{
#ifdef __AVX2__
    return Awg::findCharAvx2(beg,end,target);
#else
    return Awg::findCharScalar(beg,end,target);
#endif
}

const char* Awg::findCharParallel(const char *beg, const char *end, char target) noexcept
{
    const std::size_t length = end - beg;
    std::vector<std::size_t> chunks = Awg::splitLengthMin(length,Awg::MinArrayLength);
    std::vector< std::future<const char*> > futures;
    futures.reserve(chunks.size());

    ThreadPool* pool = Awg::globalThreadPool();
    for(std::size_t i = 0; i < chunks.size(); i++)
    {
        futures.push_back(pool->run(Awg::findChar,beg,beg+chunks[i],target));
        beg += chunks[i];
    }
    pool->waitforDone();

    //返回最小的指针
    std::vector<const char*> rets;
    for(std::size_t i = 0; i < futures.size(); i++)
    {
        const char* f = futures[i].get();
        if( f != nullptr)
            rets.push_back(f);
    }

    if(rets.empty())
        return nullptr;
    else
        return *std::min_element(rets.begin(),rets.end());
}

void Awg::reverse(char *beg, char *end)
{
    //反向原理:先提取数据范围开头的N字节数据和末尾N字节数据,然后对提取的两块数据分别做反向处理,随后再交换他们在原数组中的位置
    //例如原始数组为:[1,2,3,4,5,6,7,8,9],每次交换两字节数据:
    //提取开头末尾两字节数据:[1,2][3,4,5,6,7][8,9]
    //交换提取的两字节数据位置:[2,1][3,4,5,6,7][9,8]
    //交换他们在原始数组中的位置:[9,8][3,4,5,6,7][2,1]
    //再提取范围内的开头末尾两字节数据:[9,8][3,4][5][6,7][2,1]
    //交换这两组数据的顺序:[9,8][4,3][5][7,6][2,1]
    //交换这两组数据在数组中的位置:[9,8][7,6][5][4,3][2,1]
    //整个数组处理完毕:[9,8,7,6,5,4,3,2,1]
    int stepLength = 32;
    while (beg + stepLength*2 <= end)
    {
        __m256i mask = _mm256_set_epi8(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
                                       0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
        //分别取出数组开头和末尾的32字节数据
        __m256i headChunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(beg));
        __m256i tailChunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(end - stepLength));
        //然后将这两块数据反向
        headChunk = _mm256_shuffle_epi8(headChunk,mask);
        headChunk = _mm256_permute2x128_si256(headChunk, headChunk, 0x01);
        tailChunk = _mm256_shuffle_epi8(tailChunk,mask);
        tailChunk = _mm256_permute2x128_si256(tailChunk, tailChunk, 0x01);
        //反向完成之后写入原始数组:交换头尾32字节数据,开头的数据写入到末尾,末尾的数据写入到开头
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(beg),tailChunk);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(end-stepLength),headChunk);
        //缩小数据范围
        beg += stepLength;
        end -= stepLength;
    }
    std::reverse(beg,end);
}

void Awg::compressShort12Bit(const short *begin, const short *end, char *output)
{
#ifdef __AVX2__
    Awg::compressShort12BitAvx2(true,begin,end,output);
#else
    Awg::compressShort12BitScalar(true,begin,end,output);
#endif
}

AwgFloatArray Awg::normalization(const AwgFloatArray &input, const float inputMin, const float inputMax, const float outputMin, const float outputMax)
{
    if(input.empty())
        return AwgFloatArray{};

    AwgFloatArray output(input.size());
    const float* inputBeg = input.data();
    float* outputBeg = output.data();
#ifdef __AVX2__
    Awg::normalizationAvx2(inputBeg,inputBeg+input.size(),outputBeg,inputMin,inputMax,outputMin,outputMax);
#else
    Awg::normalizationScalar(inputBeg,inputBeg+input.size(),outputBeg,inputMin,inputMax,outputMin,outputMax);
#endif
    return output;
}

AwgFloatArray Awg::normalizationParallel(const AwgFloatArray &input, const float inputMin, const float inputMax, const float outputMin, const float outputMax)
{
    if(input.empty())
        return AwgFloatArray{};

    AwgFloatArray output(input.size());
    std::vector<std::size_t> threadChunks = Awg::splitLengthAligned(input.size(),Awg::MinArrayLength,Awg::ArrayAlignment/sizeof (float));

    ThreadPool* pool = Awg::globalThreadPool();
    float* inputBeg = input.data();
    float* outputBeg = output.data();
    for(std::size_t i = 0; i < threadChunks.size(); i++)
    {
#ifdef __AVX2__
        pool->run(Awg::normalizationAvx2,inputBeg,inputBeg+threadChunks[i],outputBeg,inputMin,inputMax,outputMin,outputMax);
#else
        pool->run(Awg::normalizationScalar,inputBeg,inputBeg+threadChunks[i],outputBeg,inputMin,inputMax,outputMin,outputMax);
#endif
        inputBeg += threadChunks[i];
        outputBeg += threadChunks[i];
    }
    pool->waitforDone();

    return output;
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
                unsigned char* data = file.map(index+chunkSize,preReadLeng);
                const char* beg = reinterpret_cast<const char*>(data);
                const char* end = beg + preReadLeng;
                const char* extra = end;

                for(std::size_t i = 0; i < spliters.size(); i++)
                {
                    const char* pos = Awg::findChar(beg,end,spliters[i]);
                    if(pos != nullptr)
                        extra = std::min(pos,extra);
                }

                std::size_t extraLeng = (extra == end) ? 0 : extra - beg;
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

std::vector<std::size_t> Awg::splitLengthMin(std::size_t length, std::size_t minChunk) noexcept
{
    if(length == 0 || minChunk == 0)
        return std::vector<std::size_t>{};

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

std::vector<std::size_t> Awg::splitLengthMax(std::size_t length, std::size_t maxChunk) noexcept
{
    if(length == 0 || maxChunk == 0)
        return std::vector<std::size_t>{};

    std::vector<std::size_t> vec;
    std::size_t vecSize = std::ceil( double(length) / maxChunk);
    vec.reserve(vecSize);

    std::size_t remaining = length;

    for(std::size_t i = 0; i < vecSize; i++)
    {
        std::size_t currentChunk = std::min(remaining, maxChunk);
        vec.push_back(currentChunk);
        remaining -= currentChunk;
    }

    return vec;
}

std::vector<std::size_t> Awg::splitLengthAligned(std::size_t length,std::size_t minChunk,std::size_t aligned)  noexcept
{
    if(length == 0 || minChunk == 0)
        return std::vector<std::size_t>{};

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

AwgFloatArray Awg::generateSin(float sampleRate, float frequency, float phase)
{
    // 计算一个完整周期内的采样点数
    const std::size_t totalPoints = std::round(sampleRate / frequency);
    const float phaseRad = phase * Awg::PI / 180.0;

    if (totalPoints == 0)
    {
        std::cerr << "Error: sampleRate must be greater than frequency" << std::endl;
        return AwgFloatArray{};
    }

    // 分配内存存储波形数据
    AwgFloatArray waveform(totalPoints);

    if (waveform == nullptr)
    {
        std::cerr << "Error: Memory allocation failed" << std::endl;
        return AwgFloatArray{};
    }

    //每一个线程处理的点数为总点数/线程数再向上取整
    std::size_t index = 0;
    std::vector<std::size_t> chunks = Awg::splitLengthAligned(totalPoints,Awg::MinArrayLength,Awg::ArrayAlignment/sizeof(double));
    ThreadPool* pool = Awg::globalThreadPool();
    for(std::size_t i = 0; i < chunks.size(); i++)
    {
        float* output = waveform.data();
        float* beg = output + index;
        float* end = beg + chunks[i];
#if __AVX2__
        pool->run<ThreadPool::Ordered>(Awg::outputSinAvx2,totalPoints,phaseRad,output,beg,end);
#else
        pool->run<ThreadPool::Ordered>(Awg::outputSinScalar,totalPoints,phaseRad,output,beg,end);
#endif
        index += chunks[i];
    }

    pool->waitforDone();
    return waveform;
}

AwgFloatArray Awg::generateSquare(float sampleRate, float frequency, float duty)
{
    //每个周期最少100个点,这样可以将占空比的精度控制到1%
    const unsigned minPointsPerPeriod = 100;
    const std::size_t totalPoints = std::round(sampleRate / frequency);
    if(totalPoints < minPointsPerPeriod)
        return AwgFloatArray{};

    // 分配内存存储波形数据
    AwgFloatArray waveform(totalPoints);

    //将占空比限制到0和1之间
    duty = std::max(duty,0.0f);
    duty = std::min(duty,1.0f);
    std::size_t edgeIndex = std::round(totalPoints * duty);
    float* output = waveform.data();
    float* edge = output + edgeIndex;

    //每一个线程处理的点数为总点数/线程数再向上取整
    std::size_t index = 0;
    std::vector<std::size_t> chunks = Awg::splitLengthAligned(totalPoints,Awg::MinArrayLength,Awg::ArrayAlignment/sizeof(double));
    ThreadPool* pool = Awg::globalThreadPool();
    for(std::size_t i = 0; i < chunks.size(); i++)
    {
        float* beg = output + index;
        float* end = beg + chunks[i];
#ifdef __AVX2__
        pool->run<ThreadPool::Ordered>(Awg::outputSquareAvx2,edge,beg,end);
#else
        pool->run<ThreadPool::Ordered>(Awg::outputSquareScalar,edge,beg,end);
#endif
        index += chunks[i];
    }

    pool->waitforDone();
    return waveform;
}

AwgFloatArray Awg::generateTriangle(float sampleRate, float frequency, float symmetry)
{
    //每个周期最少100个点,这样可以将对称性的精度控制到1%
    const unsigned minPointsPerPeriod = 100;
    const std::size_t totalPoints = std::round(sampleRate / frequency);
    if(totalPoints < minPointsPerPeriod)
        return AwgFloatArray{};

    // 分配内存存储波形数据
    AwgFloatArray waveform(totalPoints);

    //将占空比限制到0和1之间
    symmetry = std::max(symmetry,0.0f);
    symmetry = std::min(symmetry,1.0f);

    struct Points
    {
        float x = 0;
        float y = 0;
    };
    std::size_t peakIndex = std::round(totalPoints * symmetry);

    //这里需要处理对称性为0或者1的情况
    float raiseK = 1 ,raiseB = 0,fallK = 1,fallB = 0;

    if(symmetry == 0)
    {
        fallK = float(0 - Awg::Amplitude) / (totalPoints - 0);
        fallB = Awg::Amplitude;
    }
    else if (symmetry == 1)
    {
        raiseK = float(Awg::Amplitude - 0) / (totalPoints - 0);
        raiseB = 0;
    }
    else
    {
        raiseK =float(Awg::Amplitude - 0)/ (peakIndex - 0);
        raiseB = Awg::Amplitude - raiseK * peakIndex;
        fallK = float(0 - Awg::Amplitude) / (totalPoints - peakIndex);
        fallB =  Awg::Amplitude - fallK*peakIndex;
    }
    float* output = waveform.data();
    float* peak = output + peakIndex;

    //每一个线程处理的点数为总点数/线程数再向上取整
    std::size_t index = 0;
    std::vector<std::size_t> chunks = Awg::splitLengthAligned(totalPoints,Awg::MinArrayLength,Awg::ArrayAlignment/sizeof(double));
    ThreadPool* pool = Awg::globalThreadPool();
    for(std::size_t i = 0; i < chunks.size(); i++)
    {
        float* beg = output + index;
        float* end = beg + chunks[i];
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

AwgFloatArray Awg::generateNoise(float sampleRate, float bandWidth)
{
#if 0
    //mingw的编译器实现似乎有bug,std::random_device生成的种子一直是同一个值,所以这里采用别的方式获取随机种子
    std::random_device rd;
    std::size_t seed = rd();
#else
    std::size_t time = std::chrono::system_clock::now().time_since_epoch().count();
    std::size_t seed = time % std::numeric_limits<uint>::max();
#endif
    std::mt19937_64 engine(seed);
    std::uniform_int_distribution<short> dis(0,4096);
    for(int i = 0; i < 100; i++)
    {
        int a = dis(engine);
    }
}
