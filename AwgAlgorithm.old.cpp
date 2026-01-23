#include "AwgAlgorithm.old.h"
#include "UtilitySubSystem/AwgUtility.h"
#include "UtilitySubSystem/ThreadPool.hpp"
#include "AwgDefines.h"
#include <QFile>
#include <immintrin.h>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <random>
#include "UtilitySubSystem/xsimd/xsimd.hpp"

//Avx版本的sin计算函数,非原创,缺点是我看不懂
namespace AvxSin {
    static const double FABE13_TINY_THRESHOLD = 7.450580596923828125e-9;

    static const double FABE13_TWO_OVER_PI_HI = 0x1.45f306dc9c883p-1;
    static const double FABE13_TWO_OVER_PI_LO = -0x1.9f3c6a7a0b5edp-57;
    static const double FABE13_PI_OVER_2_HI = 0x1.921fb54442d18p+0;
    static const double FABE13_PI_OVER_2_LO = 0x1.1a62633145c07p-53;
    static const double FABE13_SIN_COEFFS_MAIN[] = {
        9.99999999999999999983e-01, -1.66666666666666657415e-01,
        8.33333333333329961475e-03, -1.98412698412589187999e-04,
        2.75573192235635111290e-06, -2.50521083760783692702e-08,
        1.60590438125280493886e-10, -7.64757314471113976640e-13 };
    static const double FABE13_COS_COEFFS_MAIN[] = {
        1.00000000000000000000e+00, -4.99999999999999944489e-01,
        4.16666666666664590036e-02, -1.38888888888829782464e-03,
        2.48015873015087640936e-05, -2.75573192094882420430e-07,
        2.08767569813591324530e-09, -1.14757362211242971740e-11 };

    __inline __m256d __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    polyAvx2(__m256d r_squared, const double* coeffs) {
        const __m256d C0 = _mm256_set1_pd(coeffs[0]), C1 = _mm256_set1_pd(coeffs[1]), C2 = _mm256_set1_pd(coeffs[2]), C3 = _mm256_set1_pd(coeffs[3]), C4 = _mm256_set1_pd(coeffs[4]), C5 = _mm256_set1_pd(coeffs[5]), C6 = _mm256_set1_pd(coeffs[6]), C7 = _mm256_set1_pd(coeffs[7]);
        __m256d z = r_squared; __m256d z2 = _mm256_mul_pd(z, z); __m256d z4 = _mm256_mul_pd(z2, z2);
        __m256d T01 = _mm256_fmadd_pd(C1, z, C0); __m256d T23 = _mm256_fmadd_pd(C3, z, C2); __m256d T45 = _mm256_fmadd_pd(C5, z, C4); __m256d T67 = _mm256_fmadd_pd(C7, z, C6);
        __m256d S03 = _mm256_fmadd_pd(T23, z2, T01); __m256d S47 = _mm256_fmadd_pd(T67, z2, T45);
        return _mm256_fmadd_pd(S47, z4, S03);
    }

    //这个函数来自于https://github.com/farukalpay/FABE,这个函数的精度几乎和std::sin精度一样,但是耗时会比sinAvx2长接近一倍
    __inline __m256d __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    sinAvx2HighPrecision(__m256d vx)
    {
        static const __m256d VEC_TWO_OVER_PI_HI = _mm256_set1_pd(FABE13_TWO_OVER_PI_HI);
        static const __m256d VEC_TWO_OVER_PI_LO = _mm256_set1_pd(FABE13_TWO_OVER_PI_LO);
        static const __m256d VEC_PI_OVER_2_HI = _mm256_set1_pd(FABE13_PI_OVER_2_HI);
        static const __m256d VEC_PI_OVER_2_LO = _mm256_set1_pd(FABE13_PI_OVER_2_LO);
        static const __m256d VEC_NAN = _mm256_set1_pd(NAN);
        static const __m256d VEC_TINY = _mm256_set1_pd(FABE13_TINY_THRESHOLD);
        static const __m256d VEC_SIGN_MASK = _mm256_set1_pd(-0.0); const __m256d VEC_INF = _mm256_set1_pd(INFINITY);
        static const __m256d VEC_ROUND_BIAS = _mm256_set1_pd(6755399441055744.0);
        static const __m256i VEC_ROUND_BIAS_I = _mm256_castpd_si256(VEC_ROUND_BIAS);
        static const __m256i VEC_INT_3 = _mm256_set1_epi64x(3), VEC_INT_0 = _mm256_setzero_si256(), VEC_INT_1 = _mm256_set1_epi64x(1), VEC_INT_2 = _mm256_set1_epi64x(2);

        //_mm_prefetch((const char*)(&in[i + 64]), _MM_HINT_T0);

        __m256d vax = _mm256_andnot_pd(VEC_SIGN_MASK, vx);
        __m256d nan_mask = _mm256_cmp_pd(vx, vx, _CMP_UNORD_Q);
        __m256d inf_mask = _mm256_cmp_pd(vax, VEC_INF, _CMP_EQ_OQ);
        __m256d special_mask = _mm256_or_pd(nan_mask, inf_mask);
        __m256d tiny_mask = _mm256_cmp_pd(vax, VEC_TINY, _CMP_LE_OS);
        __m256d p_hi = _mm256_mul_pd(vx, VEC_TWO_OVER_PI_HI);
        __m256d e1 = _mm256_fmsub_pd(vx, VEC_TWO_OVER_PI_HI, p_hi);
        __m256d p_lo = _mm256_fmadd_pd(vx, VEC_TWO_OVER_PI_LO, e1);
        __m256d k_dd = _mm256_round_pd(_mm256_add_pd(p_hi, p_lo), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256d t1 = _mm256_mul_pd(k_dd, VEC_PI_OVER_2_HI);
        __m256d e2 = _mm256_fmsub_pd(k_dd, VEC_PI_OVER_2_HI, t1);
        __m256d t2 = _mm256_mul_pd(k_dd, VEC_PI_OVER_2_LO);
        __m256d w = _mm256_add_pd(e2, t2);
        __m256d r = _mm256_sub_pd(_mm256_sub_pd(vx, t1), w);
        __m256d r2 = _mm256_mul_pd(r, r);

        // *** TODO: Replace with AVX2 Ψ-Hyperbasis calculation ***
        __m256d sin_poly_r2 = polyAvx2(r2, FABE13_SIN_COEFFS_MAIN);
        __m256d cos_poly_r2 = polyAvx2(r2, FABE13_COS_COEFFS_MAIN);
        __m256d sin_r = _mm256_mul_pd(r, sin_poly_r2);
        __m256d cos_r = cos_poly_r2;
        // *** END TODO ***

        __m256d k_plus_bias = _mm256_add_pd(k_dd, VEC_ROUND_BIAS);
        __m256i vk_int = _mm256_sub_epi64(_mm256_castpd_si256(k_plus_bias), VEC_ROUND_BIAS_I);
        __m256i vq = _mm256_and_si256(vk_int, VEC_INT_3);
        __m256d q0_mask = _mm256_castsi256_pd(_mm256_cmpeq_epi64(vq, VEC_INT_0));
        __m256d q1_mask = _mm256_castsi256_pd(_mm256_cmpeq_epi64(vq, VEC_INT_1));
        __m256d q2_mask = _mm256_castsi256_pd(_mm256_cmpeq_epi64(vq, VEC_INT_2));
        __m256d neg_sin_r = _mm256_xor_pd(sin_r, VEC_SIGN_MASK);
        __m256d neg_cos_r = _mm256_xor_pd(cos_r, VEC_SIGN_MASK);
        __m256d s_result = neg_cos_r;
        s_result = _mm256_blendv_pd(s_result, neg_sin_r, q2_mask);
        s_result = _mm256_blendv_pd(s_result, cos_r, q1_mask);
        s_result = _mm256_blendv_pd(s_result, sin_r, q0_mask);
        s_result = _mm256_blendv_pd(s_result, vx, tiny_mask);
        s_result = _mm256_blendv_pd(s_result, VEC_NAN, special_mask);
        return s_result;
    }
}

namespace AwgOld{
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

    const short* minAvx2(const short *begin, const short *end)
    {
        const std::size_t chunkLeng = Awg::ArrayAlignment / sizeof (short);//32字节除以每一个数据的长度
        const short* min = begin;

        __m256i minMask = _mm256_set1_epi16(*min);
        __m256i minVec = _mm256_set1_epi16(*min);
        __m256i cmpVec = _mm256_set1_epi16(*min);
        __m256i dataVec = _mm256_set1_epi16(*min);
        while (begin+chunkLeng <= end)
        {
            dataVec  = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(begin));//加载数据到寄存器
            cmpVec = _mm256_min_epi16(dataVec,minVec);//将加载到寄存器中的值和最小值寄存器中的值作比较
            minMask = _mm256_cmpgt_epi16(minVec,cmpVec);//将比较结果和最小值寄存器做比较,判断是否产生了新的最小值

            //如果有新的最小值产生,则从这一组数据中找到最小值所在的索引
            if(_mm256_movemask_epi8(minMask))
            {
                min = std::min_element(begin,begin+chunkLeng);
                minVec = _mm256_set1_epi16(*min);
            }
            begin += chunkLeng;
        }

        if(begin == end)
            return min;
        else
        {
            const short* tmpMin = std::min_element(begin,end);
            return *min <= *tmpMin ? min : tmpMin;
        }
    }

    const short* maxAvx2(const short *begin, const short *end)
    {
        const std::size_t chunkLeng = Awg::ArrayAlignment / sizeof (short);//32字节除以每一个数据的长度
        const short* max = begin;

        __m256i maxMask = _mm256_set1_epi16(*max);
        __m256i maxVec = _mm256_set1_epi16(*max);
        __m256i cmpVec = _mm256_set1_epi16(*max);
        __m256i dataVec = _mm256_set1_epi16(*max);
        while (begin+chunkLeng <= end)
        {
            dataVec  = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(begin));//加载数据到寄存器
            cmpVec = _mm256_max_epi16(dataVec,maxVec);//将加载到寄存器中的值和最大值寄存器中的值作比较
            maxMask = _mm256_cmpgt_epi16(cmpVec,maxVec);//将比较结果和最大值寄存器做比较,判断是否产生了新的最大值

            //如果有新的最小值产生,则从这一组数据中找到最小值所在的索引
            if(_mm256_movemask_epi8(maxMask))
            {
                max = std::max_element(begin,begin+chunkLeng);
                maxVec = _mm256_set1_epi16(*max);
            }

            begin += chunkLeng;
        }

        if(begin == end)
            return max;
        else
        {
            const short* tmpMax = std::max_element(begin,end);
            return *max >= *tmpMax ? max : tmpMax;
        }
    }

    std::pair<const short *, const short *> minmaxAvx2(const short *begin, const short *end)
    {
        const std::size_t chunkLeng = Awg::ArrayAlignment / sizeof (short);
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
            minMask = _mm256_cmpgt_epi16(minVec,cmpMinVec);
            maxMask = _mm256_cmpgt_epi16(cmpMaxVec,maxVec);

            if(_mm256_movemask_epi8(minMask))
            {
                min = std::min_element(begin,begin+chunkLeng);
                minVec = _mm256_set1_epi16(*min);
            }

            if(_mm256_movemask_epi8(maxMask))
            {
                 max = std::max_element(begin,begin+chunkLeng);
                 maxVec = _mm256_set1_epi16(*max);
            }

            begin += chunkLeng;
        }

        if(begin == end)
            return std::make_pair(min,max);
        else
        {
            std::pair<const short *, const short *> ret = std::minmax_element(begin,end);
            ret.first = (*min) <= (*ret.first) ? min : ret.first;
            ret.second = (*max) >= (*ret.second) ? max : ret.second;
            return ret;
        }
    }

    std::pair<const double *, const double *> minmaxAvx2(const double *begin, const double *end)
    {
        const std::size_t chunkLeng = Awg::ArrayAlignment / sizeof (double);
        const double* max = begin;
        const double* min = begin;
        __m256d dataVec = _mm256_set1_pd(*min);
        __m256d minVec = _mm256_set1_pd(*min);
        __m256d cmpMinVec = _mm256_set1_pd(*min);
        __m256d maxVec = _mm256_set1_pd(*max);
        __m256d cmpMaxVec = _mm256_set1_pd(*max);
        while (begin+chunkLeng <= end)
        {
            dataVec  = _mm256_loadu_pd(begin);
            cmpMinVec = _mm256_cmp_pd(dataVec,minVec,_CMP_LT_OS);
            cmpMaxVec = _mm256_cmp_pd(dataVec,maxVec,_CMP_NLT_US);

            if(_mm256_movemask_pd(cmpMinVec))
            {
                min = std::min_element(begin,begin+chunkLeng);
                minVec = _mm256_set1_pd(*min);
            }

            if(_mm256_movemask_pd(cmpMaxVec))
            {
                max = std::max_element(begin,begin+chunkLeng);
                maxVec = _mm256_set1_pd(*max);
            }

            begin += chunkLeng;
        }

        if(begin == end)
            return std::make_pair(min,max);
        else
        {
            std::pair<const double *, const double *> ret = std::minmax_element(begin,end);
            ret.first = (*min) <= (*ret.first) ? min : ret.first;
            ret.second = (*max) >= (*ret.second) ? max : ret.second;
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
        const __m256i littleEndianMask = _mm256_setr_epi8(0,1,5,2,6,4,8,9,13,10,14,12,3,7,11,15,
                                                          0,1,5,2,6,4,8,9,13,10,14,12,3,7,11,15);//按小端模式重排字节顺序

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

    void doubleToShortScalar(short* output,const double* beg,const double* end)
    {
        while (beg < end)
        {
            *output = std::min(std::max(std::round(*beg), -32768.0), 32767.0);
            ++output;
            ++beg;
        }
    }

    ///这个函数在仅在当前文件中被调用,因此可以保证output和beg地址是按32字节对齐的,所以这里不需要额外处理未对齐的数据
    void doubleToShortAvx2(short* output,const double* beg,const double* end)
    {
        const int chunk = Awg::ArrayAlignment / sizeof (double);
        const __m256d short_max = _mm256_set1_pd(32767.0);
        const __m256d short_min = _mm256_set1_pd(-32768.0);
        while (beg + chunk*2 <= end)
        {
            __m256d val1 = _mm256_load_pd(beg);
            __m256d val2 = _mm256_load_pd(beg+chunk);

            // 使用round指令进行四舍五入
            __m256d rounded1 = _mm256_round_pd(val1, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m256d rounded2 = _mm256_round_pd(val2, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

            // 使用min/max进行饱和处理
            rounded1 = _mm256_min_pd(_mm256_max_pd(rounded1, short_min), short_max);
            rounded2 = _mm256_min_pd(_mm256_max_pd(rounded2, short_min), short_max);

            // 转换
            __m128i int32_1 = _mm256_cvtpd_epi32(rounded1);
            __m128i int32_2 = _mm256_cvtpd_epi32(rounded2);

            // 打包
            __m128i int16_8 = _mm_packs_epi32(int32_1, int32_2);
            _mm_store_si128(reinterpret_cast<__m128i*>(output), int16_8);

            beg += chunk*2;
            output += chunk*2;
        }

        doubleToShortScalar(output,beg,end);
    }

    ///计算正弦波形数据,数组为output,计算出来的数据写到[beg,end)区间
    void outputSinScalar(std::size_t totalPoints,double phaseRad,const double* output,double* beg,double* end)
    {
        while (beg < end)
        {
            double rad = 2.0 * Awg::PI * (beg - output) / totalPoints + phaseRad;
            *beg = Awg::Amplitude * (std::sin(rad)+1)/2;
            ++beg;
        }
    }

    void outputSinAvx2(std::size_t totalPoints,double phaseRad,const double* output,double* beg,double* end)
    {
        constexpr int chunk = Awg::ArrayAlignment / sizeof (double);
        const  __m256d piMul2 = _mm256_set1_pd(2.0*Awg::PI);
        const __m256d phaseRadVec = _mm256_set1_pd(phaseRad);
        const __m256d totalPointsVec = _mm256_set1_pd(totalPoints);
        const __m256d amplVec = _mm256_set1_pd(Awg::Amplitude);
        __m256d retVec = _mm256_set1_pd(0);
        __m256d indexVc = _mm256_set1_pd(0);
        while (beg + chunk <= end)
        {
            double index = beg - output;
            indexVc = _mm256_setr_pd(index,index+1,index+2,index+3);
            retVec = _mm256_mul_pd(indexVc,piMul2);
            retVec = _mm256_div_pd(retVec,totalPointsVec);
            retVec = _mm256_add_pd(retVec,phaseRadVec);
            //retVec = AvxSin::sinAvx2HighPrecision(retVec);
            //-o0优化下xsimd::sin的效率显著低于AvxSin::sinAvx2HighPrecision,但是在-o2优化下又于AvxSin::sinAvx2HighPrecision
            retVec = xsimd::sin(xsimd::batch<double,xsimd::avx>(retVec));//这里-o0优化并且直接指定axv2反而会导致段错误,很奇怪。
            retVec = _mm256_add_pd(retVec,_mm256_set1_pd(1.0));
            retVec = _mm256_div_pd(retVec,_mm256_set1_pd(2.0));
            retVec = _mm256_mul_pd(retVec,amplVec);
            _mm256_storeu_pd(beg,retVec);

            beg += chunk;
        }
        // 处理剩余的元素(不足4个的情况)
        outputSinScalar(totalPoints,phaseRad,output,beg,end);
    }

    void outputSquareScalar(const double* edge,double* beg,double* end)
    {
        while (beg < end)
        {
            *beg = (beg < edge) * Awg::Amplitude;//小于占空比索引为高电平,大于为低电平
            ++beg;
        }
    }

    void outputSquareAvx2(const double* edge,double* beg,double* end)
    {
        constexpr int chunk = Awg::ArrayAlignment / sizeof (double);
        const __m256d amplVec = _mm256_set1_pd(Awg::Amplitude);
        const __m256d zeroVec = _mm256_setzero_pd();

        while (beg + chunk <= end)
        {
            double* thunkEnd = beg + chunk;
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
                    _mm256_storeu_pd(beg, amplVec);
                else// 整个chunk都在edgeIndex右侧
                    _mm256_storeu_pd(beg, zeroVec);
                beg += chunk;
            }
        }

        // 处理剩余元素
        outputSquareScalar(edge,beg,end);
    }

    void outputTriangleScalar(double raiseK,double raiseB,double fallK,double fallB,const double* output,const double* peak,double* beg,double* end)
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

    void outputTriangleAvx2(double raiseK,double raiseB,double fallK,double fallB,const double* output,const double* peak,double* beg,double* end)
    {
        constexpr int chunk = Awg::ArrayAlignment / sizeof (double);
        const __m256d raiseBvec = _mm256_set1_pd(raiseB);
        const __m256d fallBvec = _mm256_set1_pd(fallB);
        const __m256d raiseKvec = _mm256_set1_pd(raiseK);
        const __m256d fallKvec = _mm256_set1_pd(fallK);
        __m256d indexVec = _mm256_setzero_pd();
        __m256d dataVec = _mm256_setzero_pd();
        while (beg + chunk <= end)
        {
            double* thunkEnd = beg + chunk;
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
                indexVec = _mm256_setr_pd(index,index+1,index+2,index+3);//索引反向加载
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
                _mm256_storeu_pd(beg,dataVec);

                beg += chunk;
            }
        }
        //处理剩余元素
        outputTriangleScalar(raiseK,raiseB,fallK,fallB,output,peak,beg,end);
    }

    template<typename T>
    AlignedSharedArray<T,Awg::ArrayAlignment> generateOverview(const T *data, const std::size_t length)
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
                std::pair<const T*,const T*> result = AwgOld::minmax((*groupBeg).first,(*groupBeg).second);

                buf[2*index] = (result.first < result.second) ? (*result.first) : (*result.second);
                buf[2*index+1] = (result.first < result.second) ? (*result.second) : (*result.first);

                ++groupBeg;
                ++index;
            }

            return buf;
        }
    }

    template<typename T>
    AlignedSharedArray<T,Awg::ArrayAlignment> generateOverviewMT(const T *data, const std::size_t length)
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
                    std::pair<const T*,const T*> result = AwgOld::minmax(pair.first,pair.second);

                    buf[2*i] = (result.first < result.second) ? (*result.first) : (*result.second);
                    buf[2*i+1] = (result.first < result.second) ? (*result.second) : (*result.first);
                }
            };

            //按分出来的组groupVec划分到线程池中,计算每一个线程处理多少组数据
            std::vector<std::size_t> threadGroups = AwgOld::splitLengthMin(Awg::MaxPlotPoints,500);
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

}

bool AwgOld::isFloatBegin(char c) noexcept
{
    // 数字、小数点、正负号都可能是浮点数的开头
    return ( (c >= '0' && c <= '9') || c == '-' || c == '.' );
}

bool AwgOld::isIntegerBegin(char c) noexcept
{
    // 数字、正负号都可能是浮点数的开头
    return ( (c >= '0' && c <= '9') || c == '-'  );
}

std::size_t AwgOld::countChar( const char *beg, const char *end, char target) noexcept
{
#ifdef __AVX2__
    return AwgOld::countCharAvx2(beg,end,target);
#else
    return AwgOld::countCharScalar(beg,end,target);
#endif
}

std::size_t AwgOld::countCharMT(const char *beg, const char *end, char target) noexcept
{
    const std::size_t length = end - beg;
    std::vector<std::size_t> chunks = AwgOld::splitLengthMin(length,Awg::MinArrayLength);
    std::vector< std::future<std::size_t> > futures;
    futures.reserve(chunks.size());

    ThreadPool* pool = Awg::globalThreadPool();
    for(std::size_t i = 0; i < chunks.size(); i++)
    {
        futures.push_back(pool->run(AwgOld::countChar,beg,beg+chunks[i],target));
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

const char* AwgOld::findChar(const char *beg, const char *end, char target) noexcept
{
#ifdef __AVX2__
    return AwgOld::findCharAvx2(beg,end,target);
#else
    return AwgOld::findCharScalar(beg,end,target);
#endif
}

const char* AwgOld::findCharMT(const char *beg, const char *end, char target) noexcept
{
    const std::size_t length = end - beg;
    std::vector<std::size_t> chunks = AwgOld::splitLengthMin(length,Awg::MinArrayLength);
    std::vector< std::future<const char*> > futures;
    futures.reserve(chunks.size());

    ThreadPool* pool = Awg::globalThreadPool();
    for(std::size_t i = 0; i < chunks.size(); i++)
    {
        futures.push_back(pool->run(AwgOld::findChar,beg,beg+chunks[i],target));
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

void AwgOld::reverse(char *beg, char *end)
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

const short *AwgOld::min(const short *begin, const short *end)
{
#ifdef __AVX2__
    return AwgOld::minAvx2(begin,end);
#else
    return std::min_element(begin,end);
#endif
}

const short* AwgOld::max(const short *begin, const short *end)
{
#ifdef __AVX2__
    return AwgOld::maxAvx2(begin,end);
#else
    return std::max_element(begin,end);
#endif
}

std::pair<const short *, const short *> AwgOld::minmax(const short *begin, const short *end)
{
#ifdef __AVX2__
    return AwgOld::minmaxAvx2(begin,end);
#else
    return std::minmax_element(begin,end);
#endif
}

std::pair<const double *, const double *> AwgOld::minmax(const double *begin, const double *end)
{
#ifdef __AVX2__
    return AwgOld::minmaxAvx2(begin,end);
#else
    return std::minmax_element(begin,end);
#endif
}

std::pair<const short *, const short *> AwgOld::minmaxMT(const short *begin, const short *end)
{
    std::size_t dataLen = end - begin + 1;
    std::vector<std::size_t> threadChunks = AwgOld::splitLengthAligned(dataLen,Awg::MinArrayLength,Awg::ArrayAlignment/sizeof (short));
    std::vector< std::future<std::pair<const short *, const short *>> > futures;
    futures.reserve(threadChunks.size());

    ThreadPool* pool = Awg::globalThreadPool();
    for(std::size_t i = 0; i < threadChunks.size(); i++)
    {
        futures[i] = pool->run(static_cast<std::pair<const short*, const short*> (*)(const short*, const short*)>(AwgOld::minmax),begin,begin+threadChunks[i]);
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

std::pair<const double *, const double *> AwgOld::minmaxMT(const double *begin, const double *end)
{
    std::size_t dataLen = end - begin + 1;
    std::vector<std::size_t> threadChunks = AwgOld::splitLengthAligned(dataLen,Awg::MinArrayLength,Awg::ArrayAlignment/sizeof (double));
    std::vector< std::future<std::pair<const double *, const double *>> > futures;
    futures.reserve(threadChunks.size());

    ThreadPool* pool = Awg::globalThreadPool();
    for(std::size_t i = 0; i < threadChunks.size(); i++)
    {
        futures[i] = pool->run(static_cast<std::pair<const double*, const double*> (*)(const double*, const double*)>(AwgOld::minmax),begin,begin+threadChunks[i]);
        begin += threadChunks[i];
    }
    pool->waitforDone();

    std::pair<const double *, const double *> ret = futures.at(0).get();
    for(std::size_t i = 1; i < futures.size(); i++)
    {
        std::pair<const double *, const double *> p = futures.at(i).get();
        ret.first = (*ret.first) < (*p.first) ? ret.first : p.first;
        ret.second = (*ret.second) > (*p.second) ? ret.second : p.second;
    }

    return ret;
}

void AwgOld::compressShort12Bit(const short *begin, const short *end, char *output)
{
#ifdef __AVX2__
    AwgOld::compressShort12BitAvx2(begin,end,output);
#else
    AwgOld::compressShort12BitScalar(begin,end,output);
#endif
}

AwgDoubleArray AwgOld::normalization(const AwgDoubleArray &input, const double inputMin, const double inputMax, const double outputMin, const double outputMax)
{
    if(input.empty())
        return AwgDoubleArray{};

    AwgDoubleArray output(input.size());
    const double* inputBeg = input.data();
    double* outputBeg = output.data();
#ifdef __AVX2__
    AwgOld::normalizationAvx2(inputBeg,inputBeg+input.size(),outputBeg,inputMin,inputMax,outputMin,outputMax);
#else
    AwgOld::normalizationScalar(inputBeg,inputBeg+input.size(),outputBeg,inputMin,inputMax,outputMin,outputMax);
#endif
    return output;
}

AwgDoubleArray AwgOld::normalizationMT(const AwgDoubleArray &input, const double inputMin, const double inputMax, const double outputMin, const double outputMax)
{
    if(input.empty())
        return AwgDoubleArray{};

    AwgDoubleArray output(input.size());
    std::vector<std::size_t> threadChunks = AwgOld::splitLengthAligned(input.size(),Awg::MinArrayLength,Awg::ArrayAlignment/sizeof (double));

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

AwgShortArray AwgOld::generateOverview(const short *data, const std::size_t length)
{
    return AwgOld::generateOverview<short>(data,length);
}

AwgDoubleArray AwgOld::generateOverview(const double *data, const std::size_t length)
{
    return AwgOld::generateOverview<double>(data,length);
}

AwgShortArray AwgOld::generateOverviewMT(const short *data, const std::size_t length)
{
    return AwgOld::generateOverviewMT<short>(data,length);
}

AwgDoubleArray AwgOld::generateOverviewMT(const double *data, const std::size_t length)
{
    return AwgOld::generateOverviewMT<double>(data,length);
}

std::vector<std::size_t> AwgOld::cutTextFile(QFile &file, std::size_t minChunk, const std::vector<char> &spliters)
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
                    const char* pos = AwgOld::findChar(beg,end,spliters[i]);
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

std::vector<std::size_t> AwgOld::cutBinaryFile(const std::size_t fileSize, const std::size_t minChunk, const unsigned dataBytes) noexcept
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

std::vector<std::size_t> AwgOld::splitLengthMin(std::size_t length, std::size_t minChunk) noexcept
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

std::vector<std::size_t> AwgOld::splitLengthMax(std::size_t length, std::size_t maxChunk) noexcept
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

std::vector<std::size_t> AwgOld::splitLengthAligned(std::size_t length,std::size_t minChunk,std::size_t aligned)  noexcept
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

AwgShortArray AwgOld::doubleToShort(const AwgDoubleArray &array)
{
    const double* beg = array.data();
    const double* end = beg + array.size();
    const std::size_t length = array.size();

    AwgShortArray output(length);
    if(output == nullptr)
        return output;

#ifdef __AVX2__
    doubleToShortAvx2(output.data(),beg,end);
#else
    doubleToShortScalar(output.data(),beg,end);
#endif
    return output;
}

AwgShortArray AwgOld::doubleToShortMT(const AwgDoubleArray &array)
{
    AwgShortArray output(array.size());
    if(output == nullptr)
        return output;

    std::size_t index = 0;
    std::vector<std::size_t> chunks = AwgOld::splitLengthAligned(array.size(),Awg::MinArrayLength,Awg::ArrayAlignment/sizeof(short));//这里应该以short大小倍数为准
    ThreadPool* pool = Awg::globalThreadPool();
    for(std::size_t i = 0; i < chunks.size(); i++)
    {
        short* optBeg = output.data() + index;
        const double* beg = array.data() + index;
        const double* end = beg + chunks[i];
#ifdef __AVX2__
        pool->run<ThreadPool::Ordered>(AwgOld::doubleToShortAvx2,optBeg,beg,end);
#else
        pool->run<ThreadPool::Ordered>(AwgOld::doubleToShortScalar,optBeg,beg,end);
#endif
        index += chunks[i];
    }
    pool->waitforDone();
    return output;
}

AwgDoubleArray AwgOld::generateSin(double sampleRate, double frequency, double phase)
{
    // 计算一个完整周期内的采样点数
    const std::size_t totalPoints = std::round(sampleRate / frequency);
    const double phaseRad = phase * Awg::PI / 180.0;
    
    if (totalPoints == 0)
    {
        std::cerr << "Error: sampleRate must be greater than frequency" << std::endl;
        return AwgDoubleArray{};
    }
    
    // 分配内存存储波形数据
    AwgDoubleArray waveform(totalPoints);
    
    if (waveform == nullptr)
    {
        std::cerr << "Error: Memory allocation failed" << std::endl;
        return AwgDoubleArray{};
    }

    //每一个线程处理的点数为总点数/线程数再向上取整
    std::size_t index = 0;
    std::vector<std::size_t> chunks = AwgOld::splitLengthAligned(totalPoints,Awg::MinArrayLength,Awg::ArrayAlignment/sizeof(double));
    ThreadPool* pool = Awg::globalThreadPool();
    for(std::size_t i = 0; i < chunks.size(); i++)
    {
        double* output = waveform.data();
        double* beg = output + index;
        double* end = beg + chunks[i];
#if __AVX2__
        pool->run<ThreadPool::Ordered>(AwgOld::outputSinAvx2,totalPoints,phaseRad,output,beg,end);
#else
        pool->run<ThreadPool::Ordered>(AwgOld::outputSinScalar,totalPoints,phaseRad,output,beg,end);
#endif
        index += chunks[i];
    }

    pool->waitforDone();
    return waveform;
}

AwgDoubleArray AwgOld::generateSquare(double sampleRate, double frequency, double duty)
{
    //每个周期最少100个点,这样可以将占空比的精度控制到1%
    const unsigned minPointsPerPeriod = 100;
    const std::size_t totalPoints = std::round(sampleRate / frequency);
    if(totalPoints < minPointsPerPeriod)
        return AwgDoubleArray{};

    // 分配内存存储波形数据
    AwgDoubleArray waveform(totalPoints);

    //将占空比限制到0和1之间
    duty = std::max(duty,0.0);
    duty = std::min(duty,1.0);
    std::size_t edgeIndex = std::round(totalPoints * duty);
    double* output = waveform.data();
    double* edge = output + edgeIndex;

    //每一个线程处理的点数为总点数/线程数再向上取整
    std::size_t index = 0;
    std::vector<std::size_t> chunks = AwgOld::splitLengthAligned(totalPoints,Awg::MinArrayLength,Awg::ArrayAlignment/sizeof(double));
    ThreadPool* pool = Awg::globalThreadPool();
    for(std::size_t i = 0; i < chunks.size(); i++)
    {
        double* beg = output + index;
        double* end = beg + chunks[i];
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

AwgDoubleArray AwgOld::generateTriangle(double sampleRate, double frequency, double symmetry)
{
    //每个周期最少100个点,这样可以将对称性的精度控制到1%
    const unsigned minPointsPerPeriod = 100;
    const std::size_t totalPoints = std::round(sampleRate / frequency);
    if(totalPoints < minPointsPerPeriod)
        return AwgDoubleArray{};

    // 分配内存存储波形数据
    AwgDoubleArray waveform(totalPoints);

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
    double* output = waveform.data();
    double* peak = output + peakIndex;

    //每一个线程处理的点数为总点数/线程数再向上取整
    std::size_t index = 0;
    std::vector<std::size_t> chunks = AwgOld::splitLengthAligned(totalPoints,Awg::MinArrayLength,Awg::ArrayAlignment/sizeof(double));
    ThreadPool* pool = Awg::globalThreadPool();
    for(std::size_t i = 0; i < chunks.size(); i++)
    {
        double* beg = output + index;
        double* end = beg + chunks[i];
#ifdef __AVX2__
        pool->run<ThreadPool::Ordered>(AwgOld::outputTriangleAvx2,raiseK,raiseB,fallK,fallB,output,peak,beg,end);
#else
        pool->run<ThreadPool::Ordered>(AwgOld::outputTriangleScalar,raiseK,raiseB,fallK,fallB,output,peak,beg,end);
#endif
        index += chunks[i];
    }

    pool->waitforDone();
    return waveform;
}

AwgDoubleArray AwgOld::generateNoise(double sampleRate, double bandWidth)
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

