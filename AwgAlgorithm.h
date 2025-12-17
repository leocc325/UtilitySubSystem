#ifndef AWGALGORITHM_H
#define AWGALGORITHM_H

#ifndef AWGARRAY_H
#include "AwgArray.hpp"
#endif

class QFile;
namespace Awg {
///函数后缀Avx2表示这个函数内部使用了Avx2指令进行硬件计算加速,后缀MT表示函数内部使用了多线程进行软件计算加速,Fast表示同时使用进行了软件和硬件加速

///2025.12.1更新:所有带后缀的函数都移动到cpp文件中,函数在编译器允许的条件下始终会使用更快的算法,对于函数调用者来说不需要再自行区分需要调用哪种算法
/// 函数内部都做了向量计算和标量计算两种实现以及线程池并行计算(如果可行的话),所以大部分函数的执行效率都是很高的,未使用软件和硬件加速的算法也会在后期补充

///2025.12.2更新:函数名不再区分标量和矢量版本,函数内部会自动选择对应的版本,只在签名上区分是否使用了多线程并行计算
///
/// 2025.12.10更新:波形生成算法返回double数组,新增double数组转short数组的函数

///判断字符c是否是浮点数的开头
bool isFloatBegin(char c) noexcept;

///判断字符c是否是整数的开头
bool isIntegerBegin(char c) noexcept;

///计算字符串中出现字符target的次数
std::size_t countChar(const char* beg, const char* end, char target) noexcept;

///计算字符串中出现字符target的次数(多线程版本)
std::size_t countCharMT(const char* beg, const char* end, char target) noexcept;

///查找字符串中第一次出现字符target的位置,如果未找到匹配的字符则返回空指针,否则返回字符指针
const char* findChar(const char* beg,const char* end, char target) noexcept;

///查找字符串中第一次出现字符target的位置(多线程版本)
const char* findCharMT(const char* beg,const char* end, char target) noexcept;

/// 从给定short数组中查找最小值并返回第一个最小值指针,在极端情况下(数组已经按从大到小排序)这个函数的效率可能会略低于std::min_element
const short* min(const short* beg,const short* end);

/// 从给定short数组中查找最大值并返回第一个最大值指针,在极端情况下(数组已经按从小到大排序)这个函数的效率可能会略低于std::max_element
const short* max(const short* beg,const short* end);

///从给定的数组中查找最小值和最大值,返回最小值和最大值指针,avx2版本和std版本返回结果不一样
/// avx2版本返回的是第一个最小值和第一个最大值指针,std版本返回的是第一个最小值和最后一个最大值指针
std::pair<const short*,const short*> minmax(const short *beg, const short*end);
std::pair<const double*,const double*> minmax(const double *beg, const double*end);

///从给定的数组中查找最小值和最大值,返回最小值和最大值指针(多线程版)
std::pair<const short*,const short*> minmaxMT(const short *beg, const short*end);
std::pair<const double*,const double*> minmaxMT(const double *beg, const double*end);

///将short数组中的每一个值取12bit压缩写入到二进制内存中,返回写入的长度,需要保证output长度足够写入全部数据,否则会导致程序崩溃
void compressShort12Bit(const short* beg,const short* end,char* output);//1024bit,前768bit存放，后256比特不适用

///将给定的double数组归一化,并返回一个字节对齐的数据,rangeLow和rangeHigh表示原始数据的范围,min和max表示归一化之后的数据范围
AwgDoubleArray normalization(const AwgDoubleArray& input,const double inputMin,const double inputMax,const double outputMin,const double outputMax);

///将给定的double数组归一化,并返回一个字节对齐的数据,rangeLow和rangeHigh表示原始数据的范围,min和max表示归一化之后的数据范围(多线程版本)
AwgDoubleArray normalizationMT(const AwgDoubleArray& input,const double inputMin,const double inputMax,const double outputMin,const double outputMax);

///数据点超过最大像素点之后按包络生成数据的略缩图
AwgShortArray generateOverview(const short* data,const std::size_t length);
AwgDoubleArray generateOverview(const double* data,const std::size_t length);

///数据点超过最大像素点之后按包络生成数据的略缩图(多线程版本)
AwgShortArray generateOverviewMT(const short* data,const std::size_t length);
AwgDoubleArray generateOverviewMT(const double* data,const std::size_t length);

///根据线程池线程数将文本文件切割成若干个大小至少为minCunk的小内存块,同时保证里面的数据不被切割到不同的块中,返回每一个块的大小
std::vector<std::size_t> cutTextFile(QFile& file, std::size_t minChunk, const std::vector<char>& spliters);

///根据线程池线程数将二进制文件切割成若干个大小至少为minCunk的小内存块,同时保证里面的数据不被切割到不同的块中,不同的块的大小为dataBytes的整倍数,返回每一个块的大小
std::vector<std::size_t> cutBinaryFile(const std::size_t fileSize,const std::size_t minChunk,const unsigned dataBytes) noexcept;

///根据线程池线程数将一个长度为length块切割成若干个大小至少为minChunk的小数组,返回这些数组的长度
std::vector<std::size_t> cutArrayMin(std::size_t length,std::size_t minChunk) noexcept;

///将一个长度为length块切割成若干个大小至多为maxChunk的小数组,返回这些数组的长度
std::vector<std::size_t> cutArrayMax(std::size_t length,std::size_t maxChunk) noexcept;

///根据线程池线程数将一个长度为length块切割成若干个大小至少为minChun而且为aligned整倍数的小数组,返回这些数组的长度
std::vector<std::size_t> cutArrayAligned(std::size_t length,std::size_t minChunk,std::size_t aligned) noexcept;

///数据类型转换
AwgShortArray doubleToShort(const AwgDoubleArray& array);

///生成正弦波波形
AwgDoubleArray generateSin(double sampleRate,double frequency,double phase);

///生成方波波形
AwgDoubleArray generateSquare(double sampleRate,double frequency,double duty);

///生成三角波形
AwgDoubleArray generateTriangle(double sampleRate,double frequency,double symmetry);

///生成噪声波形
AwgDoubleArray generateNoise(double sampleRate,double bandWidth);

}

#endif // AWGALGORITHM_H
