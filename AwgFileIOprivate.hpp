#ifndef AWGFILEIOPRIVATE_H
#define AWGFILEIOPRIVATE_H

#ifndef FMT_FORMAT_H_
#define FMT_HEADER_ONLY
#include "UtilitySubSystem/fmt/format.h"
#endif

namespace Awg {
    /// 换行符
    const std::string NewLine = "\r\n";

    ///检查参数是否全部都是数字类型
    template<typename... T>
    struct IsArithmetic;

    template<typename T>
    struct IsArithmetic<T>
    {
        constexpr static bool value = std::is_arithmetic<typename std::decay<T>::type>::value;
    };

    template<typename First, typename... Rest>
    struct IsArithmetic<First, Rest...>
    {
        constexpr static bool value = IsArithmetic<First>::value && IsArithmetic<Rest...>::value;
    };

    ///每一种数据类型对应的字符串最大长度,可以根据实际情况更改
    ///目前定义这些长度主要目的是在转换为字符串之前预估所需内存大小
    /// 这里没有包含全部类型的数据,比如long、unsigned long 、long double等,如有需求请自行添加
    template<typename T>
    struct ArithmeticLength
    {
        static_assert (IsArithmetic<T>::value,"Target Type must be a Arithmetic");
        constexpr static int value = 0;
    };

    template<> struct ArithmeticLength<bool>{constexpr static int value = 1;};

    template<> struct ArithmeticLength<char>{constexpr static int value = 1;};

    template<> struct ArithmeticLength<unsigned char>{constexpr static int value = 1;};

    template<> struct ArithmeticLength<short>{constexpr static int value = 6;};

    template<> struct ArithmeticLength<unsigned short>{constexpr static int value = 5;};

    template<> struct ArithmeticLength<int>{constexpr static int value = 11;};

    template<> struct ArithmeticLength<unsigned int>{constexpr static int value = 10;};

    template<> struct ArithmeticLength<long long>{constexpr static int value = 20;};

    template<> struct ArithmeticLength<unsigned long long>{constexpr static int value = 20;};

    ///fmt转换float字符串默认格式长度一般都是20字节以内
    template<> struct ArithmeticLength<float>{constexpr static int value = 20;};

    ///fmt转换double字符串默认格式长度一般都是30字节以内
    template<> struct ArithmeticLength<double>{constexpr static int value = 30;};

    ///求所有给定数据类型转换为字符串之后的最大总长度
    template<typename... T>
    struct ArithmeticLengthSum
    {
        constexpr static int value = 0;
        };

    template<typename F, typename... T>
    struct ArithmeticLengthSum<F, T...>
    {
        constexpr static int value = ArithmeticLength<F>::value + ArithmeticLengthSum<T...>::value;
    };

    ///让全部指针指向下一个位置,这里需要添加对类型T的限制,判断是否全部都是数字类型
    template<typename First>
    void next(const First*& first)
    {
        ++first;
    }

    template<typename First,typename...T>
    typename std::enable_if<IsArithmetic<T...>::value,void>::type
    next(const First*& first,const T*&...Arrays)
    {
        ++first;
        next(Arrays...);
    }

    ///生成一个cvs格式的fmt::format_to的格式字符串,用于将N个数组写为N列的csv文件
    template<typename...T>
    typename std::enable_if<IsArithmetic<T...>::value,std::string>::type
    csvRowFormat(const T*...arrays)
    {
        std::string ret = "";
        constexpr unsigned ArrayNum = sizeof... (arrays);
        for(unsigned i = 0; i < ArrayNum-1; i++)
        {
            ret.append("{},");
        }
        return ret.append("{}"+NewLine);
    }

    ///生成一个txt格式的fmt::format_to的格式字符串,用于将N个数组写为N列的txt文件
    template<typename...T>
    typename std::enable_if<IsArithmetic<T...>::value,std::string>::type
    txtRowFormat(const T*...arrays)
    {
        std::string ret = "";
        constexpr unsigned ArrayNum = sizeof... (arrays);
        for(unsigned i = 0; i < ArrayNum-1; i++)
        {
            ret.append("{} ");//这里是{}+空格
        }
        return ret.append("{}"+NewLine);
    }

    ///将指定的数组合集转换为csv格式的字符串并返回std::string
    /*template<typename...T>
    typename std::enable_if<IsArithmetic<T...>::value,std::string>::type
    toBinaryCsv(const std::size_t length,const T*...Arrays)
    {
        //计算数值转换为字符串之后每一行的长度:每一个数据的字符串长度和+分隔符数量(分隔符长度为1)+换行符长度。(实际上这里应该是分隔符-1)
        constexpr int lineLength = ArithmeticLengthSum<T...>::value + sizeof... (Arrays) + NewLine.size();
        const std::string format = csvRowFormat(Arrays...);

        std::string s;
        s.reserve(lineLength * length);
        auto inserter = std::back_inserter(s);

        for(std::size_t i = 0; i < length; i++)
        {
            fmt::format_to(inserter, format,*Arrays...);
            next(Arrays...);
        }
        return s;
    }*/

    ///将数组以csv格式写到output指针中,返回值为输出的起始到结束范围
    template<typename...T>
    typename std::enable_if<IsArithmetic<T...>::value,std::pair<char*,char*>>::type
    toBinaryCsv(char* output,const std::size_t length,const T*...Arrays)
    {
        std::pair<char*,char*> range{output,nullptr};
        const std::string format = csvRowFormat(Arrays...);
        for(std::size_t i = 0; i < length; i++)
        {
            output = fmt::format_to(output, format,*Arrays...);
            next(Arrays...);
        }
        range.second = output;
        return range;
    }

    ///将指定的数组合集转换为txt格式的字符串并返回std::string
    /*template<typename...T>
    typename std::enable_if<IsArithmetic<T...>::value,std::string>::type
    toBinaryTsv(const std::size_t length,const T*...Arrays)
    {
        //计算数值转换为字符串之后每一行的长度:每一个数据的字符串长度和+分隔符数量(分隔符长度为1)+换行符长度。(实际上这里应该是分隔符-1)
        constexpr int lineLength = ArithmeticLengthSum<T...>::value + sizeof... (Arrays) + NewLine.size();
        const std::string format = txtRowFormat(Arrays...);

        std::string s;
        s.reserve(lineLength * length);
        auto inserter = std::back_inserter(s);

        for(std::size_t i = 0; i < length; i++)
        {
            fmt::format_to(inserter, format,*Arrays...);
            next(Arrays...);
        }
        return s;
    }*/

    ///将数组以txt格式写到output指针中,返回值为输出的起始到结束范围
    template<typename...T>
    typename std::enable_if<IsArithmetic<T...>::value,std::pair<char*,char*>>::type
    toBinaryTxt(char* output,const std::size_t length,const T*...Arrays)
    {
        std::pair<char*,char*> range{output,nullptr};
        std::string format = txtRowFormat(Arrays...);
        for(std::size_t i = 0; i < length; i++)
        {
            output = fmt::format_to(output, format,*Arrays...);
            next(Arrays...);
        }
        range.second = output;
        return range;
    }
}

#endif // AWGFILEIOPRIVATE_H
