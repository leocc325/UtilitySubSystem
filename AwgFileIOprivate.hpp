#ifndef AWGFILEIOPRIVATE_H
#define AWGFILEIOPRIVATE_H

#define FMT_HEADER_ONLY
#include "UtilitySubSystem/fmt/format.h"
#include "UtilitySubSystem/AwgSignals.h"

namespace Awg {
    //2026.1.15txt和csv的枚举值要和AwgFileIO.h中的枚举值保持一致
    enum TextFormat{Txt,Csv};

    inline const std::string& numericSpliter(TextFormat format)
    {
        static const std::string csv(",");
        static const std::string txt(" ");
        switch (format)
        {
            case Txt:return txt;
            case Csv:return csv;
            default:assert("error split type");
        }
    }

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
    ///目前定义这些长度主要目的是在转换为字符串之前预估所需内存大小,实际使用时可以根据数据范围调整输出字符串最大长度
    ///比如整形long long数值范围在[4096,-4095]就可以将对应模板ArithmeticLength值调整为5字节而非20
    ///这里没有包含全部类型的数据,比如long、unsigned long 、long double等,如有需求请自行添加
    template<typename T>
    struct ArithmeticLength
    {
        static_assert (IsArithmetic<T>::value,"Target Type must be a Arithmetic");
        constexpr static int value = 0;
    };

    template<typename T,typename...Pack>
    struct PackType
    {
        using type = T;
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

    ///2026.1.14
    ///fmt转换float字符串默认格式长度一般都是20字节以内,但是导出的浮点值本身不会大于1000,所以这里暂时就只固定的到10字节宽度
    ///这样好处有很多:
    ///可以根据点数数量计算对应的文本文件大小,直接创建一个大小合适的文件
    ///在多线程并发写入文件的时候,可以直接根据每个线程划分的数据长度确定文本数据应该写入到哪一块内存中
    ///由于可以直接写入到文件映射的内存中,不需要再预估所需的内存大小,省去临时内存申请、临时内存拷贝到文件内存、最终文件大小resize这三个耗时的操作
    ///不用担心内存不足(数据量较大的情况)导致文件写入失败的情况,因为数据本身、保存字符串的临时内存,文件映射内存这三部分内存开销很高,现在在文件创建时就可以判断内存是否足够写入全部数据,没有中间的内存开销
    ///2026.1.15
    ///浮点值默认长度保持10字节(如果用fmt默认输出长度1G数据输出的文本文件大小可能达到20G),由此可以精确计算最终的文件大小,随后就可以根据文件大小和数据长度划分线程任务
    ///同时可以得到每一组数据写入文件的起始位置,然后再多线程写入文件即可,同样避免了上述的三个耗时操作(计算文件总长度开销小于上述三个操作)以及内存不足的风险
    template<> struct ArithmeticLength<float>{constexpr static int value = 10;};

    ///2026.1.14
    ///fmt转换double字符串默认格式长度一般都是30字节以内,但是导出的浮点值本身不会大于1000,所以这里暂时就只固定的到10字节宽度
    ///2026.1.15
    ///同上
    template<> struct ArithmeticLength<double>{constexpr static int value = 10;};

    ///求所有给定数据类型转换为字符串之后的最大总长度
    template<typename... T>
    struct ArithmeticLengthSum;

    template<typename F, typename... T>
    struct ArithmeticLengthSum<F, T...>
    {
        constexpr static int value = ArithmeticLength<F>::value + 1 + ArithmeticLengthSum<T...>::value;//如果不是最后一个就+1,加的是分隔符,对应csv文件的','和txt文件的空格
    };

    template<typename F>
    struct ArithmeticLengthSum<F>
    {
        constexpr static int value = ArithmeticLength<F>::value + 2;//如果是最后一个就+2,这个2是换行符的长度
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

    ///根据数据类型生成一个fmt格式字符串,浮点值
    template<typename T>
    typename std::enable_if<std::is_floating_point<T>::value,const std::string&>::type
    numericFormat()
    {
        constexpr int Width = ArithmeticLength<T>::value;
        static std::string s = fmt::format("{{:<{}f}}", Width);//{{:<Nf}},其中N=Width表示:对于浮点值,左对齐,宽度不足width时补空格,浮点值不使用科学计数法
        return s;
    }

    ///根据数据类型生成一个fmt格式字符串,整型值
    template<typename T>
    typename std::enable_if<std::is_integral<T>::value,const std::string&>::type
    numericFormat()
    {
        constexpr int Width = ArithmeticLength<T>::value;
        static std::string s = fmt::format("{{:<{}}}", Width);//{{:<N}},其中N=Width表示:对于整形值,左对齐,宽度不足width时补空格
        return s;
    };

    ///对N个数组按行生成fmt::format_to的输出格式字符串,用于将N个数组写为N列的文本文件
    template<typename...T>
    typename std::enable_if<IsArithmetic<T...>::value,const std::string>::type
    rowFormat(TextFormat tf,const T*...arrays)
    {
        std::string s;
        rowFormatImpl(tf,s,arrays...);
        return s;
    }

    template<typename F,typename...T>
    void rowFormatImpl(TextFormat tf,std::string& output,const F*,const T*...arrays)
    {
        output.append(numericFormat<F>() + numericSpliter(tf));
        rowFormatImpl(tf,output,arrays...);
    }

    template<typename F>
    void rowFormatImpl(TextFormat,std::string& output,const F*)
    {
        output.append(numericFormat<F>() + NewLine);
    }

    ///将数组以csv格式写到output指针中,返回值为输出的起始到结束范围
    template<typename...T>
    typename std::enable_if<IsArithmetic<T...>::value>::type
    toBinaryCsv(char* output,const std::size_t length,const T*...Arrays)
    {
        const std::string format = rowFormat(Awg::Csv,Arrays...);
        std::size_t process = 0;
        for(std::size_t i = 0; i < length; ++i,++process)
        {
            output = fmt::format_to(output, format,*Arrays...);
            next(Arrays...);
            if(process > 1e6)
            {
                emit AWGSIG->sigFileProcess(process);//每转换1M个点发送一次信号更新进度
                process = 0;
            }
        }
        emit AWGSIG->sigFileProcess(process);
    }

    ///将数组以txt格式写到output指针中,返回值为输出的起始到结束范围
    template<typename...T>
    typename std::enable_if<IsArithmetic<T...>::value>::type
    toBinaryTxt(char* output,const std::size_t length,const T*...Arrays)
    {
        std::string format = rowFormat(Awg::Txt,Arrays...);
        std::size_t process = 0;
        char* pos = output;
        for(std::size_t i = 0; i < length; ++i,++process)
        {
            pos = fmt::format_to(output, format,*Arrays...);
            next(Arrays...);
            if(process > 1e6)
            {
                emit AWGSIG->sigFileProcess(process);//每转换1M个点发送一次信号更新进度
                process = 0;
            }
            output = pos;
        }
        emit AWGSIG->sigFileProcess(process);
    }

    ///计算长度为length的数组T...转换为字符串,但是每个数值转换的字符串长度固定
    template<typename...T>
    typename std::enable_if<IsArithmetic<T...>::value>::type
    toBinaryTxtFixedWidth(TextFormat tf,char* output,const std::size_t arrayLenth,const T*...Arrays)
    {
        std::size_t process = 0;
        char* pos = output;
        for(std::size_t i = 0; i < arrayLenth; ++i,++process)
        {
            pos = toText_FWImpl(tf,output,Arrays...);//递归写入第i行数据
            next(Arrays...);//准备下一行
            if(process > 1e6)
            {
                emit AWGSIG->sigFileProcess(process);//每转换1M个点发送一次信号更新进度
                process = 0;
            }
            output = pos;
        }
        emit AWGSIG->sigFileProcess(process);
    }

    template<typename F,typename...T>
    typename std::enable_if<IsArithmetic<T...>::value,char*>::type
    toText_FWImpl(TextFormat tf,char* output,const F* First,const T*...Arrays)
    {
        const std::string& nf = numericFormat<F>();
        output = fmt::format_to_n(output,ArithmeticLength<F>::value, nf,*First).out;//将浮点值以固定宽度写入内存
        output = fmt::format_to(output,"{}",numericSpliter(tf));//数值输出完成之后添加一个分隔符(如果后面还有数组N列数组)
        return toText_FWImpl(tf,output,Arrays...);//递归输出下一个数组的元素
    }

    template<typename F>
    typename std::enable_if<IsArithmetic<F>::value,char*>::type
    toText_FWImpl(TextFormat tf,char* output,const F* First)
    {
        const std::string& nf = numericFormat<F>();
        output = fmt::format_to_n(output,ArithmeticLength<F>::value, nf,*First).out;//将浮点值以固定宽度写入内存
        output = fmt::format_to(output,"{}",Awg::NewLine);//如果是最后一个数组在写入完毕之后添加换行符
        return output;
    }

}

#endif // AWGFILEIOPRIVATE_H
