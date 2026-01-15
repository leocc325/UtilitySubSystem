#ifndef AWGFILEIO_H
#define AWGFILEIO_H

#ifndef QLIST_H
#include <QList>
#endif

#ifndef QSTRING_H
#include <QString>
#endif

#ifndef AWGARRAY_H
#include "AwgArray.hpp"
#endif

class QFile;
namespace Awg
{
/**
     * 超过1G的文件最好不要直接将整个文件映射到虚拟内存中,由于操作系统内存管理机制,被映射的内存在读取的过程中会被逐步加载到物理内存中
     * 物理内存的占用取决于你实际访问了多少数据
     * 如果只访问了文件开头100MB的数据，那么大约只有100MB的文件内容被加载到物理内存中。
     * 如果从头到尾顺序访问了所有10GB的数据，那么最终这10GB文件内容都会在物理内存中(假设系统有足够的可用RAM)
     * 被加载到物理内存中的文件数据只有在解除映射之后才会被回收,也就是说:
     * 如果不分块,一次性映射10GB文本文件,并且同时还要存储1GB的double数组，那么峰值内存占用将是:10GB(文本文件在物理内存中) + 8GB(double数组) = 18GB。
     * 所以在读取大文件的时候需要确保各个线程映射的文件大小总和的两倍不会耗尽系统内存
     *
     * 超过1G的文件就算是超大文件,通过文件映射和并发读取,处理这类文件需要同时考虑内存剩余空间和CPU核心能处理的线程数
     * 首先确定能够开辟的线程数,预留两个核心给系统,其他核心全部执行文件读取任务
     * 随后根据文件大小剩余内存空间的大小确定每一个线程映射大小
     * 最后统一处理被映射阶段的数据
     *
     * 文件加载不需要根据大小区分不同的接口,接口内部会自动根据文件大小划分任务
    */

    enum FileFormat
    {
        FmtTxt,
        FmtCsv,
        FmtBin,
        FmtNum
    };//2026.1.15txt和csv的枚举值要和AwgFileIOprivate.hpp中的枚举值保持一致

    static const QList<QString> FileSuffixStringList
    {
        QString("txt"),QString("csv"),QString("bin")
    };

    static const QList<QString> FormatStringList
    {
        QString(".txt"),QString(".csv"),QString(".bin")
    };

    static const QList<QString> FilterStringList
    {
        QString("*.txt"),QString("*.csv"),QString("*.bin")
    };

    void storeBinFile(const QString& path,const double* array,const std::size_t length);

    void storeCsvFile(const QString& path,const double* array,const std::size_t length);

    void storeTxtFile(const QString& path,const double* array,const std::size_t length);

    AwgDoubleArray loadBinFile(const QString& path);

    AwgDoubleArray loadCsvFile(const QString& path);

    AwgDoubleArray loadTxtFile(const QString& path);

    AwgDoubleArray processBinFile(QFile *file, std::size_t mapStart, std::size_t mapSize);

    AwgDoubleArray processTextFile(QFile *file, std::size_t mapStart, std::size_t mapSize,const std::vector<char>& spliters);
}

#endif // AWGFILEIO_H
