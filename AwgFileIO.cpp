#include "AwgFileIO.h"
#include <iostream>
#include <fstream>
#include <QFile>
#include "UtilitySubSystem/AwgUtility.h"
#include "UtilitySubSystem/ThreadPool.hpp"
#include "UtilitySubSystem/fastfloat/fast_float.h"
#include "UtilitySubSystem/fmt/format.h"

std::mutex FileMutex;

void generateBin()
{
    const int64_t MAX_NUMBER = 1000000000; // 10亿
    const size_t BUFFER_SIZE = 1000000;    // 缓冲区大小：100万个整数（约4MB）

    std::cout << "Generating binary file with integers from 1 to " << MAX_NUMBER << std::endl;
    std::cout << "Estimated file size: " << (MAX_NUMBER * sizeof(int32_t) / (1024.0 * 1024.0 * 1024.0))
              << " GB" << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();

    std::ofstream outfile("D:/Desktop/testData1G.bin", std::ios::binary);
    if (!outfile) {
        std::cerr << "Error: Could not open file for writing!" << std::endl;
        return;
    }

    std::vector<int32_t> buffer;
    buffer.reserve(BUFFER_SIZE);

    int64_t current = 0;
    int64_t percent_step = MAX_NUMBER / 100;
    int last_percent = -1;

    while (current < MAX_NUMBER) {
        // 填充缓冲区
        buffer.clear();
        for (size_t i = 0; i < BUFFER_SIZE && current <= MAX_NUMBER; ++i, ++current) {
            buffer.push_back(static_cast<int32_t>(current));
        }

        // 写入缓冲区内容
        outfile.write(reinterpret_cast<const char*>(buffer.data()),
                      buffer.size() * sizeof(int32_t));

        // 显示进度
        int percent = static_cast<int>((current * 100) / MAX_NUMBER);
        if (percent != last_percent && percent % 5 == 0) {
            std::cout << "Progress: " << percent << "%" << std::endl;
            last_percent = percent;
        }
    }

    outfile.close();
}

void generateCsv()
{
    const int totalRows = 1000; // 1000万行
    const std::string filename = "D:/Desktop/testData1k.csv";

    // 打开文件
    std::ofstream outFile(filename,std::ios::trunc);
    if (!outFile.is_open()) {
        std::cerr << "无法创建文件: " << filename << std::endl;
        return;
    }

    // 设置输出格式
    outFile << std::fixed;

    std::cout << "正在生成 " << totalRows << " 行数据..." << std::endl;

    // 生成CSV数据
    for (int i = 0; i < totalRows; ++i) {
        outFile << i;

        // 如果不是最后一行，添加换行符
        if (i < totalRows - 1) {
            outFile << "\n";
        }
    }

    // 关闭文件
    outFile.close();
}

void generateTxt()
{
    const int totalRows = 1000000000; // 1000万行
    const std::string filename = "D:/Desktop/testData1G.txt";

    // 打开文件
    std::ofstream outFile(filename,std::ios::trunc);
    if (!outFile.is_open()) {
        std::cerr << "无法创建文件: " << filename << std::endl;
        return;
    }

    // 设置输出格式
    outFile << std::fixed;

    // 生成Txt数据
    for (int i = 0; i < totalRows; ++i) {
        outFile << i;

        // 如果不是最后一行，添加换行符
        if (i < totalRows - 1) {
            outFile << "\n";
        }
    }

    // 关闭文件
    outFile.close();
}

void Awg::storeBinFile(const QString &path)
{

}

void Awg::storeCsvFile(const QString &path,const AwgShortArray& data)
{
    QFile file(path);
    if(file.open(QIODevice::ReadWrite|QIODevice::Truncate))
    {
        //首先预估文件大小,假设short数组每一个数都是-32768
        //那么每一个数写入到csv文件之后需要占用-32768/r/n字节(8字节)长度(因为这是单行的,所以不需要计算逗号的长度)
        const std::size_t rowLen = 8;
        const std::size_t fileSize = data.size() * rowLen;
        if(file.resize(fileSize))
        {
            //根据当前内存使用情况规划线程数和线程处理的内存块大小避免写入文件的时候内存耗尽
            const std::size_t freeMemory = Awg::getFreeMemoryWindows() * 0.9;
            const std::size_t averageSize = freeMemory / Awg::PoolSize;
            const std::size_t chunkSize = averageSize > Awg::MinFileChunk ? averageSize : Awg::MinFileChunk;
            //根据每一个线程可以处理的最大内存长度计算每一个线程可以处理的数据长度
            const std::size_t chunkLen = chunkSize / rowLen;

            std::vector<std::size_t> chunkVec = Awg::cutArrayMax(data.size(),chunkLen);
            auto task = [](Awg::DT* beg,Awg::DT* end)->std::string
            {
                std::string ret;
                ret.reserve((end-beg)*rowLen);

                while (beg < end)
                {
                    std::string s = fmt::to_string(*beg);
                    ++beg;
                }
                return ret;
            };
            //先在各个线程中将数字转换为字符串,才能确定每个数据块写入文件时的起始位置
        }
        else
        {
            std::cout<<"sssssssssssss";
        }
    }
    else
    {
        std::cout<<"ddddddddddddd";
    }
}

void Awg::stroeTxtFile(const QString& path)
{

}

AwgShortArray Awg::loadBinFile(const QString &path)
{
    QFile file(path);
    if(file.size() == 0  ||  file.size() > Awg::getFreeMemoryWindows()*0.9)
    {
        std::cout<<"file is empty or no enough memory for load"<<std::endl<<std::flush;
        return AwgShortArray{};
    }

    if(file.open(QIODevice::ReadOnly))
    {
        ThreadPool* pool = Awg::globalThreadPool();

        std::vector<std::size_t> chunkSizes = Awg::cutBinaryFile(file.size(),Awg::MinFileChunk,sizeof (Awg::DT));
        unsigned taskNum = chunkSizes.size();

        if(taskNum == 0)
        {
            std::cout<<"cut file failed"<<std::endl<<std::flush;
            return AwgShortArray{};
        }
        else
        {
            std::vector<std::future<AwgShortArray>> futures;
            futures.reserve(taskNum);

            std::size_t mapOffset = 0;
            for(unsigned i = 0; i < taskNum; i++)
            {
                std::future<AwgShortArray> f = pool->run<ThreadPool::Ordered>(&Awg::processBinFile,&file,mapOffset,chunkSizes[i]);
                futures.push_back(std::move(f));
                mapOffset += chunkSizes[i];
            }
            pool->waitforDone();

            //将各个线程的运算结果汇总
            std::vector<AwgShortArray> vec;
            vec.reserve(futures.size());
            for(std::future<AwgShortArray>& f : futures)
            {
                vec.push_back(f.get());
            }
            AwgShortArray result = AwgShortArray::combine(vec);
            return result;
        }
    }

    return AwgShortArray{};
}

AwgShortArray Awg::loadCsvFile(const QString &path)
{
    QFile file(path);
    if(file.size() == 0  ||  file.size() > Awg::getFreeMemoryWindows()*0.9)
    {
        std::cout<<"file is empty or no enough memory for load"<<std::endl<<std::flush;
        return AwgShortArray{};
    }

    if(file.open(QIODevice::ReadOnly))
    {
        ThreadPool* pool = Awg::globalThreadPool();
        const std::vector<char> spliters{'\n',','};

        std::vector<std::size_t> chunkSizes = Awg::cutTextFile(file,Awg::MinFileChunk,spliters);
        unsigned taskNum = chunkSizes.size();

        if(taskNum == 0)
        {
            std::cout<<"cut file failed"<<std::endl<<std::flush;
            return AwgShortArray{};
        }
        else
        {
            std::vector<std::future<AwgShortArray>> futures;
            futures.reserve(taskNum);

            std::size_t mapOffset = 0;
            for(unsigned i = 0; i < taskNum; i++)
            {
                std::future<AwgShortArray> f = pool->run<ThreadPool::Ordered>(&Awg::processTextFile,&file,mapOffset,chunkSizes[i],spliters);
                futures.push_back(std::move(f));
                mapOffset += chunkSizes[i];
            }

            pool->waitforDone();

            //将各个线程的运算结果汇总
            std::vector<AwgShortArray> vec;
            vec.reserve(futures.size());
            for(std::future<AwgShortArray>& f : futures)
            {
                vec.push_back(f.get());
            }
            AwgShortArray result = AwgShortArray::combine(vec);
            return result;
        }
    }

    return AwgShortArray{};
}

AwgShortArray Awg::loadTxtFile(const QString &path)
{
    QFile file(path);
    if(file.size() == 0  ||  file.size() > Awg::getFreeMemoryWindows()*0.9)
    {
        std::cout<<"file is empty or no enough memory for load"<<std::endl<<std::flush;
        return AwgShortArray{};
    }

    if(file.open(QIODevice::ReadOnly))
    {
        ThreadPool* pool = Awg::globalThreadPool();
        const std::vector<char> spliters{'\n'};

        std::vector<std::size_t> chunkSizes = Awg::cutTextFile(file,Awg::MinFileChunk,spliters);
        unsigned taskNum = chunkSizes.size();
        if(taskNum == 0)
        {
            std::cout<<"cut file failed"<<std::endl<<std::flush;
            return AwgShortArray{};
        }
        else
        {
            std::vector<std::future<AwgShortArray>> futures;
            futures.reserve(taskNum);

            std::size_t mapOffset = 0;
            for(unsigned i = 0; i < taskNum; i++)
            {
                std::future<AwgShortArray> f = pool->run<ThreadPool::Ordered>(&Awg::processTextFile,&file,mapOffset,chunkSizes[i],spliters);
                futures.push_back(std::move(f));
                mapOffset += chunkSizes[i];
            }

            pool->waitforDone();

            //将各个线程的运算结果汇总
            std::vector<AwgShortArray> vec;
            vec.reserve(futures.size());
            for(std::future<AwgShortArray>& f : futures)
            {
                vec.push_back(f.get());
            }
            AwgShortArray result = AwgShortArray::combine(vec);
            return result;
        }
    }

    return AwgShortArray{};
}

AwgShortArray Awg::processBinFile(QFile *file, std::size_t mapStart, std::size_t mapSize)
{
    FileMutex.lock();
    unsigned char* buf = file->map(mapStart,mapSize);
    FileMutex.unlock();

    if(buf == nullptr)
    {
        std::cout<<"map failed from "<<file->fileName().toStdString()<<" at chunk:"<<mapStart<<std::endl<<std::flush;
        return AwgShortArray{};
    }

    //直接将映射的内存拷贝到目标数组中,这里的mapSize是经过Awg::cutBinaryFile处理的,所以一定能整除sizeof (Awg::DT)
    const std::size_t arrayLeng = mapSize/sizeof (Awg::DT);
    AwgShortArray array(arrayLeng);
    memcpy(array,buf,mapSize);

    FileMutex.lock();
    while (!file->unmap(buf))
    {
        std::cout<<"unmap failed from "<<file->fileName().toStdString()<<" at chunk:"<<mapStart<<std::endl<<std::flush;
    }
    FileMutex.unlock();

    return array;
}

AwgShortArray Awg::processTextFile(QFile *file, std::size_t mapStart, std::size_t mapSize, const std::vector<char> &spliters)
{
    FileMutex.lock();
    unsigned char* buf = file->map(mapStart,mapSize);
    FileMutex.unlock();

    if(buf == nullptr)
    {
        std::cout<<"map failed from "<<file->fileName().toStdString()<<" at chunk:"<<mapStart<<std::endl<<std::flush;
        return AwgShortArray{};
    }

    const char* data = reinterpret_cast<const char*>(buf);
    const char* start = reinterpret_cast<const char*>(data);
    const char* end = start + mapSize;

    std::size_t index = 0;
    std::size_t arraySize = 1; //vector的容量比找到的分隔符数量多一个
    for(std::size_t i = 0; i < spliters.size(); i++)
    {
        arraySize += Awg::countCharAvx2(data,mapSize,spliters[i]);
    }

    //如果文本是以分隔符结尾或者开头,则让数组长度减1,因为开头的分隔符前面没有数据,结尾的分隔符后面也没有数据,这样可以准确地确定数组长度
    for(std::size_t i = 0; i < spliters.size(); i++)
    {
        if( (*start) == spliters[i] )
            --arraySize;

        if( (*(end-1)) == spliters[i] )
            --arraySize;
    }
    AwgShortArray array(arraySize);

    while (start < end)
    {
        //这里手动跳过非数字字符,虽然from_chars也可以自动跳过,但是影响效率
        if( !Awg::isIntegerBegin(*start) )
        {
            ++start;
            continue;
        }

        auto ret = fast_float::from_chars(start, end, array[index]);
        if(ret.ec == std::errc())
        {
            ++index;
        }
        //出现错误时不打印信息,跳过即可,当错误信息较多时会严重影响读取效率
//        else if(ret.ec == std::errc::invalid_argument)
//        {
//            std::cerr << "Error: invalid argument"<<std::endl<<std::flush;
//        }
        else if(ret.ec == std::errc::result_out_of_range)
        {
            // 根据值的大小决定使用最大值还是最小值
            if(*start == '-')
                array[index] = -std::numeric_limits<Awg::DT>::max();
            else
                array[index] = std::numeric_limits<Awg::DT>::max();
            ++index;
        }
//        else
//        {
//            std::cout<<"fast_float error:"<<int(ret.ec)<<std::endl<<std::flush;
//        }

        start = (ret.ptr == start) ? start+1 : ret.ptr;//更新指针位置
    }

    //这一部分文件读取完成之后解除映射释放内存,节省出来的内存可以用于创建新数组拷贝读取结果(如果需要地话)
    FileMutex.lock();
    while (!file->unmap(buf))
    {
        std::cout<<"unmap failed from "<<file->fileName().toStdString()<<" at chunk:"<<mapStart<<std::endl<<std::flush;
    }
    FileMutex.unlock();

    return array;
}
