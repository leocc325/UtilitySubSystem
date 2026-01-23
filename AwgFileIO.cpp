#include "AwgFileIO.h"
#include <QFile>
#include <QCoreApplication>

#include "UtilitySubSystem/AwgFileIOprivate.hpp"
#include "UtilitySubSystem/AwgUtility.h"
#include "UtilitySubSystem/AwgAlgorithm.old.h"
#include "UtilitySubSystem/ThreadPool.hpp"
#include "UtilitySubSystem/fastfloat/fast_float.h"

std::mutex FileMutex;

template<Awg::FileFormat FT>
void storeTextFile(const QString &path, const Awg::DT *array, const std::size_t arrayLength)
{
    //首先计算文件和剩余内存的大小
    const std::size_t rowLength = Awg::ArithmeticLengthSum<Awg::DT>::value;
    const std::size_t fileSize = rowLength * arrayLength;
    const std::size_t freeMem = Awg::getFreeMemoryWindows() * 0.9;
    //文件最大映射长度以
    const std::size_t maxMapSize = std::min(fileSize,freeMem);
    //根据最大映射长度计算线程池每一次循环能处理的最大点数
    const std::size_t maxLengthPerLoop = std::ceil(double(maxMapSize) / rowLength);
    //每一个线程任务能分配到的点数,但是这个点数不能小于每个线程能处理点数的最小值
    const std::size_t maxLengthPerTask = std::max(std::ceil(double(maxLengthPerLoop)/Awg::PoolSize),std::ceil(Awg::MinArrayLength));
    //根据每个线程能处理的最大点数划分线程任务
    std::vector<std::size_t> taskCountVec = Awg::splitLengthMax(arrayLength,maxLengthPerTask);

    QFile file(path);
    if(file.resize(fileSize))
    {
        if(file.open(QIODevice::ReadWrite))
        {
            emit AWGSIG->sigFileProcessMax(arrayLength + arrayLength*0.02);//这里将最大进度增加2%,因为数据处理完毕之后解除映射可能还需要一点时间

            std::size_t fileOffset = 0;
            auto taskCountVecBeg = taskCountVec.cbegin();
            auto taskCountVecEnd = taskCountVec.cend();

            ThreadPool* pool =  Awg::globalThreadPool();
            while (taskCountVecBeg < taskCountVecEnd)
            {
                //每一轮获取线程池最大容量的任务数量,防止数组越界
                auto taskCountVecL = taskCountVecBeg;
                auto taskCountVecR = std::min(taskCountVecBeg + Awg::PoolSize,taskCountVecEnd);

                //计算这一轮任务需要映射的长度和数组长度
                std::size_t mapSize = 0;
                while (taskCountVecL < taskCountVecR)
                {
                    mapSize += (*taskCountVecL) * rowLength;
                    ++taskCountVecL;
                }
                //将起始迭代器位置归位
                taskCountVecL = taskCountVecBeg;

                unsigned char* map = file.map(fileOffset,mapSize);
                char* buf = reinterpret_cast<char*>(map);
                while (taskCountVecL < taskCountVecR)
                {
                    pool->run(&Awg::toBinaryTxtFixedWidth<Awg::DT>,Awg::TextFormat(FT),buf,(*taskCountVecL),array);
                    buf += (*taskCountVecL) * rowLength;
                    array += (*taskCountVecL);
                    ++taskCountVecL;
                }
                pool->waitforDone();
                file.unmap(map);

                //更新迭代器位置和映射数据等信息
                taskCountVecBeg = taskCountVecR;
                fileOffset += mapSize;
            }
            file.close();
            emit AWGSIG->sigFileProcess(arrayLength*0.02);//发送最后一点进度
        }
        else
        {
            emit AWGSIG->sigWarningMessage(QCoreApplication::translate("Awg","文件%1打开失败").arg(file.fileName()));
        }
    }
    else
    {
        emit AWGSIG->sigWarningMessage(QCoreApplication::translate("Awg","文件%1大小重置失败").arg(file.fileName()));
    }
}

void Awg::storeBinFile(const QString &path, const Awg::DT *array, const std::size_t length)
{
    if(array == nullptr || length == 0)
    {
        emit AWGSIG->sigWarningMessage(QCoreApplication::translate("Awg","目标数据为空,无法保存"));
        return;
    }

    QFile file(path);
    const std::size_t totalSize = sizeof (Awg::DT) * length;
    if(file.resize(totalSize))
    {
        unsigned char* buf = file.map(0,totalSize);
        if(buf)
            memcpy(buf,array,totalSize);
        file.unmap(buf);
    }
}

void Awg::storeCsvFile(const QString &path, const Awg::DT *array, const std::size_t length)
{
    if(array == nullptr || length == 0)
    {
        emit AWGSIG->sigWarningMessage(QCoreApplication::translate("Awg","目标数据为空,无法保存"));
        return;
    }

    storeTextFile<Awg::FmtCsv>(path,array,length);
}

void Awg::storeTxtFile(const QString &path, const Awg::DT *array, const std::size_t length)
{
    if(array == nullptr || length == 0)
    {
        emit AWGSIG->sigWarningMessage(QCoreApplication::translate("Awg","目标数据为空,无法保存"));
        return;
    }

    storeTextFile<Awg::FmtTxt>(path,array,length);
}

AwgFloatArray Awg::loadBinFile(const QString &path)
{
    QFile file(path);
    if(file.size() == 0  ||  file.size() > Awg::getFreeMemoryWindows()*0.9)
    {
        emit AWGSIG->sigWarningMessage(QCoreApplication::translate("Awg","文件%1为空或者内存不足无法加载").arg(file.fileName()));
        return AwgFloatArray{};
    }

    if(file.open(QIODevice::ReadOnly))
    {
        ThreadPool* pool = Awg::globalThreadPool();

        std::vector<std::size_t> chunkSizes = Awg::cutBinaryFile(file.size(),Awg::MinFileChunk,sizeof (double));
        unsigned taskNum = chunkSizes.size();

        if(taskNum == 0)
        {
            emit AWGSIG->sigWarningMessage(QCoreApplication::translate("Awg","文件%1拆分失败").arg(file.fileName()));
            return AwgFloatArray{};
        }
        else
        {
            std::vector<std::future<AwgFloatArray>> futures;
            futures.reserve(taskNum);

            emit AWGSIG->sigFileProcessMax(file.size());
            std::size_t mapOffset = 0;
            for(unsigned i = 0; i < taskNum; i++)
            {
                std::future<AwgFloatArray> f = pool->run<ThreadPool::Ordered>(&Awg::processBinFile,&file,mapOffset,chunkSizes[i]);
                futures.push_back(std::move(f));
                mapOffset += chunkSizes[i];
            }
            pool->waitforDone();

            //将各个线程的运算结果汇总
            std::vector<AwgFloatArray> vec;
            vec.reserve(futures.size());
            for(std::future<AwgFloatArray>& f : futures)
            {
                vec.push_back(f.get());
            }
            AwgFloatArray result = AwgFloatArray::combine(vec);
            return result;
        }
    }

    return AwgFloatArray{};
}

AwgFloatArray Awg::loadCsvFile(const QString &path)
{
    QFile file(path);
    if(file.size() == 0  ||  file.size() > Awg::getFreeMemoryWindows()*0.9)
    {
        emit AWGSIG->sigWarningMessage(QCoreApplication::translate("Awg","文件%1为空或者内存不足无法加载").arg(file.fileName()));
        return AwgFloatArray{};
    }

    if(file.open(QIODevice::ReadOnly))
    {
        ThreadPool* pool = Awg::globalThreadPool();
        const std::vector<char> spliters{'\n',','};

        std::vector<std::size_t> chunkSizes = Awg::cutTextFile(file,Awg::MinFileChunk,spliters);
        unsigned taskNum = chunkSizes.size();

        if(taskNum == 0)
        {
            emit AWGSIG->sigWarningMessage(QCoreApplication::translate("Awg","文件%1拆分失败").arg(file.fileName()));
            return AwgFloatArray{};
        }
        else
        {
            std::vector<std::future<AwgFloatArray>> futures;
            futures.reserve(taskNum);

            emit AWGSIG->sigFileProcessMax(file.size());
            std::size_t mapOffset = 0;
            for(unsigned i = 0; i < taskNum; i++)
            {
                std::future<AwgFloatArray> f = pool->run<ThreadPool::Ordered>(&Awg::processTextFile,&file,mapOffset,chunkSizes[i],spliters);
                futures.push_back(std::move(f));
                mapOffset += chunkSizes[i];
            }

            pool->waitforDone();

            //将各个线程的运算结果汇总
            std::vector<AwgFloatArray> vec;
            vec.reserve(futures.size());
            for(std::future<AwgFloatArray>& f : futures)
            {
                vec.push_back(f.get());
            }
            AwgFloatArray result = AwgFloatArray::combine(vec);
            return result;
        }
    }

    return AwgFloatArray{};
}

AwgFloatArray Awg::loadTxtFile(const QString &path)
{
    QFile file(path);
    if(file.size() == 0  ||  file.size() > Awg::getFreeMemoryWindows()*0.9)
    {
        emit AWGSIG->sigWarningMessage(QCoreApplication::translate("Awg","文件%1为空或者内存不足无法加载").arg(file.fileName()));
        return AwgFloatArray{};
    }

    if(file.open(QIODevice::ReadOnly))
    {
        ThreadPool* pool = Awg::globalThreadPool();
        const std::vector<char> spliters{'\n'};

        std::vector<std::size_t> chunkSizes = Awg::cutTextFile(file,Awg::MinFileChunk,spliters);
        unsigned taskNum = chunkSizes.size();
        if(taskNum == 0)
        {
            emit AWGSIG->sigWarningMessage(QCoreApplication::translate("Awg","文件%1拆分失败").arg(file.fileName()));
            return AwgFloatArray{};
        }
        else
        {
            std::vector<std::future<AwgFloatArray>> futures;
            futures.reserve(taskNum);

            emit AWGSIG->sigFileProcessMax(file.size());
            std::size_t mapOffset = 0;
            for(unsigned i = 0; i < taskNum; i++)
            {
                std::future<AwgFloatArray> f = pool->run<ThreadPool::Ordered>(&Awg::processTextFile,&file,mapOffset,chunkSizes[i],spliters);
                futures.push_back(std::move(f));
                mapOffset += chunkSizes[i];
            }

            pool->waitforDone();

            //将各个线程的运算结果汇总
            std::vector<AwgFloatArray> vec;
            vec.reserve(futures.size());
            for(std::future<AwgFloatArray>& f : futures)
            {
                vec.push_back(f.get());
            }
            AwgFloatArray result = AwgFloatArray::combine(vec);
            return result;
        }
    }

    return AwgFloatArray{};
}

AwgFloatArray Awg::processBinFile(QFile *file, std::size_t mapStart, std::size_t mapSize)
{
    FileMutex.lock();
    unsigned char* buf = file->map(mapStart,mapSize);
    FileMutex.unlock();

    if(buf == nullptr)
    {
        emit AWGSIG->sigWarningMessage(QCoreApplication::translate("Awg","文件%1映射失败,在分块%2").arg(file->fileName()).arg(mapStart));
        return AwgFloatArray{};
    }

    //直接将映射的内存拷贝到目标数组中,这里的mapSize是经过Awg::cutBinaryFile处理的,所以一定能整除sizeof (float)
    const std::size_t arrayLeng = mapSize/sizeof (float);
    AwgFloatArray array(arrayLeng);
    memcpy(array,buf,mapSize);
    emit AWGSIG->sigFileProcess(mapSize);

    FileMutex.lock();
    while (!file->unmap(buf))
    {
        emit AWGSIG->sigWarningMessage(QCoreApplication::translate("Awg","文件%1映射解除失败,在分块%2").arg(file->fileName()).arg(mapStart));
    }
    FileMutex.unlock();

    return array;
}

AwgFloatArray Awg::processTextFile(QFile *file, std::size_t mapStart, std::size_t mapSize, const std::vector<char> &spliters)
{
    FileMutex.lock();
    unsigned char* buf = file->map(mapStart,mapSize);
    FileMutex.unlock();

    if(buf == nullptr)
    {
        emit AWGSIG->sigWarningMessage(QCoreApplication::translate("Awg","文件%1映射失败,在分块%2").arg(file->fileName()).arg(mapStart));
        return AwgFloatArray{};
    }

    const char* start = reinterpret_cast<const char*>(buf);
    const char* process = reinterpret_cast<const char*>(buf);
    const char* end = start + mapSize;

    std::size_t index = 0;
    std::size_t arraySize = 1; //vector的容量比找到的分隔符数量多一个
    for(std::size_t i = 0; i < spliters.size(); i++)
    {
        arraySize += Awg::countChar(start,end,spliters[i]);
    }

    //如果文本是以分隔符结尾或者开头,则让数组长度减1,因为开头的分隔符前面没有数据,结尾的分隔符后面也没有数据,这样可以准确地确定数组长度
    for(std::size_t i = 0; i < spliters.size(); i++)
    {
        if( (*start) == spliters[i] )
            --arraySize;

        if( (*(end-1)) == spliters[i] )
            --arraySize;
    }
    AwgFloatArray array(arraySize);

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
                array[index] = -std::numeric_limits<float>::min();
            else
                array[index] = std::numeric_limits<float>::max();
            ++index;
        }
//        else
//        {
//            std::cout<<"fast_float error:"<<int(ret.ec)<<std::endl<<std::flush;
//        }
        start = (ret.ptr == start) ? start+1 : ret.ptr;//更新指针位置
        if(start - process > 1e6)
        {
            emit AWGSIG->sigFileProcess(start - process);//每读取1M字节数据发送一次信号更新进度
            process = start;
        }
    }
    emit AWGSIG->sigFileProcess(start - process);//发送最后一部分数据进度

    //这一部分文件读取完成之后解除映射释放内存,节省出来的内存可以用于创建新数组拷贝读取结果(如果需要地话)
    FileMutex.lock();
    while (!file->unmap(buf))
    {
        emit AWGSIG->sigWarningMessage(QCoreApplication::translate("Awg","文件%1映射解除失败,在分块%2").arg(file->fileName()).arg(mapStart));
    }
    FileMutex.unlock();

    return array;
}
