#include "AwgFileIO.h"
#include <QFile>
#include <QCoreApplication>

#include "UtilitySubSystem/AwgFileIOprivate.hpp"
#include "UtilitySubSystem/AwgUtility.h"
#include "UtilitySubSystem/AwgAlgorithm.h"
#include "UtilitySubSystem/ThreadPool.hpp"
#include "UtilitySubSystem/fastfloat/fast_float.h"

std::mutex FileMutex;

template<Awg::FileFormat FT>
void storeTextFile(const QString &path, const Awg::DT *array, const std::size_t length)
{
    QFile file(path);
    const int rowLength = (Awg::ArithmeticLengthSum<Awg::DT>::value + Awg::NewLine.size());
    ThreadPool* pool =  Awg::globalThreadPool();
    if(file.resize(rowLength * length))
    {
        if(file.open(QIODevice::ReadWrite))
        {
            //获取当前内存大小计算每一次可以转换为字符串的数据的总长度
            const std::size_t freeMem = Awg::getFreeMemoryWindows() * 0.9;
            const std::size_t lineLength = Awg::ArithmeticLengthSum<short>::value + Awg::NewLine.size();
            const std::size_t menLenth = freeMem / lineLength;

            //根据实际数据长度和最大处理长度确定每一轮计算能处理的实际长度
            const std::size_t maxStepLenth = std::min(menLenth,length);
            //根据每一轮计算能处理的最大数据长度计算每一个线程需要处理的数据长度,但是不让每个线程能处理的数据长度小于最小长度限制
            const std::size_t taskLen = std::max( std::size_t(Awg::MinArrayLength) , std::size_t(maxStepLenth/Awg::PoolSize)) ;
            //根据计算得到的每一个线程的数据长度划分线程任务
            std::vector<std::size_t> pointsVec = Awg::splitLengthMax(length,taskLen);
            //计算线程池每一轮能执行的任务数量
            const std::size_t taskNumOnce = std::min(unsigned(Awg::PoolSize),unsigned(pointsVec.size()));
            //根据每一轮能执行的任务数量创建相应的缓冲区保存转换结果
            std::vector<std::future<std::pair<char*,char*>>> futures;
            std::vector<std::unique_ptr<char[]>> buffers(taskNumOnce);//这里直接分配一整块再划分给各个buffer，多次分配效率很慢
            for(std::size_t i = 0; i < buffers.size(); i++)
                buffers.at(i) = std::make_unique<char[]>(rowLength * taskLen);

            std::size_t fileOffset = 0;
            std::size_t totalSize = 0;
            std::size_t dataIndex = 0;

            //和读取文本文件不同的是,每一个线程处理完数据之后写入到文件的位置是不确定的,所以不能直接让各个线程自行运行,必须等待前一轮任务处理完毕之后才能处理后一轮任务
            emit AWGSIG->sigFileProcessMax(length);
            for(std::size_t i = 0; i < pointsVec.size(); i++)
            {
                //每一轮任务的最大数量为线程池的大小,同时也确保任务索引不会超过总的数量
                for(std::size_t j = 0; j < taskNumOnce && i < pointsVec.size(); i++,j++)
                {
                    std::future<std::pair<char*,char*>> future;
                    switch (FT)
                    {
                        case Awg::FmtCsv: future =  pool->run(&Awg::toBinaryCsv<short>,buffers.at(i).get(),pointsVec.at(i),array+dataIndex);break;
                        case Awg::FmtTxt: future = pool->run(&Awg::toBinaryTxt<short>,buffers.at(i).get(),pointsVec.at(i),array+dataIndex);break;
                    }
                    futures.push_back(std::move(future));
                    dataIndex += pointsVec.at(i);
                }
                pool->waitforDone();

                //线程执行完毕之后提取线程执行结果,计算转换为字符串的总长度
                std::size_t outputSize = 0;
                std::vector<std::pair<char*,char*>> result;
                for(std::size_t k = 0; k < futures.size(); k++)
                {
                    result.push_back(futures.at(k).get());
                    outputSize += result.back().second - result.back().first;
                }

                //将文件从上一次写入的位置处映射,映射长度为这一次转换的长度
                unsigned char* buf = file.map(fileOffset,outputSize);
                std::size_t mapOffset = 0;
                for(std::size_t k = 0; k < result.size(); k++)
                {
                    char* start = result.at(k).first;
                    std::size_t mapSize = result.at(k).second - start;
                    memcpy(buf+mapOffset,start,mapSize);

                    //更新映射偏置和文件偏置
                    fileOffset += mapSize;
                    mapOffset += mapSize;
                    totalSize += mapSize;
                }
                file.unmap(buf);
                futures.clear();
            }
            file.close();
            file.resize(totalSize);
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

AwgShortArray Awg::loadBinFile(const QString &path)
{
    QFile file(path);
    if(file.size() == 0  ||  file.size() > Awg::getFreeMemoryWindows()*0.9)
    {
        emit AWGSIG->sigWarningMessage(QCoreApplication::translate("Awg","文件%1为空或者内存不足无法加载").arg(file.fileName()));
        return AwgShortArray{};
    }

    if(file.open(QIODevice::ReadOnly))
    {
        ThreadPool* pool = Awg::globalThreadPool();

        std::vector<std::size_t> chunkSizes = Awg::cutBinaryFile(file.size(),Awg::MinFileChunk,sizeof (Awg::DT));
        unsigned taskNum = chunkSizes.size();

        if(taskNum == 0)
        {
            emit AWGSIG->sigWarningMessage(QCoreApplication::translate("Awg","文件%1拆分失败").arg(file.fileName()));
            return AwgShortArray{};
        }
        else
        {
            std::vector<std::future<AwgShortArray>> futures;
            futures.reserve(taskNum);

            emit AWGSIG->sigFileProcessMax(file.size());
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
        emit AWGSIG->sigWarningMessage(QCoreApplication::translate("Awg","文件%1为空或者内存不足无法加载").arg(file.fileName()));
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
            emit AWGSIG->sigWarningMessage(QCoreApplication::translate("Awg","文件%1拆分失败").arg(file.fileName()));
            return AwgShortArray{};
        }
        else
        {
            std::vector<std::future<AwgShortArray>> futures;
            futures.reserve(taskNum);

            emit AWGSIG->sigFileProcessMax(file.size());
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
        emit AWGSIG->sigWarningMessage(QCoreApplication::translate("Awg","文件%1为空或者内存不足无法加载").arg(file.fileName()));
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
            emit AWGSIG->sigWarningMessage(QCoreApplication::translate("Awg","文件%1拆分失败").arg(file.fileName()));
            return AwgShortArray{};
        }
        else
        {
            std::vector<std::future<AwgShortArray>> futures;
            futures.reserve(taskNum);

            emit AWGSIG->sigFileProcessMax(file.size());
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
        emit AWGSIG->sigWarningMessage(QCoreApplication::translate("Awg","文件%1映射失败,在分块%2").arg(file->fileName()).arg(mapStart));
        return AwgShortArray{};
    }

    //直接将映射的内存拷贝到目标数组中,这里的mapSize是经过Awg::cutBinaryFile处理的,所以一定能整除sizeof (Awg::DT)
    const std::size_t arrayLeng = mapSize/sizeof (Awg::DT);
    AwgShortArray array(arrayLeng);
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

AwgShortArray Awg::processTextFile(QFile *file, std::size_t mapStart, std::size_t mapSize, const std::vector<char> &spliters)
{
    FileMutex.lock();
    unsigned char* buf = file->map(mapStart,mapSize);
    FileMutex.unlock();

    if(buf == nullptr)
    {
        emit AWGSIG->sigWarningMessage(QCoreApplication::translate("Awg","文件%1映射失败,在分块%2").arg(file->fileName()).arg(mapStart));
        return AwgShortArray{};
    }

    const char* start = reinterpret_cast<const char*>(buf);
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
        emit AWGSIG->sigWarningMessage(QCoreApplication::translate("Awg","文件%1映射解除失败,在分块%2").arg(file->fileName()).arg(mapStart));
    }
    FileMutex.unlock();

    emit AWGSIG->sigFileProcess(mapSize);
    return array;
}
