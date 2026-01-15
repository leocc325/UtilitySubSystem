#include "AwgFileIO.h"
#include <QFile>
#include <QCoreApplication>

#include "UtilitySubSystem/AwgFileIOprivate.hpp"
#include "UtilitySubSystem/AwgUtility.h"
#include "UtilitySubSystem/AwgAlgorithm.h"
#include "UtilitySubSystem/ThreadPool.hpp"
#include "UtilitySubSystem/fastfloat/fast_float.h"

std::mutex FileMutex;
#include <QDebug>
#include <QElapsedTimer>
template<Awg::FileFormat FT>
void storeTextFile(const QString &path, const double *array, const std::size_t arrayLength)
{
    //判断当前内存大小需要几次才能完成文件保存和每一次能处理的点数
    const double freeMem = Awg::getFreeMemoryWindows() * 0.9;
    //文件保存过程中需要映射文件占用内存,所以依然需要按数值的预定长度估算每一轮能处理的最大数据长度
    constexpr int RowLength = Awg::ArithmeticLengthSum<double>::value;
    //线程池每一轮计算能处理的最大数据长度,这里向上或者向下取整都可以,内存有余量
    std::size_t maxCountPerLoop = freeMem / RowLength;
    //根据实际数据长度和每一轮能处理的数据最大长度判断实际上能处理的数据长度
    maxCountPerLoop = std::min(maxCountPerLoop,arrayLength);
    //根据每一轮实际能处理的数据长度计算每个线程需要处理的数据长度
    std::size_t maxCountPerTask = std::max( std::size_t(Awg::MinArrayLength) , std::size_t(maxCountPerLoop/Awg::PoolSize));
    //根据每个线程任务能处理的数据长度确定需要划分多少个任务以及每个任务处理的数据长度,这里无需担心任务过多内存溢出,线程池执行完一轮任务才开始下一轮,所以是安全的
    std::vector<std::size_t> taskCountVec = Awg::splitLengthMax(arrayLength,maxCountPerTask);

    QElapsedTimer timer;
    //开始计算文件总大小和每一个任务的映射长度[需要注意顺序]
    ThreadPool* pool =  Awg::globalThreadPool();
    std::vector<std::future<std::size_t>> futures(taskCountVec.size());
    std::size_t offset = 0;
    timer.restart();
    for(std::size_t i = 0; i < taskCountVec.size(); i++)
    {
        switch (FT)
        {
        case Awg::FmtTxt:futures[i] = pool->run(Awg::calculateTextLenght<Awg::Txt,double>,taskCountVec[i],array+offset);break;
        case Awg::FmtCsv:futures[i] = pool->run(Awg::calculateTextLenght<Awg::Csv,double>,taskCountVec[i],array+offset);break;
        }
        offset += taskCountVec[i];
    }
    pool->waitforDone();
    qDebug()<<"cal size:"<<timer.elapsed();

    //线程池计算映射大小任务执行完毕之后开始计算文件总大小
    std::size_t fileSize = 0;
    std::vector<std::size_t> mapSizeVec(futures.size());
    for(std::size_t i = 0; i < futures.size(); i++)
    {
        mapSizeVec[i] = futures[i].get();
        fileSize += mapSizeVec[i];
    }
    timer.restart();
    QFile file(path);
    if(file.resize(fileSize))
    {
        qDebug()<<"resize:"<<timer.elapsed();
        if(file.open(QIODevice::ReadWrite))
        {
            timer.restart();
            emit AWGSIG->sigFileProcessMax(arrayLength + arrayLength*0.02);//这里将最大进度增加2%,因为数据处理完毕之后拷贝到文件的映射中还需要一点时间
            //文件resize和打开之后计算线程池需要执行多少轮任务
            std::size_t fileOffset = 0;
            auto mapSizeVecBeg = mapSizeVec.cbegin();
            auto mapSizeVecEnd = mapSizeVec.cend();
            auto taskCountVecBeg = taskCountVec.cbegin();
            const double* arrayBeg = array;
            int loops = std::ceil(double(mapSizeVec.size()) / Awg::PoolSize);
            for(int loop = 0; loop < loops; loop++)
            {
                std::size_t mapSize = 0;
                auto mapSizeVecL = mapSizeVecBeg;
                auto mapSizeVecR = mapSizeVecBeg + Awg::PoolSize;
                //获取本轮任务的最后一个数组指针,避免数组越界
                mapSizeVecR = std::min(mapSizeVecR,mapSizeVecEnd);
                //计算映射大小
                while (mapSizeVecL < mapSizeVecR)
                {
                    mapSize += (*mapSizeVecL);
                    ++mapSizeVecL;
                }

                char* mapped = reinterpret_cast<char*>(file.map(fileOffset,mapSize));
                char* bufStart = mapped;
                while(mapSizeVecBeg < mapSizeVecR)
                {
                    switch (FT)
                    {
                    case Awg::FmtCsv: pool->run(&Awg::toBinaryCsv<double>,bufStart,(*taskCountVecBeg),arrayBeg);break;
                    case Awg::FmtTxt: pool->run(&Awg::toBinaryTxt<double>,bufStart,(*taskCountVecBeg),arrayBeg);break;
                    }
                    fileOffset += (*mapSizeVecBeg);
                    bufStart += (*mapSizeVecBeg);
                    arrayBeg += (*taskCountVecBeg);

                    ++mapSizeVecBeg;
                    ++taskCountVecBeg;
                }
                pool->waitforDone();
                file.unmap(reinterpret_cast<unsigned char*>(mapped));
            }
            file.close();
            qDebug()<<"output:"<<timer.elapsed();
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

void Awg::storeBinFile(const QString &path, const double *array, const std::size_t length)
{
    if(array == nullptr || length == 0)
    {
        emit AWGSIG->sigWarningMessage(QCoreApplication::translate("Awg","目标数据为空,无法保存"));
        return;
    }

    QFile file(path);
    const std::size_t totalSize = sizeof (double) * length;
    if(file.resize(totalSize))
    {
        unsigned char* buf = file.map(0,totalSize);
        if(buf)
            memcpy(buf,array,totalSize);
        file.unmap(buf);
    }
}

void Awg::storeCsvFile(const QString &path, const double *array, const std::size_t length)
{
    if(array == nullptr || length == 0)
    {
        emit AWGSIG->sigWarningMessage(QCoreApplication::translate("Awg","目标数据为空,无法保存"));
        return;
    }

    storeTextFile<Awg::FmtCsv>(path,array,length);
}

void Awg::storeTxtFile(const QString &path, const double *array, const std::size_t length)
{
    if(array == nullptr || length == 0)
    {
        emit AWGSIG->sigWarningMessage(QCoreApplication::translate("Awg","目标数据为空,无法保存"));
        return;
    }

    storeTextFile<Awg::FmtTxt>(path,array,length);
}

AwgDoubleArray Awg::loadBinFile(const QString &path)
{
    QFile file(path);
    if(file.size() == 0  ||  file.size() > Awg::getFreeMemoryWindows()*0.9)
    {
        emit AWGSIG->sigWarningMessage(QCoreApplication::translate("Awg","文件%1为空或者内存不足无法加载").arg(file.fileName()));
        return AwgDoubleArray{};
    }

    if(file.open(QIODevice::ReadOnly))
    {
        ThreadPool* pool = Awg::globalThreadPool();

        std::vector<std::size_t> chunkSizes = Awg::cutBinaryFile(file.size(),Awg::MinFileChunk,sizeof (double));
        unsigned taskNum = chunkSizes.size();

        if(taskNum == 0)
        {
            emit AWGSIG->sigWarningMessage(QCoreApplication::translate("Awg","文件%1拆分失败").arg(file.fileName()));
            return AwgDoubleArray{};
        }
        else
        {
            std::vector<std::future<AwgDoubleArray>> futures;
            futures.reserve(taskNum);

            emit AWGSIG->sigFileProcessMax(file.size());
            std::size_t mapOffset = 0;
            for(unsigned i = 0; i < taskNum; i++)
            {
                std::future<AwgDoubleArray> f = pool->run<ThreadPool::Ordered>(&Awg::processBinFile,&file,mapOffset,chunkSizes[i]);
                futures.push_back(std::move(f));
                mapOffset += chunkSizes[i];
            }
            pool->waitforDone();

            //将各个线程的运算结果汇总
            std::vector<AwgDoubleArray> vec;
            vec.reserve(futures.size());
            for(std::future<AwgDoubleArray>& f : futures)
            {
                vec.push_back(f.get());
            }
            AwgDoubleArray result = AwgDoubleArray::combine(vec);
            return result;
        }
    }

    return AwgDoubleArray{};
}

AwgDoubleArray Awg::loadCsvFile(const QString &path)
{
    QFile file(path);
    if(file.size() == 0  ||  file.size() > Awg::getFreeMemoryWindows()*0.9)
    {
        emit AWGSIG->sigWarningMessage(QCoreApplication::translate("Awg","文件%1为空或者内存不足无法加载").arg(file.fileName()));
        return AwgDoubleArray{};
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
            return AwgDoubleArray{};
        }
        else
        {
            std::vector<std::future<AwgDoubleArray>> futures;
            futures.reserve(taskNum);

            emit AWGSIG->sigFileProcessMax(file.size());
            std::size_t mapOffset = 0;
            for(unsigned i = 0; i < taskNum; i++)
            {
                std::future<AwgDoubleArray> f = pool->run<ThreadPool::Ordered>(&Awg::processTextFile,&file,mapOffset,chunkSizes[i],spliters);
                futures.push_back(std::move(f));
                mapOffset += chunkSizes[i];
            }

            pool->waitforDone();

            //将各个线程的运算结果汇总
            std::vector<AwgDoubleArray> vec;
            vec.reserve(futures.size());
            for(std::future<AwgDoubleArray>& f : futures)
            {
                vec.push_back(f.get());
            }
            AwgDoubleArray result = AwgDoubleArray::combine(vec);
            return result;
        }
    }

    return AwgDoubleArray{};
}

AwgDoubleArray Awg::loadTxtFile(const QString &path)
{
    QFile file(path);
    if(file.size() == 0  ||  file.size() > Awg::getFreeMemoryWindows()*0.9)
    {
        emit AWGSIG->sigWarningMessage(QCoreApplication::translate("Awg","文件%1为空或者内存不足无法加载").arg(file.fileName()));
        return AwgDoubleArray{};
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
            return AwgDoubleArray{};
        }
        else
        {
            std::vector<std::future<AwgDoubleArray>> futures;
            futures.reserve(taskNum);

            emit AWGSIG->sigFileProcessMax(file.size());
            std::size_t mapOffset = 0;
            for(unsigned i = 0; i < taskNum; i++)
            {
                std::future<AwgDoubleArray> f = pool->run<ThreadPool::Ordered>(&Awg::processTextFile,&file,mapOffset,chunkSizes[i],spliters);
                futures.push_back(std::move(f));
                mapOffset += chunkSizes[i];
            }

            pool->waitforDone();

            //将各个线程的运算结果汇总
            std::vector<AwgDoubleArray> vec;
            vec.reserve(futures.size());
            for(std::future<AwgDoubleArray>& f : futures)
            {
                vec.push_back(f.get());
            }
            AwgDoubleArray result = AwgDoubleArray::combine(vec);
            return result;
        }
    }

    return AwgDoubleArray{};
}

AwgDoubleArray Awg::processBinFile(QFile *file, std::size_t mapStart, std::size_t mapSize)
{
    FileMutex.lock();
    unsigned char* buf = file->map(mapStart,mapSize);
    FileMutex.unlock();

    if(buf == nullptr)
    {
        emit AWGSIG->sigWarningMessage(QCoreApplication::translate("Awg","文件%1映射失败,在分块%2").arg(file->fileName()).arg(mapStart));
        return AwgDoubleArray{};
    }

    //直接将映射的内存拷贝到目标数组中,这里的mapSize是经过Awg::cutBinaryFile处理的,所以一定能整除sizeof (double)
    const std::size_t arrayLeng = mapSize/sizeof (double);
    AwgDoubleArray array(arrayLeng);
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

AwgDoubleArray Awg::processTextFile(QFile *file, std::size_t mapStart, std::size_t mapSize, const std::vector<char> &spliters)
{
    FileMutex.lock();
    unsigned char* buf = file->map(mapStart,mapSize);
    FileMutex.unlock();

    if(buf == nullptr)
    {
        emit AWGSIG->sigWarningMessage(QCoreApplication::translate("Awg","文件%1映射失败,在分块%2").arg(file->fileName()).arg(mapStart));
        return AwgDoubleArray{};
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
    AwgDoubleArray array(arraySize);

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
                array[index] = -std::numeric_limits<double>::max();
            else
                array[index] = std::numeric_limits<double>::max();
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
