#ifndef AWGUTILITY_H
#define AWGUTILITY_H

#ifndef AWGDEFINES_H
#include "AwgDefines.h"
#endif

class ThreadPool;
namespace Awg
{
    ///一个全局线程池,全局线程池主要用于并行计算和读写数据 !!!不要给这个线程池传入计算和读写以外的函数,否则可能会导致计算结果获取被延后!!!
    ThreadPool* globalThreadPool() noexcept;

    ///查询系统当前可用内存
    unsigned long long getFreeMemoryWindows();
}

#endif // AWGUTILITY_H
