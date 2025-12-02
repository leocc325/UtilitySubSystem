#include "AwgUtility.h"
#include <windows.h>
#include "UtilitySubSystem/ThreadPool.hpp"

ThreadPool *Awg::globalThreadPool() noexcept
{
    static ThreadPool pool(PoolSize);
    return &pool;
}

unsigned long long Awg::getFreeMemoryWindows()
{
    MEMORYSTATUSEX memoryStatus;
    memoryStatus.dwLength = sizeof(memoryStatus);

    if (GlobalMemoryStatusEx(&memoryStatus))
    {
        return static_cast<std::size_t>(memoryStatus.ullAvailPhys);
    }

    return -1; // 错误
}
