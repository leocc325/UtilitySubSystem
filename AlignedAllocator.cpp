#include "AlignedAllocator.h"
#include <stdlib.h>

void *Awg::alignedMalloc(std::size_t alignment, std::size_t bufSize) noexcept
{
    if (alignment == 0 || (alignment & (alignment - 1)) != 0)
        return nullptr;

    std::size_t totalSize = bufSize + alignment + sizeof (void*) - 1;
    void* rawPtr = std::malloc(totalSize);
    if(rawPtr == nullptr)
        return nullptr;

    uintptr_t rawAddress = reinterpret_cast<uintptr_t>(rawPtr);
    uintptr_t alignedAddress = (rawAddress + sizeof(void*) + alignment - 1) & ~(alignment - 1);

    // 在对齐地址之前存储原始指针
    void** storeRawPtr = reinterpret_cast<void**>(alignedAddress - sizeof(void*));
    *storeRawPtr = rawPtr;
    return reinterpret_cast<void*>(alignedAddress);
}

void Awg::alignedFree(void *ptr) noexcept
{
    if (ptr)
    {
        void** storeRawPtr = reinterpret_cast<void**>(reinterpret_cast<uintptr_t>(ptr) - sizeof(void*));
        void* rawPtr = *storeRawPtr;
        std::free(rawPtr);
    }
}

bool Awg::alignedCheck(const void *ptr,const std::size_t alignSize) noexcept
{
    return !(reinterpret_cast<uintptr_t>(ptr) % alignSize);
}
