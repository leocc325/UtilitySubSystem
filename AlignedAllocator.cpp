#include "AlignedAllocator.h"
#include <stdlib.h>

void *Awg::alignedMalloc(unsigned long long alignment, unsigned long long bufSize) noexcept
{
    if (alignment == 0 || (alignment & (alignment - 1)) != 0)
        return nullptr;

    unsigned long long totalSize = bufSize + alignment + sizeof (void*) - 1;
    void* rawPtr = std::malloc(totalSize);
    if(rawPtr == nullptr)
        return nullptr;

    uintptr_t rawAddress = reinterpret_cast<uintptr_t>(rawPtr);
    uintptr_t alignedAddress = Awg::alignUp(rawAddress + sizeof(void*),alignment);/*(rawAddress + sizeof(void*) + alignment - 1) & ~(alignment - 1);*/

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

bool Awg::alignedCheck(const void *ptr,const unsigned long long alignSize) noexcept
{
    return !(reinterpret_cast<uintptr_t>(ptr) % alignSize);
}

unsigned long long Awg::alignUp(unsigned long long input, unsigned long long aligned)
{
    return (input + aligned - 1) & ~(aligned  - 1);
}
