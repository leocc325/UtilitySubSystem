#ifndef ALIGNEDALLOCATOR_H
#define ALIGNEDALLOCATOR_H

namespace Awg {
    ///分配一块N字节对齐,大小为Size字节的内存块
    void *alignedMalloc(unsigned long long alignment,unsigned long long bufSize) noexcept;

    ///释放由alignedMalloc分配的内存
    void alignedFree(void* ptr) noexcept;

    ///检查指针ptr是否是alignSize字节对齐的地址
    bool alignedCheck(const void* ptr,const unsigned long long alignSize) noexcept;
}


#endif // ALIGNEDALLOCATOR_H
