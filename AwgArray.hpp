#ifndef AWGARRAY_H
#define AWGARRAY_H

#include "AwgDefines.h"
#include <vector>
#include <memory>
#include "UtilitySubSystem/AlignedAllocator.h"

///前置声明
template<typename T,std::size_t Align>
class AlignedSharedArray;

///AlignedSharedArray模板工具
namespace Awg {
    namespace  Asa //ASA = AlignedSharedArray
    {
        ///判断类型T是否是数值类型
        template<typename T>
        struct IsArithmetic
        {  constexpr static bool value = std::is_arithmetic<T>::value;  };

        ///判断Align是否是2的N次方
        template<std::size_t Align>
        struct IsAlignment
        { constexpr static bool value = (Align != 0 && (Align & (Align - 1)) == 0);  };

        template<typename T,std::size_t Align>
        constexpr bool  Illegal =IsAlignment<Align>::value && IsArithmetic<T>::value;

        template <typename T>
        struct IsAsa:public std::false_type{};

        template<typename T,std::size_t Align>
        struct IsAsa<AlignedSharedArray<T, Align>>:std::true_type{};

        template<typename T,typename U>
        struct SameAsa:public std::false_type{};

        template<typename T1, std::size_t Align1, typename T2, std::size_t Align2>
        struct SameAsa<AlignedSharedArray<T1, Align1>, AlignedSharedArray<T2, Align2>>
        { constexpr static bool value = std::is_same<T1,T2>::value && (Align1 == Align2); };

    }
}

template<typename T,std::size_t Align>
class AlignedSharedArray
{
    static_assert (Awg::Asa::Illegal<T,Align>,"Align size or Data type is Illagle" );

public:
    AlignedSharedArray(){}

    AlignedSharedArray(std::size_t length)
    {
        std::size_t totalSize = length * sizeof (T) + Align + sizeof (void*) - 1;
        T* ptr = reinterpret_cast<T*>(Awg::alignedMalloc(Align,totalSize));

        this->length = length;
        this->array = std::shared_ptr<T>(ptr,Awg::alignedFree) ;
    }

    AlignedSharedArray(const AlignedSharedArray& other)
    {
        this->copyImpl(other);
    }

    AlignedSharedArray(AlignedSharedArray&& other)
    {
        this->moveImpl(std::move(other));
    }

    AlignedSharedArray& operator = (const AlignedSharedArray& other)
    {
        this->copyImpl(other);
        return  *this;
    }

    AlignedSharedArray& operator = (AlignedSharedArray&& other)
    {
        this->moveImpl(std::move(other));
        return  *this;
    }

    T& operator [](std::size_t index)
    {
        if(index >= length)
            throw std::out_of_range("AlignedSharedArray out of range");

        return array.get()[index];
    }

    operator T* () const noexcept
    {
        return array.get();
    }

    bool empty() const noexcept
    {
        return (length == 0);
    }

    T* data() const noexcept
    {
        return array.get();
    }

    std::size_t size() const noexcept
    {
        return length;
    }

    AlignedSharedArray clone()
    {
        AlignedSharedArray ret(length);
        if(this->array != nullptr && ret.array != nullptr)
            memcpy(ret.array.get(),this->array.get(),sizeof(T)*length);
        return ret;
    }

    //确保U和当前数组类型一致,如果类型一致则将vector中的数组依次拼接到一起
    template<typename U>
    static typename std::enable_if<Awg::Asa::SameAsa<AlignedSharedArray<T,Align>,U>::value,U>::type
    combine(const std::vector<U>& vec)
    {
        //内存足够的情况下直接拼接
        //内存不足的情况下先写入文件再读取
        std::size_t totalLength = 0;
        std::size_t bufPos = 0;

        for(const U& ary : vec)
            totalLength += ary.size();

        AlignedSharedArray newAry(totalLength);
        char* data = reinterpret_cast<char*>(newAry.data());
        for(int i = 0; i < vec.size(); i++)
        {
            std::size_t bytes = vec[i].size() * sizeof (T);
            memcpy(data+bufPos,vec[i],bytes);
            bufPos += bytes;
        }

        return newAry;
    }

private:
    void copyImpl(const AlignedSharedArray& other)
    {
        if(this != &other)
        {
            this->length = other.length;
            this->array = other.array;
        }
    }

    void moveImpl(AlignedSharedArray&& other)
    {
        if (this != &other)
        {
            this->length = other.length;
            other.length = 0;
            this->array = std::move(other.array);
        }
    }

private:
    std::size_t length = 0;
    std::shared_ptr<T> array = nullptr;
};

using AwgCharArray = AlignedSharedArray<char,Awg::ArrayAlignment>;
using AwgShortArray = AlignedSharedArray<short,Awg::ArrayAlignment>;
using AwgIntArray = AlignedSharedArray<int,Awg::ArrayAlignment>;
using AwgDoubleArray = AlignedSharedArray<double,Awg::ArrayAlignment>;
using AwgFloatArray = AlignedSharedArray<float,Awg::ArrayAlignment>;


#endif // AWGARRAY_H
