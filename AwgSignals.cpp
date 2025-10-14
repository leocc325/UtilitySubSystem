#include "AwgSignals.h"

AwgSignals* AwgSignals::instance = nullptr;
AwgSignals *AwgSignals::getInstance()
{
    if(instance == nullptr)
        instance = new AwgSignals();
    return instance;
}

AwgSignals::AwgSignals()
{

}
