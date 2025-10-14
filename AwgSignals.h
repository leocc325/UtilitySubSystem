#ifndef AWGSIGNALS_H
#define AWGSIGNALS_H

#ifndef QOBJECT_H
#include <QObject>
#endif

#define AWGSIG AwgSignals::getInstance()

class AwgSignals:public QObject
{
    Q_OBJECT
public:
    static AwgSignals* getInstance();

signals:
    void sigProcessMax(std::size_t size);

    void sigProcess(std::size_t size);

private:
    AwgSignals();
private:
    static AwgSignals* instance;
};

#endif // AWGSIGNALS_H
