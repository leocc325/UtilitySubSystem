#ifndef AWGSIGNALS_H
#define AWGSIGNALS_H

#include <QObject>

#define AWGSIG AwgSignals::getInstance()

class AwgSignals:public QObject
{
    Q_OBJECT
public:
    static AwgSignals* getInstance();

signals:
    void sigWarningMessage(QString msg);

    void sigSaveWavePixmap(int chanIndex,const QString& fileName);

    void sigSaveFile(int chanIndex,const QString& fileName);

    void sigLoadFile(int chanIndex,const QString& fileName);

    void sigProcessMax(unsigned long long size);

    void sigProcess(unsigned long long size);

    void sigFileProcessMax(unsigned long long size);

    void sigFileProcess(unsigned long long size);

    void sigFpgaWaveSendStart();

    void sigFpgaWaveSendStop();

private:
    AwgSignals();

private:
    static AwgSignals* instance;
};

#endif // AWGSIGNALS_H
