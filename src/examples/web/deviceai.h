//*************************************************************//
//                                                             //
//   network of binary neurons                                 //
//   created by Ilya Shishkin                                  //
//   cortl@8iter.ru                                            //
//   http://8iter.ru/ai.html                                   //
//   https://github.com/cortl0/network_of_binary_neurons_cpp   //
//   licensed by GPL v3.0                                      //
//                                                             //
//*************************************************************//

#ifndef DEVICEAI_H
#define DEVICEAI_H

#include <memory>
//#include <QEvent>
//#include <QKeyEvent>
#include <QWebEngineView>

#include "brain_friend.h"

class SensorPixmap
{
    QPixmap qPixmapSmall;
    QSize qSizeBig;
    double zoom_max = 1.0;
    double zoom_min = 0.5;
    double zoom_koef = 1.03125;
    double deltaXY = 1;
    double epsilon = 256;
public:
    SensorPixmap() = delete;
    SensorPixmap(QSize qSize, QSize qSizeBig, int gradation_bit, bool black_white = false);
    bool black_white;
    int gradation_bit = 2;
    double x;
    double y;
    double zoom = zoom_max;
    void Zoom_in();
    void Zoom_out();
    void X_plus();
    void X_minus();
    void Y_plus();
    void Y_minus();
    QPixmap& GetPixmap(){ return qPixmapSmall; }
    void PixmapNormalize();
    void FillBinary(QPixmap &qPixmapWeb, brain &brn);
};

class DeviceAI
{
public:
    bool* stepOld = nullptr;
    _word stepOld_count = 0;
    void Go (brain &brn);
    QWebEngineView* qwev;
    std::unique_ptr<SensorPixmap> sensorPixmap;
    std::unique_ptr<brain> brn;
    std::unique_ptr<brain_friend> brain_friend_;
    ~DeviceAI();
    DeviceAI() = delete;
    DeviceAI(_word random_array_length_in_power_of_two,
             _word motorCount,
             _word brainBits,
             QSize qSize,
             QSize qSizeBig,
             void (*tick_web_engine)(),
             QWebEngineView* qwev_);
    SensorPixmap& GetSensorPixmap(){ return *sensorPixmap; }
    brain& GetBrain(){ return *brn.get(); }
};

#endif // !DEVICEAI_H
