/*
 *   binary neurons network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#ifndef DEVICEAI_H
#define DEVICEAI_H

#include <memory>
//#include <QEvent>
//#include <QKeyEvent>
#include <QWebEngineView>

#include "brain_tools_web.h"

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
    void FillBinary(QPixmap&, bnn::architecture&);
};

class DeviceAI
{
public:
    bool* stepOld = nullptr;
    u_word stepOld_count = 0;
    void Go();
    QWebEngineView* qwev;
    std::unique_ptr<SensorPixmap> sensorPixmap;
    std::unique_ptr<bnn::brain_tools_web> brain_;
    ~DeviceAI();
    DeviceAI() = delete;
    DeviceAI(u_word motorCount,
             u_word brainBits,
             QSize qSize,
             QSize qSizeBig,
             QWebEngineView* qwev_);
    SensorPixmap& GetSensorPixmap(){ return *sensorPixmap; }
    bnn::architecture& GetBrain(){ return *brain_.get(); }
};

#endif // DEVICEAI_H
