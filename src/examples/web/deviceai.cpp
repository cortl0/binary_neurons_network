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

#include "deviceai.h"

SensorPixmap::SensorPixmap(QSize qSize, QSize qSizeBig_, int gradation_bit_, bool black_white_)
{
    qSizeBig = QSize(qSizeBig_.width(),qSizeBig_.height());
    qPixmapSmall = QPixmap(QSize(qSize.width(),qSize.height()));
    x = qSizeBig.width()/2;
    y = qSizeBig.height()/2;
    gradation_bit = gradation_bit_;
    black_white = black_white_;
}
void SensorPixmap::PixmapNormalize()
{
    if(zoom>zoom_max)
        zoom=zoom_max;
    if(zoom<zoom_min)
        zoom=zoom_min;
    if(x+qPixmapSmall.size().width()/zoom/2 + epsilon>=qSizeBig.width())
        x=qSizeBig.width()-qPixmapSmall.size().width()/zoom/2 - epsilon;
    if(x-qPixmapSmall.size().width()/zoom/2<0 + epsilon)
        x=qPixmapSmall.size().width()/zoom/2 + epsilon;
    if(y+qPixmapSmall.size().height()/zoom/2 + epsilon>=qSizeBig.height())
        y=qSizeBig.height()-qPixmapSmall.size().height()/zoom/2 - epsilon;
    if(y-qPixmapSmall.size().height()/zoom/2<0 + epsilon)
        y=qPixmapSmall.size().height()/zoom/2 + epsilon;
}
void SensorPixmap::FillBinary(QPixmap &qPixmapWeb, brain &brn)
{
    QPainter qPainter(&qPixmapSmall);
    QImage qImageWeb = qPixmapWeb.toImage();
    QImage qImage = (qPixmapSmall).toImage();
    int kgr = (256 >> gradation_bit);
    int d = kgr / 2 - 1;
    for (int j=0; j<qPixmapSmall.size().height(); j++)
        for (int i=0; i<qPixmapSmall.size().width(); i++)
        {
            QRgb qRgb1 = qImageWeb.pixel(
                        static_cast<int>(x + i / zoom - (qPixmapSmall.size().width() >> 1) / zoom),
                        static_cast<int>(y + j / zoom - (qPixmapSmall.size().height() >> 1) / zoom));
            int r=qRed(qRgb1);
            int g=qGreen(qRgb1);
            int b=qBlue(qRgb1);
            if (black_white)
            {
                int rgb = ((r+g+b)/3/kgr)*kgr+d;
                qRgb1 = qRgb(rgb,rgb,rgb);
            }
            else
            {
                int r1=(r/kgr)*kgr+d;
                int g1=(g/kgr)*kgr+d;
                int b1=(b/kgr)*kgr+d;
                qRgb1 = qRgb(r1,g1,b1);
            }
            qPainter.setPen(QPen(QColor(qRgb1),1,Qt::SolidLine));
            qPainter.drawPoint(QPoint(i, j));
            for(int k=0;k<gradation_bit;k++)
            {
                if (black_white)
                    brn.set_in((j*qImage.size().width()+i)*gradation_bit+k, ((r + g + b)/3/kgr)&(k+1));
                    //array[(j*qImage.size().width()+i)*gradation_bit+k]=((r + g + b)/3/kgr)&(k+1);
                else
                {
                    brn.set_in((j*qImage.size().width()+i)*3*gradation_bit+k, (r/kgr)&(k+1));
                    //array[(j*qImage.size().width()+i)*3*gradation_bit+k]=(r/kgr)&(k+1);
                    brn.set_in((j*qImage.size().width()+i)*3*gradation_bit+k + gradation_bit, (g/kgr)&(k+1));
                    //array[(j*qImage.size().width()+i)*3*gradation_bit+k + gradation_bit]=(g/kgr)&(k+1);
                    brn.set_in((j*qImage.size().width()+i)*3*gradation_bit+k + gradation_bit * 2, (b/kgr)&(k+1));
                    //array[(j*qImage.size().width()+i)*3*gradation_bit+k + gradation_bit * 2]=(b/kgr)&(k+1);
                }
            }
        }
}
void SensorPixmap::Zoom_in()
{
    zoom*=zoom_koef;
}
void SensorPixmap::Zoom_out()
{
    zoom/=zoom_koef;
}
void SensorPixmap::X_plus()
{
    x+=deltaXY/zoom;
}
void SensorPixmap::X_minus()
{
    x-=deltaXY/zoom;
}
void SensorPixmap::Y_plus()
{
    y+=deltaXY/zoom;
}
void SensorPixmap::Y_minus()
{
    y-=deltaXY/zoom;
}
DeviceAI::~DeviceAI()
{
    delete [] stepOld;
}
DeviceAI::DeviceAI(_word randomLength2PowBit, _word motorCount, _word brainBits, QSize qSize, QSize qSizeBig, void (*tick_web_engine)(), QWebEngineView* qwev_)
{
    qwev = qwev_;
    stepOld_count = motorCount;
    stepOld = new bool[stepOld_count];
    for(_word i = 0; i < stepOld_count; i++)
        stepOld[i] = 0;
    sensorPixmap.reset(new SensorPixmap(qSize, qSizeBig, 2, true));
    if(sensorPixmap->black_white)
        brn.reset(new brain(randomLength2PowBit, brainBits, static_cast<uint>(qSize.width() * qSize.height() * sensorPixmap->gradation_bit), motorCount, /*ping_pong,*/ tick_web_engine));
    else
        brn.reset(new brain(randomLength2PowBit, brainBits, static_cast<uint>(qSize.width() * qSize.height() * sensorPixmap->gradation_bit*3), motorCount, /*ping_pong,*/ tick_web_engine));
    brain_friend_.reset(new brain_friend(*brn.get()));
}
void DeviceAI::Go (brain &brn)
{
    if(stepOld[0]!= brn.get_out(0))
    {
        sensorPixmap->Y_minus();
        stepOld[0]=brn.get_out(0);
    }
    if(stepOld[1]!=brn.get_out(1))
    {
        sensorPixmap->Y_plus();
        stepOld[1]=brn.get_out(1);
    }
    if(stepOld[2]!=brn.get_out(2))
    {
        sensorPixmap->X_minus();
        stepOld[2]=brn.get_out(2);
    }
    if(stepOld[3]!=brn.get_out(3))
    {
        sensorPixmap->X_plus();
        stepOld[3]=brn.get_out(3);
        //        QKeyEvent* pe = new QKeyEvent(QEvent::KeyPress, Qt::Key_K, Qt::NoModifier);
        //        //QApplication::sendEvent(this, pe);
        //        QApplication::sendEvent(qobject_cast<QMainWindow*>(qApp->topLevelWidgets()[0])->centralWidget()->, pe);
    }
    if(stepOld[4]!=brn.get_out(4))
    {
        sensorPixmap->Zoom_in();
        stepOld[4]=brn.get_out(4);
    }
    if(stepOld[5]!=brn.get_out(5))
    {
        sensorPixmap->Zoom_out();
        stepOld[5]=brn.get_out(5);
    }

    sensorPixmap->PixmapNormalize();
}
