/*
 *   binary neurons network
 *   created by Ilya Shishkin
 *   cortl@8iter.ru
 *   http://8iter.ru/ai.html
 *   https://github.com/cortl0/binary_neurons_network
 *   licensed by GPL v3.0
 */

#include "mainwindow.h"
#include "ui_mainwindow.h"
#include<QImage>
#include<QColor>
#include "qpainter.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    int k=4;
    deviceAi.reset(new DeviceAI(6, 16,
                                QSize(ui->qLabel->size().width()/k, ui->qLabel->size().height()/k),
                                ui->preview->size(), ui->preview));
    ui->lineEditAddress->setText("http://youtube.com/");
    ui->preview->load(QUrl(ui->lineEditAddress->text()));
    ui->preview->show();
    timer.reset(new QTimer());
    connect(timer.get(), SIGNAL(timeout()), this, SLOT(slotTimerAlarm()));
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::slotTimerAlarm()
{
    if(busy)
        return;
    else
        busy=true;
    deviceAi->Go();
    ui->qLabel->setPixmap((deviceAi->GetSensorPixmap()).GetPixmap());
    QPixmap qPixmap = ui->preview->grab(QRect(QPoint(0,0), ui->preview->size()));
    deviceAi->GetSensorPixmap().FillBinary(qPixmap, *deviceAi->brain_);
    ui->labelDebug->setText(deviceAi->brain_->brain_get_state() + '\n' +
                            "x=" + QString::number(static_cast<int>(deviceAi->GetSensorPixmap().x)) +
                            " y=" + QString::number(static_cast<int>(deviceAi->GetSensorPixmap().y)) +
                            " zoom=" + QString::number(deviceAi->GetSensorPixmap().zoom));
    busy=false;
    return;
}

void MainWindow::on_pushButtonZoomIncr_clicked()
{
    deviceAi->GetSensorPixmap().Zoom_in();
    deviceAi->GetSensorPixmap().PixmapNormalize();
}

void MainWindow::on_pushButtonZoomDecr_clicked()
{
    deviceAi->GetSensorPixmap().Zoom_out();
    deviceAi->GetSensorPixmap().PixmapNormalize();
}

void MainWindow::on_pushButtonUp_clicked()
{
    deviceAi->GetSensorPixmap().Y_minus();
    deviceAi->GetSensorPixmap().PixmapNormalize();
}

void MainWindow::on_pushButtonDown_clicked()
{
    deviceAi->GetSensorPixmap().Y_plus();
    deviceAi->GetSensorPixmap().PixmapNormalize();
}

void MainWindow::on_pushButtonLeft_clicked()
{
    deviceAi->GetSensorPixmap().X_minus();
    deviceAi->GetSensorPixmap().PixmapNormalize();
}

void MainWindow::on_pushButtonRight_clicked()
{
    deviceAi->GetSensorPixmap().X_plus();
    deviceAi->GetSensorPixmap().PixmapNormalize();
}

void MainWindow::on_pushButtonStart_clicked()
{
    if(ft)
    {
        timer->stop();
        ui->pushButtonStart->setText("Start");
        ui->pushButtonLoad->setEnabled(true);
        ui->pushButtonSave->setEnabled(true);
        deviceAi->brain_->stop();
    }
    else
    {
        timer->start(50);
        ui->pushButtonStart->setText("Stop");
        ui->pushButtonLoad->setEnabled(false);
        ui->pushButtonSave->setEnabled(false);

        deviceAi->brain_->primary_filling();

        deviceAi->brain_->start(/*this, clock_cycle_handler*/);
    }
    ft=!ft;
}

void MainWindow::on_pushButtonLoad_clicked()
{
    deviceAi->brain_->load();
}

void MainWindow::on_pushButtonSave_clicked()
{
    deviceAi->brain_->save();
}

void MainWindow::on_pushButtonAddress_clicked()
{
    ui->preview->load(QUrl(ui->lineEditAddress->text()));
}

void MainWindow::on_pushButton_graphical_representation_pressed()
{
    deviceAi->brain_->stop();
    std::map<u_word, u_word> m = deviceAi->brain_->graphical_representation();
    QPixmap qPixmap;
    qPixmap = QPixmap(QSize(64*4, 36*4));
    QPainter qPainter(&qPixmap);
    QRgb qRgb1 = QRgb(0);
    qPainter.setPen(QPen(QColor(qRgb1),1,Qt::SolidLine));
    qPainter.fillRect(qPixmap.rect(),QColor(255, 255, 255, 255));
    int zoom = 1;
    int max_level = 0;
    std::for_each(m.begin(),m.end(),[&](std::pair<int, int> p)
    {
        if(p.first / zoom < qPixmap.size().width())
            if(p.second / zoom < qPixmap.size().height())
                qPainter.drawPoint(QPoint(p.first / zoom, p.second / zoom));
        if (max_level < p.first)
            max_level = p.first;
    });
    ui->qLabel->setPixmap(qPixmap);
    QString qString;
    qString += "max_level = " + QString::number(max_level) + "\n";
    qString += deviceAi->brain_->brain_get_representation();
    ui->labelDebug->setText(qString);
}

void MainWindow::on_pushButton_graphical_representation_released()
{
    deviceAi->brain_->primary_filling();
    deviceAi->brain_->start();
}
