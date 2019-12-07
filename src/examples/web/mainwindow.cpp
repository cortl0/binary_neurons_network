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

#include "mainwindow.h"
#include "ui_mainwindow.h"
#include<QImage>
#include<QColor>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    int k=4;
    deviceAI.reset(new DeviceAI(27, 4, 17, QSize(ui->qLabel->size().width()/k, ui->qLabel->size().height()/k), ui->preview->size(), nullptr, ui->preview));
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
    if (!deviceAI->GetBrain().clock_cycle_completed)
        return;
    if(busy)
        return;
    else
        busy=true;
    deviceAI->Go(deviceAI->GetBrain());
    ui->qLabel->setPixmap((deviceAI->GetSensorPixmap()).GetPixmap());
    QPixmap qPixmap = ui->preview->grab(QRect(QPoint(0,0), ui->preview->size()));
    deviceAI->GetSensorPixmap().FillBinary(qPixmap, deviceAI->GetBrain());
    deviceAI->brain_friend_->brain_get_state();
    ui->labelDebug->setText(deviceAI->brain_friend_->brain_get_state() + '\n' +
                            "x=" + QString::number(deviceAI->GetSensorPixmap().x) +
                            " y=" + QString::number(deviceAI->GetSensorPixmap().y) +
                            " zoom=" + QString::number(deviceAI->GetSensorPixmap().zoom));
    deviceAI->GetBrain().clock_cycle_completed = false;
    busy=false;
    return;
}
void MainWindow::on_pushButtonZoomIncr_clicked()
{
    deviceAI->GetSensorPixmap().Zoom_in();
    deviceAI->GetSensorPixmap().PixmapNormalize();
}
void MainWindow::on_pushButtonZoomDecr_clicked()
{
    deviceAI->GetSensorPixmap().Zoom_out();
    deviceAI->GetSensorPixmap().PixmapNormalize();
}
void MainWindow::on_pushButtonUp_clicked()
{
    deviceAI->GetSensorPixmap().Y_minus();
    deviceAI->GetSensorPixmap().PixmapNormalize();
}
void MainWindow::on_pushButtonDown_clicked()
{
    deviceAI->GetSensorPixmap().Y_plus();
    deviceAI->GetSensorPixmap().PixmapNormalize();
}
void MainWindow::on_pushButtonLeft_clicked()
{
    deviceAI->GetSensorPixmap().X_minus();
    deviceAI->GetSensorPixmap().PixmapNormalize();
}
void MainWindow::on_pushButtonRight_clicked()
{
    deviceAI->GetSensorPixmap().X_plus();
    deviceAI->GetSensorPixmap().PixmapNormalize();
}
void MainWindow::on_pushButtonStart_clicked()
{
    if(ft)
    {
        timer->stop();
        ui->pushButtonStart->setText("Start");
        ui->pushButtonLoad->setEnabled(true);
        ui->pushButtonSave->setEnabled(true);
        deviceAI->brain_friend_->stop();
    }
    else
    {
        timer->start(50);
        ui->pushButtonStart->setText("Stop");
        ui->pushButtonLoad->setEnabled(false);
        ui->pushButtonSave->setEnabled(false);
        deviceAI->GetBrain().start();
    }
    ft=!ft;
}
void MainWindow::on_pushButtonLoad_clicked()
{
    deviceAI->brain_friend_->load();
}
void MainWindow::on_pushButtonSave_clicked()
{
    deviceAI->brain_friend_->save();
}
void MainWindow::on_pushButtonAddress_clicked()
{
    ui->preview->load(QUrl(ui->lineEditAddress->text()));
}
