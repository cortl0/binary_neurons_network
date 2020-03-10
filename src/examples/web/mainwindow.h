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

#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <memory>
#include <QMainWindow>
#include <QTimer>

#include "deviceai.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT
public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
    bool ft = false;
    bool busy = false;
private slots:
    void on_pushButtonZoomIncr_clicked();
    void on_pushButtonZoomDecr_clicked();
    void on_pushButtonUp_clicked();
    void on_pushButtonDown_clicked();
    void on_pushButtonLeft_clicked();
    void on_pushButtonRight_clicked();
    void on_pushButtonStart_clicked();
    void on_pushButtonLoad_clicked();
    void on_pushButtonSave_clicked();
    void on_pushButtonAddress_clicked();
    void slotTimerAlarm();
    void on_pushButton_graphical_representation_pressed();
    void on_pushButton_graphical_representation_released();
private:
    Ui::MainWindow *ui;
    std::unique_ptr<DeviceAI> deviceAI;
    void Web_qPixmap_refresh();
    std::unique_ptr<QTimer> timer;
};

#endif // !MAINWINDOW_H
