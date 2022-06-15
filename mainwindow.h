#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "camera.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_btnStart_clicked();

    void on_btnStartDetect_clicked();

private:
    Ui::MainWindow *ui;

    Camera *cam;


public slots:
    void showImage(QImage img);
};
#endif // MAINWINDOW_H
