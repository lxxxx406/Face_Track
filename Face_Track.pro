QT += widgets

CONFIG += c++17
DEFINES += NCNN_SHARED_LIB
# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    mtcnn.cpp \
    main.cpp \
    mainwindow.cpp \
    showpic.cpp \
    thread.cpp

HEADERS += \
    LandmarkTracking.h \
    mtcnn.h \
    LandmarkTracking.h \
    mainwindow.h \
    showpic.h \
    thread.h

FORMS += \
    mainwindow.ui \
    showpic.ui
INCLUDEPATH +=\
    D:/opencv/build/include \
    D:/ncnn-20260526-windows-vs2022-shared/x64/include/ncnn
LIBS += \
    -LD:/ncnn-20260526-windows-vs2022-shared/x64/lib \
    -LD:/opencv/build/x64/vc16/lib
LIBS += -lncnn
LIBS += -lopencv_world4120d
# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
