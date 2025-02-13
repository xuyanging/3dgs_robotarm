# -*- coding: utf-8 -*-
 
import sys
import threading
import scrcpy
from PySide6.QtGui import QMouseEvent, QImage, QPixmap, QKeyEvent
from adbutils import adb
from PySide6.QtCore import *
from PySide6.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QVBoxLayout, \
    QCheckBox, QLabel, QGridLayout, QSpacerItem, QSizePolicy
from numpy import ndarray
 
# 创建QApplication对象
if not QApplication.instance():
    app = QApplication([])
else:
    app = QApplication.instance()
 
 
items = [i.serial for i in adb.device_list()]  # 设备列表
client_dict = {}  # 设备scrcpy客服端字典
# 为所有设备建立scrcpy服务
for i in items:
    client_dict[i] = scrcpy.Client(device=i, bitrate=1000000000)
 
 
def thread_ui(func, *args):
    """
    开启一个新线程任务\n
    :param func: 要执行的线程函数;
    :param args: 函数中需要传入的参数 Any
    :return:
    """
    t = threading.Thread(target=func, args=args)  # 定义新线程
    t.setDaemon(True)  # 开启线程守护
    t.start()  # 执行线程
 
 
class SignThread(QThread):
    """信号线程"""
 
    def __new__(cls, parent: QWidget, func, *types: type):
        cls.__update_date = Signal(*types, name=func.__name__)  # 定义信号(*types)一个信号中可以有一个或多个类型的数据(int,str,list,...)
        return super().__new__(cls)  # 使用父类__new__方法创建SignThread实例对象
 
    def __init__(self, parent: QWidget, func, *types: type):
        """
        信号线程初始化\n
        :param parent: 界面UI控件
        :param func: 信号要绑定的方法
        :param types: 信号类型,可以是一个或多个(type,...)
        """
        super().__init__(parent)  # 初始化父类
        self.__update_date.connect(func)  # 绑定信号与方法
 
    def send_sign(self, *args):
        """
        使用SignThread发送信号\n
        :param args: 信号的内容
        :return:
        """
        self.__update_date.emit(*args)  # 发送信号元组(type,...)
 
 
class MyWindow(QWidget):
    """UI界面"""
 
    def __init__(self):
        """UI界面初始化"""
        super().__init__()  # 初始化父级
        self.setWindowTitle('多台手机投屏控制示例(python & scrcpy)')  # 设置窗口标题
        self.max_width = 600  # 设置手机投屏宽度
        # 定义元素
        self.check_box = QCheckBox("控制所有设备")  # 定义是否控制所有设备选择框
        self.back_button = QPushButton("BACK")  # 定义返回键
        self.home_button = QPushButton("HOME")  # 定义home键
        self.recent_button = QPushButton("RECENT")  # 定义最近任务键
        self.video = QLabel("设备屏幕信息加载......")  # 定义手机投屏控制标签
        self.video.setStyleSheet("border-width: 3px;border-style: solid;border-color: black;")  # 定义投屏标签样式
        self.video_list = []  # 定义手机投屏标签列表
        for i in items:
            self.video_list.append(QLabel(i))  # 把投屏标签加入列表
        self.main_layout = QHBoxLayout(self)  # 定义主布局容器
        self.frame_layout = QVBoxLayout()  # 定义投屏操控框容器
        self.button_layout = QHBoxLayout()
        self.device_layout = QVBoxLayout()  # 定义投屏容器
        self.list_layout = QGridLayout()  # 定义投屏列表布局容器
        self.spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)  # 弹性空间
        self.device_spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)  # 弹性空间
        self.v_spacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)  # 弹性空间
        # 页面布局
        self.main_layout.addLayout(self.frame_layout)
        self.main_layout.addLayout(self.device_layout)
        self.main_layout.addItem(self.v_spacer)
        self.frame_layout.addWidget(self.video)
        self.frame_layout.addLayout(self.button_layout)
        self.frame_layout.addWidget(self.check_box)
        self.frame_layout.addItem(self.spacer)
        self.button_layout.addWidget(self.back_button)
        self.button_layout.addWidget(self.home_button)
        self.button_layout.addWidget(self.recent_button)
        self.device_layout.addLayout(self.list_layout)
        self.device_layout.addItem(self.device_spacer)
        # 交互事件
        self.back_button.clicked.connect(self.click_key(scrcpy.KEYCODE_BACK))
        self.home_button.clicked.connect(self.click_key(scrcpy.KEYCODE_HOME))
        self.recent_button.clicked.connect(self.click_key(scrcpy.KEYCODE_APP_SWITCH))
        self.video.mousePressEvent = self.mouse_event(scrcpy.ACTION_DOWN)
        self.video.mouseMoveEvent = self.mouse_event(scrcpy.ACTION_MOVE)
        self.video.mouseReleaseEvent = self.mouse_event(scrcpy.ACTION_UP)
        self.keyPressEvent = self.on_key_event(scrcpy.ACTION_DOWN)
        self.keyReleaseEvent = self.on_key_event(scrcpy.ACTION_UP)
        # 所有设备屏幕有序排布，最多15台设备，可按需修改
        if len(items) > 0:
            self.now_device = items[0]
            self.now_client = client_dict[items[0]]
            self.now_client.add_listener(scrcpy.EVENT_FRAME, self.main_frame)
            for num in range(len(items)):
                self.video_list[num].setStyleSheet("border-width: 3px;border-style: solid;border-color: black;")
                self.video_list[num].mousePressEvent = self.switch_video(items[num])
                client = client_dict[items[num]]
                client.add_listener(scrcpy.EVENT_FRAME, self.on_frame(num, client))
                if num < 5:
                    self.list_layout.addWidget(self.video_list[num], 0, num, 1, 1)
                elif num < 10:
                    self.list_layout.addWidget(self.video_list[num], 1, num - 5, 1, 1)
                elif num < 15:
                    self.list_layout.addWidget(self.video_list[num], 2, num - 10, 1, 1)
 
        self.mouse_thread = SignThread(self, self.mouse_exe, int, int, int)
        self.main_thread = SignThread(self, self.main_exe, ndarray)
        self.on_thread = SignThread(self, self.on_exe, int, int, ndarray)
 
    def click_key(self, key_value: int):
        """
        按键事件\n
        :param key_value: 键值
        :return:
        """
 
        def key_event():
            if self.check_box.isChecked():
                for i in client_dict:
                    client_dict[i].control.keycode(key_value, scrcpy.ACTION_DOWN)
                    client_dict[i].control.keycode(key_value, scrcpy.ACTION_UP)
            else:
                self.now_client.control.keycode(key_value, scrcpy.ACTION_DOWN)
                self.now_client.control.keycode(key_value, scrcpy.ACTION_UP)
 
        return key_event
 
    def switch_video(self, device: str):
        """
        切换设备屏幕为主控屏幕\n
        :param device: 设备序列号
        :return:
        """
 
        def now_video(evt: QMouseEvent):
            app.processEvents()
            self.now_client.remove_listener(scrcpy.EVENT_FRAME, self.main_frame)
            self.now_client = client_dict[device]
            self.now_client.add_listener(scrcpy.EVENT_FRAME, self.main_frame)
            self.now_client.control.keycode(224, scrcpy.ACTION_DOWN)
            self.now_client.control.keycode(224, scrcpy.ACTION_UP)
            bound = self.now_client.resolution
            self.now_client.control.swipe(bound[0] / 2, bound[1] / 2, bound[0] / 2, bound[1] / 2 - 20)
            self.now_client.control.swipe(bound[0] / 2, bound[1] / 2 - 20, bound[0] / 2, bound[1] / 2)
            self.now_device = device
 
        return now_video
 
    def main_frame(self, frame: ndarray):
        """
        监听设备屏幕数据,设置控制窗口图像\n
        :param frame: 图像帧
        :return:
        """
        if frame is not None:
            self.main_thread.send_sign(frame)
 
    def main_exe(self, frame):
        """
        主控屏幕图像设置\n
        :param frame: 图像帧
        :return:
        """
        ratio = self.max_width / max(self.now_client.resolution)
        image = QImage(
            frame,
            frame.shape[1],
            frame.shape[0],
            frame.shape[1] * 3,
            QImage.Format_BGR888,
        )
        pix = QPixmap(image)
        pix.setDevicePixelRatio(1 / ratio)
        self.video.setPixmap(pix)
 
    def on_frame(self, num: int, client: scrcpy.Client):
        """
        监听设备屏幕数据,设置小窗口图像\n
        :param num: 设备投屏序号
        :param client: scrcpy服务
        :return:
        """
 
        def client_frame(frame: ndarray):
            if frame is not None:
                self.on_thread.send_sign(num, max(client.resolution), frame)
 
        return client_frame
 
    def on_exe(self, num: int, resolution: int, frame):
        """
        小窗口图像设置\n
        :param num: 设备投屏序号
        :param resolution: 设备宽度
        :param frame: 图像帧
        :return:
        """
        ratio = 300 / resolution
        image = QImage(
            frame,
            frame.shape[1],
            frame.shape[0],
            frame.shape[1] * 3,
            QImage.Format_BGR888,
        )
        pix = QPixmap(image)
        pix.setDevicePixelRatio(1 / ratio)
        self.video_list[num].setPixmap(pix)
 
    def mouse_event(self, action=scrcpy.ACTION_DOWN):
        """
        鼠标事件\n
        :param action: 事件类型
        :return: 对应的事件函数
        """
 
        def event(evt: QMouseEvent):
            focused_widget = QApplication.focusWidget()
            if focused_widget is not None:
                focused_widget.clearFocus()
            ratio = self.max_width / max(self.now_client.resolution)
            self.mouse_thread.send_sign(evt.position().x() / ratio, evt.position().y() / ratio, action)
 
        return event
 
    def mouse_exe(self, x: int, y: int, action: int):
        """
        执行鼠标事件\n
        :param x: x坐标
        :param y: y坐标
        :param action: 事件类型
        :return:
        """
        if self.check_box.isChecked():
            for i in client_dict:
                client_dict[i].control.touch(x, y, action)
        else:
            self.now_client.control.touch(x, y, action)
 
    def on_key_event(self, action=scrcpy.ACTION_DOWN):
        """
        键盘按键事件\n
        :param action: 事件类型
        :return: 对应的事件函数
        """
 
        def handler(evt: QKeyEvent):
            code = self.key_code(evt.key())
            if code != -1:
                if self.check_box.isChecked():
                    for i in client_dict:
                        client_dict[i].control.keycode(code, action)
                else:
                    self.now_client.control.keycode(code, action)
 
        return handler
 
    @staticmethod
    def key_code(code):
        """
        Map qt keycode ti android keycode
        Args:
            code: qt keycode
            android keycode, -1 if not founded
        """
        if 48 <= code <= 57:
            return code - 48 + 7
        if 65 <= code <= 90:
            return code - 65 + 29
        if 97 <= code <= 122:
            return code - 97 + 29
 
        hard_code = {
            35: scrcpy.KEYCODE_POUND,
            42: scrcpy.KEYCODE_STAR,
            44: scrcpy.KEYCODE_COMMA,
            46: scrcpy.KEYCODE_PERIOD,
            32: scrcpy.KEYCODE_SPACE,
            16777219: scrcpy.KEYCODE_DEL,
            16777248: scrcpy.KEYCODE_SHIFT_LEFT,
            16777220: scrcpy.KEYCODE_ENTER,
            16777217: scrcpy.KEYCODE_TAB,
            16777249: scrcpy.KEYCODE_CTRL_LEFT,
            16777235: scrcpy.KEYCODE_DPAD_UP,
            16777237: scrcpy.KEYCODE_DPAD_DOWN,
            16777234: scrcpy.KEYCODE_DPAD_LEFT,
            16777236: scrcpy.KEYCODE_DPAD_RIGHT,
        }
        if code in hard_code:
            return hard_code[code]
 
        print(f"Unknown keycode: {code}")
        return -1
 
    def closeEvent(self, _):
        """窗口关闭事件"""
        for i in client_dict:
            client_dict[i].stop()  # 关闭scrcpy服务
 
 
def main():
    for i in client_dict:
        thread_ui(client_dict[i].start)  # 给每一台设备单独开启一个scrcpy服务线程
    widget = MyWindow()  # 实例化UI线程
    widget.resize(1200, 800)  # 设置窗口大小
    widget.show()  # 展示窗口
    sys.exit(app.exec())  # 持续刷新窗口
 
 
if __name__ == '__main__':
    main()