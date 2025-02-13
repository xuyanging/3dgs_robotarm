# -*- coding: utf-8 -*-

import sys
import threading
import cv2
import scrcpy
from adbutils import adb
from PySide6.QtWidgets import QApplication, QWidget



# 创建QApplication对象
if not QApplication.instance():
    app = QApplication([])
else:
    app = QApplication.instance()

# 获取设备列表并创建 scrcpy 客户端字典
items = [i.serial for i in adb.device_list()]
client_dict = {i: scrcpy.Client(device=i, bitrate=8000000) for i in items}  # 降低码率可能有助于性能

def thread_ui(func, *args):
    """开启一个新线程任务"""
    t = threading.Thread(target=func, args=args)
    t.setDaemon(True)
    t.start()

class MyWindow(QWidget):
    """简化的 UI 界面，只包含图像帧显示功能"""

    def __init__(self):
        super().__init__()
        #self.setWindowTitle('实时视频流显示')
        self.now_device = items[0] if items else None
        self.now_client = client_dict.get(self.now_device)

        if self.now_client:
            # 添加帧监听器并启动客户端
            self.now_client.add_listener(scrcpy.EVENT_FRAME, self.main_frame)
            thread_ui(self.now_client.start)

    def main_frame(self, frame):
        """监听设备屏幕数据并使用 OpenCV 显示图像帧"""
        if frame is not None:
            # 转换图像格式并使用 OpenCV 显示
            #frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #print('pass')
            cv2.imwrite("test.jpg", frame)

    def closeEvent(self, event):
        """窗口关闭事件"""
        if self.now_client:
            self.now_client.stop()
        #cv2.destroyAllWindows()

def main():
    widget = MyWindow()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()