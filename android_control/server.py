# -*- coding: utf-8 -*-

import sys
import threading
import cv2
import scrcpy
import asyncio
import websockets
from adbutils import adb
from PySide6.QtWidgets import QApplication, QWidget
import json

# 创建QApplication对象
if not QApplication.instance():
    app = QApplication([])
else:
    app = QApplication.instance()

# 获取设备列表并创建 scrcpy 客户端字典
items = [i.serial for i in adb.device_list()]
client_dict = {i: scrcpy.Client(device=i, bitrate=8000000) for i in items}  # 降低码率可能有助于性能

# 创建一个全局变量来存储图像帧
latest_frame = None

def thread_ui(func, *args):
    """开启一个新线程任务"""
    t = threading.Thread(target=func, args=args)
    t.setDaemon(True)
    t.start()

async def frame_server(websocket, path):
    """WebSocket 服务器，向客户端发送最新的图像帧"""
    global latest_frame
    while True:
        if latest_frame is not None:
            # 将图像帧编码为 JPEG 格式
            _, buffer = cv2.imencode('.jpg', latest_frame)
            # 将图像转换为字节并发送
            await websocket.send(buffer.tobytes())
        await asyncio.sleep(0.03)  # 控制发送帧率

async def click_server(websocket, path):
    """WebSocket 服务器，接收点击坐标并在设备上执行点击"""
    while True:
        # 接收来自客户端的点击坐标数据
        message = await websocket.recv()
        data = json.loads(message)
        x, y = data.get("x"), data.get("y")

        # 获取当前设备和客户端
        now_device = items[0] if items else None
        now_client = client_dict.get(now_device)

        if now_client and x is not None and y is not None:
            # 执行点击操作
            now_client.control.touch(x, y, scrcpy.ACTION_DOWN)
            now_client.control.touch(x, y, scrcpy.ACTION_UP)

class MyWindow(QWidget):
    """简化的 UI 界面，只包含图像帧显示功能"""

    def __init__(self):
        super().__init__()
        self.now_device = items[0] if items else None
        self.now_client = client_dict.get(self.now_device)

        if self.now_client:
            # 添加帧监听器并启动客户端
            self.now_client.add_listener(scrcpy.EVENT_FRAME, self.main_frame)
            thread_ui(self.now_client.start)

    def main_frame(self, frame):
        """监听设备屏幕数据并使用 OpenCV 显示图像帧"""
        global latest_frame
        if frame is not None:
            # 转换图像格式并存储在全局变量
            latest_frame = frame

    def closeEvent(self, event):
        """窗口关闭事件"""
        if self.now_client:
            self.now_client.stop()

def start_websocket_server():
    """启动 WebSocket 服务器"""
    asyncio.set_event_loop(asyncio.new_event_loop())
    # 启动用于发送图像帧的 WebSocket 服务器
    frame_server_task = websockets.serve(frame_server, "localhost", 8765)
    # 启动用于接收点击事件的 WebSocket 服务器
    click_server_task = websockets.serve(click_server, "localhost", 8766)
    asyncio.get_event_loop().run_until_complete(frame_server_task)
    asyncio.get_event_loop().run_until_complete(click_server_task)
    asyncio.get_event_loop().run_forever()

def main():
    # 启动 WebSocket 服务器线程
    thread_ui(start_websocket_server)

    # 启动 Qt 应用程序
    widget = MyWindow()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
