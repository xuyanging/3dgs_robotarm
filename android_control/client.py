import cv2
import numpy as np
import asyncio
import websockets
import threading

# 全局变量 frame
frame = None

# 接收帧的协程
async def receive_frame():
    global frame
    uri = "ws://localhost:8765"
    async with websockets.connect(uri) as websocket:
        while True:
            frame_data = await websocket.recv()
            # 将字节数据转换为 NumPy 数组并解码为图像
            np_arr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

# 在单独的线程中运行 receive_frame() 协程
def start_receive_frame():
    asyncio.run(receive_frame())

# 启动接收帧的线程
thread = threading.Thread(target=start_receive_frame)
thread.start()

# 主线程运行 OpenCV 显示逻辑
while True:
    if frame is not None:
        cv2.imshow("Received Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
