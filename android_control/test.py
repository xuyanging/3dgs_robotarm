import scrcpy

# 初始化 scrcpy 客户端
client = scrcpy.Client()

# 定义一个回调函数，用于处理每一帧视频
@client.listener("frame")
def on_frame(frame):
    import cv2
    # frame 是一个 numpy 数组 (RGB 格式)
    cv2.imshow("Scrcpy Video Stream", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        client.stop()

# 启动 scrcpy 客户端
client.start()
