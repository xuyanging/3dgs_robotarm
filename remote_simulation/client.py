import pybullet as p
import websockets
import asyncio
import json


SERVER_IP = 'ws://172.16.92.211:8769' 

p.connect(p.GUI)
async def receive_data():
    async with websockets.connect(SERVER_IP) as websocket:
        print("已连接到远程服务器，开始接收数据...")
        objects = {}

        while True:
            try:
                data = await websocket.recv()
                sim_data = json.loads(data)
                for uid, info in sim_data.items():
                    pos = info['position']
                    orn = info['orientation']

                    if uid not in objects:
                        objects[uid] = p.createMultiBody(0, p.createCollisionShape(p.GEOM_SPHERE, radius=0.05), -1, pos, orn)
                    else:
                        p.resetBasePositionAndOrientation(objects[uid], pos, orn)

                p.stepSimulation()

            except websockets.ConnectionClosed:
                print("与服务器的连接已关闭")
                break

asyncio.run(receive_data())
p.disconnect()
