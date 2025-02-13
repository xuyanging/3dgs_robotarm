import pybullet as p
import pybullet_data

import time

from Env.robot import Robot

class World:
	def __init__(self, urdf_path, mode=p.GUI):
		self.server_id = p.connect(mode)
		p.setGravity(0, 0, -10)
		p.setAdditionalSearchPath(pybullet_data.getDataPath())
		self.plane_id = p.loadURDF("plane.urdf")
		self.robot = Robot(urdf_path=urdf_path, server_id=self.server_id)
		p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
		p.setRealTimeSimulation(0)
		self.startPos = [0, 0, 0]
		self.startOrientation = p.getQuaternionFromEuler([0, 0, 0])
		p.resetBasePositionAndOrientation(self.robot.robot_id, self.startPos, self.startOrientation)
		self.done = False

	def step(self, capture_image=False):
		p.stepSimulation()
		if capture_image:
			self.robot.capture_image()
		location, _ = p.getBasePositionAndOrientation(self.robot.robot_id)
		p.resetDebugVisualizerCamera(
			cameraDistance=1,
			cameraYaw=110,
			cameraPitch=-30,
			cameraTargetPosition=location
		)
		# time.sleep(1 / 240)

	def reset(self):
		p.resetBasePositionAndOrientation(self.robot.robot_id, self.startPos, self.startOrientation)
		self.done = False

	def keyboard_control(self):
		keys_dict = p.getKeyboardEvents()
		self.robot.get_key_events(keys_dict=keys_dict)
		if ord("q") in keys_dict:
			self.done = True

	def close(self):
		p.disconnect(self.server_id)
