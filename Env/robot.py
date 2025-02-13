import pybullet as p
import pybullet_data
import numpy as np

class Camera:
	def __init__(self, robot_id, server_id, fov=50, aspect=1.0, nearVal=0.01, farVal=20):
		self.robot_id = robot_id
		self.server_id = server_id
		self.fov = fov
		self.aspect = aspect
		self.nearVal = nearVal
		self.farVal = farVal

	def capture_image(self, width=224, height=224):
		position, orientation = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.server_id)
		r_mat = p.getMatrixFromQuaternion(orientation, physicsClientId=self.server_id)
		tx_vec = np.array([r_mat[0], r_mat[3], r_mat[6]])
		tz_vec = np.array([r_mat[2], r_mat[5], r_mat[8]])

		camera_position = position + 0.266 * tx_vec + 0.5 * 0.5 * tz_vec
		target_position = camera_position + 1 * tx_vec
		view_mat = p.computeViewMatrix(cameraEyePosition=camera_position,
									   cameraTargetPosition=target_position,
									   cameraUpVector=tz_vec,
									   physicsClientId=self.server_id)
		proj_mat = p.computeProjectionMatrixFOV(fov=self.fov,
												aspect=self.aspect,
												nearVal=self.nearVal,
												farVal=self.farVal,
												physicsClientId=self.server_id)

		w, h, rgb, depth, seg = p.getCameraImage(width=width,
												 height=height,
												 viewMatrix=view_mat,
												 projectionMatrix=proj_mat,
												 renderer=p.ER_BULLET_HARDWARE_OPENGL)
		return rgb[:, :, :3]

class Robot(Camera):
	def __init__(self, urdf_path: str, server_id, fov=50, aspect=1.0, nearVal=0.01, farVal=20):
		self.server_id = server_id
		self.robot_id = p.loadURDF(urdf_path)

		super(Robot, self).__init__(robot_id=self.robot_id, server_id=server_id,
									fov=fov, aspect=aspect, nearVal=nearVal, farVal=farVal)

		self.available_joints_indexes = [i for i in range(p.getNumJoints(self.robot_id))
										 if p.getJointInfo(self.robot_id, i)[2] != p.JOINT_FIXED]
		self.available_joints_names = [p.getJointInfo(self.robot_id, i)[1]
									   for i in self.available_joints_indexes]

	def action(self, action_index):
		pass

	def get_key_events(self, keys_dict):
		if len(keys_dict):
			if p.B3G_UP_ARROW in keys_dict and p.B3G_LEFT_ARROW in keys_dict:
				p.setJointMotorControlArray(bodyUniqueId=self.robot_id,
											jointIndices=[1, 2],
											controlMode=p.VELOCITY_CONTROL,
											targetVelocities=[5, 10],
											forces=[5, 10]
											)
			elif p.B3G_UP_ARROW in keys_dict and p.B3G_RIGHT_ARROW in keys_dict:
				p.setJointMotorControlArray(bodyUniqueId=self.robot_id,
											jointIndices=[1, 2],
											controlMode=p.VELOCITY_CONTROL,
											targetVelocities=[10, 5],
											forces=[10, 5]
											)
			elif p.B3G_DOWN_ARROW in keys_dict and p.B3G_LEFT_ARROW in keys_dict:
				p.setJointMotorControlArray(bodyUniqueId=self.robot_id,
											jointIndices=[1, 2],
											controlMode=p.VELOCITY_CONTROL,
											targetVelocities=[-5, -10],
											forces=[5, 10]
											)
			elif p.B3G_DOWN_ARROW in keys_dict and p.B3G_RIGHT_ARROW in keys_dict:
				p.setJointMotorControlArray(bodyUniqueId=self.robot_id,
											jointIndices=[1, 2],
											controlMode=p.VELOCITY_CONTROL,
											targetVelocities=[-10, -5],
											forces=[10, 5]
											)
			elif p.B3G_DOWN_ARROW in keys_dict:
				p.setJointMotorControlArray(bodyUniqueId=self.robot_id,
											jointIndices=[1, 2],
											controlMode=p.VELOCITY_CONTROL,
											targetVelocities=[-10, -10],
											forces=[10, 10]
											)
			elif p.B3G_UP_ARROW in keys_dict:
				p.setJointMotorControlArray(bodyUniqueId=self.robot_id,
											jointIndices=[1, 2],
											controlMode=p.VELOCITY_CONTROL,
											targetVelocities=[10, 10],
											forces=[10, 10]
											)
			elif p.B3G_LEFT_ARROW in keys_dict:
				p.setJointMotorControlArray(bodyUniqueId=self.robot_id,
											jointIndices=[1, 2],
											controlMode=p.VELOCITY_CONTROL,
											targetVelocities=[0, 10],
											forces=[0, 10]
											)
			elif p.B3G_RIGHT_ARROW in keys_dict:
				p.setJointMotorControlArray(bodyUniqueId=self.robot_id,
											jointIndices=[1, 2],
											controlMode=p.VELOCITY_CONTROL,
											targetVelocities=[10, 0],
											forces=[10, 0]
											)
			elif ord("a") in keys_dict:
				p.setJointMotorControlArray(bodyUniqueId=self.robot_id,
											jointIndices=[11],
											controlMode=p.VELOCITY_CONTROL,
											targetVelocities=[1],
											forces=[1]
											)
			elif ord("d") in keys_dict:
				p.setJointMotorControlArray(bodyUniqueId=self.robot_id,
											jointIndices=[11],
											controlMode=p.VELOCITY_CONTROL,
											targetVelocities=[-1],
											forces=[1]
											)
			elif ord("f") in keys_dict:
				p.setJointMotorControlArray(bodyUniqueId=self.robot_id,
											jointIndices=[12],
											controlMode=p.VELOCITY_CONTROL,
											targetVelocities=[1],
											forces=[1]
											)
			elif ord("h") in keys_dict:
				p.setJointMotorControlArray(bodyUniqueId=self.robot_id,
											jointIndices=[12],
											controlMode=p.VELOCITY_CONTROL,
											targetVelocities=[-1],
											forces=[1]
											)
			elif ord("j") in keys_dict:
				p.setJointMotorControlArray(bodyUniqueId=self.robot_id,
											jointIndices=[13],
											controlMode=p.VELOCITY_CONTROL,
											targetVelocities=[1],
											forces=[1]
											)
			elif ord("k") in keys_dict:
				p.setJointMotorControlArray(bodyUniqueId=self.robot_id,
											jointIndices=[13],
											controlMode=p.VELOCITY_CONTROL,
											targetVelocities=[-1],
											forces=[1]
											)
			elif ord("n") in keys_dict:
				p.setJointMotorControlArray(bodyUniqueId=self.robot_id,
											jointIndices=[14],
											controlMode=p.VELOCITY_CONTROL,
											targetVelocities=[1],
											forces=[1]
											)
			elif ord("m") in keys_dict:
				p.setJointMotorControlArray(bodyUniqueId=self.robot_id,
											jointIndices=[14],
											controlMode=p.VELOCITY_CONTROL,
											targetVelocities=[-1],
											forces=[1]
											)
			elif ord("o") in keys_dict:
				p.setJointMotorControlArray(bodyUniqueId=self.robot_id,
											jointIndices=[15, 16],
											controlMode=p.VELOCITY_CONTROL,
											targetVelocities=[5, 5],
											forces=[5, 5]
											)
			elif ord("c") in keys_dict:
				p.setJointMotorControlArray(bodyUniqueId=self.robot_id,
											jointIndices=[15, 16],
											controlMode=p.VELOCITY_CONTROL,
											targetVelocities=[-5, -5],
											forces=[5, 5]
											)
		else:
			p.setJointMotorControlArray(
				bodyUniqueId=self.robot_id,
				jointIndices=self.available_joints_indexes,
				controlMode=p.VELOCITY_CONTROL,
				targetVelocities=[0 for _ in self.available_joints_indexes],
				forces=[10 for _ in self.available_joints_indexes]
			)


