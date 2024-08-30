import sys
import numpy as np
import open3d as o3d
from PyQt5 import QtWidgets, uic, QtCore

class PointCloudViewer(QtWidgets.QMainWindow):
    def __init__(self):
        super(PointCloudViewer, self).__init__()
        uic.loadUi('calibration_tools/point_cloud_viewer.ui', self)  # Load .ui file

        self.loadButton.clicked.connect(self.load_point_cloud)

        self.yawSlider.valueChanged.connect(self.update_transformation_from_sliders)
        self.pitchSlider.valueChanged.connect(self.update_transformation_from_sliders)
        self.rollSlider.valueChanged.connect(self.update_transformation_from_sliders)
        self.xTranslationSlider.valueChanged.connect(self.update_transformation_from_sliders)
        self.yTranslationSlider.valueChanged.connect(self.update_transformation_from_sliders)
        self.zTranslationSlider.valueChanged.connect(self.update_transformation_from_sliders)

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name='Open3D', width=800, height=600, left=50, top=50)
        self.point_cloud = None

        # Create coordinate frame
        self.coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0, origin=[0, 0, 0])
        self.vis.add_geometry(self.coordinate_frame)

        # Initialize transformations
        self.initial_rotation = np.eye(3)
        self.initial_translation = np.zeros(3)

        # Save the initial state of the coordinate frame
        self.saved_initial_rotation = np.eye(3)
        self.saved_initial_translation = np.zeros(3)

        # Create a timer to update Open3D visualization window
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_visualizer)
        self.timer.start(30)

        self.installEventFilter(self)

    def update_visualizer(self):
        self.vis.poll_events()
        self.vis.update_renderer()

    def load_point_cloud(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open PLY file', '', 'PLY files (*.ply)')
        if file_path:
            self.point_cloud = o3d.io.read_point_cloud(file_path)
            self.point_cloud.colors = o3d.utility.Vector3dVector(np.asarray(self.point_cloud.colors) * 0.8 + 0.2)
            self.vis.clear_geometries()
            self.vis.add_geometry(self.point_cloud)
            self.vis.add_geometry(self.coordinate_frame)

            # Set rendering options
            opt = self.vis.get_render_option()
            opt.point_size = 2.0
            opt.background_color = np.asarray([1, 1, 1])  # White background

            self.vis.update_geometry(self.point_cloud)
            self.vis.poll_events()
            self.vis.update_renderer()
            self.statusbar.showMessage(f'Loaded point cloud from {file_path}')
            self.reset_view()

            # Save the initial state of the coordinate frame
            self.saved_initial_rotation = np.eye(3)
            self.saved_initial_translation = np.zeros(3)

    def update_transformation_from_sliders(self):
        yaw = self.yawSlider.value() / 100.0 * np.pi / 180
        pitch = self.pitchSlider.value() / 100.0 * np.pi / 180
        roll = self.rollSlider.value() / 100.0 * np.pi / 180

        x_translation = self.xTranslationSlider.value() / 100.0
        y_translation = self.yTranslationSlider.value() / 100.0
        z_translation = self.zTranslationSlider.value() / 100.0

        self.update_transformation(yaw, pitch, roll, x_translation, y_translation, z_translation)

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.KeyPress:
            key = event.key()
            if key == QtCore.Qt.Key_Up:
                self.pitchSlider.setValue(self.pitchSlider.value() + 1)
            elif key == QtCore.Qt.Key_Down:
                self.pitchSlider.setValue(self.pitchSlider.value() - 1)
            elif key == QtCore.Qt.Key_Left:
                self.yawSlider.setValue(self.yawSlider.value() - 1)
            elif key == QtCore.Qt.Key_Right:
                self.yawSlider.setValue(self.yawSlider.value() + 1)
            elif key == QtCore.Qt.Key_Q:
                self.rollSlider.setValue(self.rollSlider.value() + 1)
            elif key == QtCore.Qt.Key_E:
                self.rollSlider.setValue(self.rollSlider.value() - 1)
            return True
        return super().eventFilter(obj, event)

    def update_transformation(self, yaw, pitch, roll, x_translation, y_translation, z_translation):
        self.yawSlider.setValue(int(yaw * 180 / np.pi * 100))
        self.pitchSlider.setValue(int(pitch * 180 / np.pi * 100))
        self.rollSlider.setValue(int(roll * 180 / np.pi * 100))
        self.xTranslationSlider.setValue(int(x_translation * 100))
        self.yTranslationSlider.setValue(int(y_translation * 100))
        self.zTranslationSlider.setValue(int(z_translation * 100))

        self.yawLabel.setText(f'Yaw: {self.yawSlider.value() / 100.0}')
        self.pitchLabel.setText(f'Pitch: {self.pitchSlider.value() / 100.0}')
        self.rollLabel.setText(f'Roll: {self.rollSlider.value() / 100.0}')
        self.xTranslationLabel.setText(f'X Translation: {self.xTranslationSlider.value() / 100.0}')
        self.yTranslationLabel.setText(f'Y Translation: {self.yTranslationSlider.value() / 100.0}')
        self.zTranslationLabel.setText(f'Z Translation: {self.zTranslationSlider.value() / 100.0}')

        # Compute rotation matrix
        R = self.euler_to_rotation_matrix(yaw, pitch, roll)
        self.rotationMatrixLabel.setText(f'Rotation Matrix:\n{R}')
        print(R)

        # Compute translation vector
        T = np.array([x_translation, y_translation, z_translation])
        self.translationMatrixLabel.setText(f'Translation Vector: {T}')
        print(T)

        # Apply the new transformation to the coordinate frame
        self.coordinate_frame.translate(-self.saved_initial_translation)
        self.coordinate_frame.rotate(np.linalg.inv(self.saved_initial_rotation), center=(0, 0, 0))
        

        self.coordinate_frame.rotate(R, center=(0, 0, 0))
        self.coordinate_frame.translate(T)

        # Update the saved state
        self.saved_initial_rotation = R
        self.saved_initial_translation = T

        self.vis.update_geometry(self.coordinate_frame)
        self.vis.poll_events()
        self.vis.update_renderer()

    def reset_view(self):
        if self.point_cloud is not None:
            # Calculate the center of the point cloud
            bounding_box = self.point_cloud.get_axis_aligned_bounding_box()
            center = bounding_box.get_center()

            # Set a suitable view angle
            view_ctl = self.vis.get_view_control()
            view_ctl.set_lookat(center)
            view_ctl.set_up([0, 1, 0])
            view_ctl.set_front([0, 0, -1])
            view_ctl.set_zoom(0.45)  # Adjust zoom to fit the view

    @staticmethod
    def euler_to_rotation_matrix(yaw, pitch, roll):
        Ryaw = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                         [np.sin(yaw), np.cos(yaw), 0],
                         [0, 0, 1]])

        Rpitch = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                           [0, 1, 0],
                           [-np.sin(pitch), 0, np.cos(pitch)]])

        Rroll = np.array([[1, 0, 0],
                          [0, np.cos(roll), -np.sin(roll)],
                          [0, np.sin(roll), np.cos(roll)]])

        R = Ryaw @ Rpitch @ Rroll
        return R

app = QtWidgets.QApplication(sys.argv)
window = PointCloudViewer()
window.show()
sys.exit(app.exec_())
