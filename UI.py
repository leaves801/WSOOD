import sys
import cv2
from PySide6.QtCore import QRectF, Qt
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, \
    QFileDialog, QGraphicsScene, QGraphicsView, QLabel, QComboBox, QSizePolicy
from PySide6.QtGui import QPixmap, QImage
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import os

class ModelInferenceApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Model Inference")
        self.setGeometry(100, 100, 800, 600)

        # 创建标题标签
        self.title_label = QLabel("基于SOOD模型的小麦病害检测")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setMinimumWidth(100)

        # 设置标题字体
        font = self.title_label.font()
        font.setPointSize(24)  # 设置字体大小
        font.setBold(True)  # 设置加粗
        self.title_label.setFont(font)


        # 创建下拉表
        self.model_selection_combo = QComboBox()
        self.model_selection_combo.addItem("小麦赤霉病检测")
        self.model_selection_combo.addItem("小麦黑穗病检测")
        self.model_selection_combo.currentIndexChanged.connect(self.change_model)

        self.model_selection_combo.setStyleSheet("QComboBox { font-size: 20px; }")
        self.model_selection_combo.setMinimumWidth(300)


        # 创建左侧视图和场景
        self.left_scene = QGraphicsScene()
        self.left_view = QGraphicsView(self.left_scene)

        # 设置左侧视图大小和样式
        self.left_view.setFixedSize(600, 600)
        self.left_view.setFrameStyle(QGraphicsView.Box)

        # 创建右侧视图和场景
        self.right_scene = QGraphicsScene()
        self.right_view = QGraphicsView(self.right_scene)

        # 设置右侧视图大小和样式
        self.right_view.setFixedSize(600, 600)
        self.right_view.setFrameStyle(QGraphicsView.Box)

        # 创建水平布局，用于放置左右两个视图
        self.horizontal_layout = QHBoxLayout()
        self.horizontal_layout.addWidget(self.left_view)
        self.horizontal_layout.addWidget(self.right_view)

        # 创建底部布局
        self.bottom_layout = QHBoxLayout()
        self.load_image_button = QPushButton("选择图片")
        self.load_image_button.clicked.connect(self.load_image)
        self.bottom_layout.addWidget(self.load_image_button)
        self.start_detection_button = QPushButton("开始检测")
        self.start_detection_button.clicked.connect(self.start_detection)
        self.bottom_layout.addWidget(self.start_detection_button)
        # 创建清空视图按钮
        self.clear_view_button = QPushButton("清空图片")
        self.clear_view_button.clicked.connect(self.clear_view)  # 修正函数名称
        self.bottom_layout.addWidget(self.clear_view_button)

        self.load_image_button.setFixedWidth(120)  # 设置按钮宽度为 120 像素
        self.start_detection_button.setFixedWidth(120)
        self.clear_view_button.setFixedWidth(120)
        self.load_image_button.setFixedHeight(40)  # 设置按钮高度为 40 像素
        self.start_detection_button.setFixedHeight(40)
        self.clear_view_button.setFixedHeight(40)

        font = self.load_image_button.font()
        font.setPointSize(12)  # 设置字体大小为 12
        self.load_image_button.setFont(font)

        font = self.start_detection_button.font()
        font.setPointSize(12)
        self.start_detection_button.setFont(font)

        font = self.clear_view_button.font()
        font.setPointSize(12)
        self.clear_view_button.setFont(font)

        # 整体布局
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.title_label)  # 添加标题标签
        main_layout.addWidget(self.model_selection_combo)
        main_layout.addLayout(self.horizontal_layout)
        main_layout.addLayout(self.bottom_layout)

        # 主窗口设置布局
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # 初始化模型
        self.model = None
        self.config = None
        self.checkpoint = None

    def load_image(self):
        file_dialog = QFileDialog(parent=None)
        file_dialog.setNameFilter("Image Files (*.jpg *.png)")
        file_dialog.setViewMode(QFileDialog.Detail)
        file_dialog.setFileMode(QFileDialog.ExistingFile)

        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]
            self.img_path = file_path
            # 读取图像并转换为RGB颜色空间
            self.img = cv2.imread(file_path)
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)  # 转换颜色空间
            self.display_image(self.img_path)


    def display_image(self,img_path):

        # 清除左侧场景中的所有项目
        self.left_scene.clear()

        # 添加图像到左侧场景中，并设置位置为左上角
        pixmap = QPixmap(img_path)
        pixmap_item = self.left_scene.addPixmap(pixmap)
        pixmap_item.setPos(0, 0)
        # 设置右侧场景的大小并适配到右侧视图中
        scene_rect = QRectF(0, 0, pixmap.width(), pixmap.height())
        self.left_scene.setSceneRect(scene_rect)
        self.left_view.fitInView(scene_rect, Qt.KeepAspectRatio)

    def change_model(self, index):
            if index == 0:
                # 使用第一套文件
                self.config = 'configs/ssad_fcos/sood_fcos_pinkdata.py'
                self.checkpoint = 'weight/pink_0.701682_mAP.pth'
            elif index == 1:
                # 使用第二套文件
                self.config = 'configs/ssad_fcos/sood_fcos_smutdata.py'
                self.checkpoint = 'weight/smut_0.789181_mAP.pth'

    def start_detection(self):
        # 初始化模型
        model_inference_app.model = init_detector(self.config, self.checkpoint, device='cpu')
        result = inference_detector(self.model, self.img_path)
        result_img = self.visualize_result(result)
        self.display_detection_result(result_img)


    def visualize_result(self,result):
        # 保存结果图像
        result_file = "result.jpg"
        show_result_pyplot(self.model,self.img_path, result, palette='red',out_file=result_file)
        return result_file

    def display_detection_result(self, img_path):
        # 清除右侧场景中的所有项目
        self.right_scene.clear()
        # 添加图像到右侧场景中，并设置位置为左上角
        pixmap = QPixmap(img_path)
        pixmap_item = self.right_scene.addPixmap(pixmap)
        pixmap_item.setPos(0, 0)
        # 设置右侧场景的大小并适配到右侧视图中
        scene_rect = QRectF(0, 0, pixmap.width(), pixmap.height())
        self.right_scene.setSceneRect(scene_rect)
        self.right_view.fitInView(scene_rect, Qt.KeepAspectRatio)


    def clear_view(self):
        # 清空左侧和右侧场景中的所有项目
        self.left_scene.clear()
        self.right_scene.clear()

if __name__ == "__main__":
    device = 'cpu'

    app = QApplication(sys.argv)
    model_inference_app = ModelInferenceApp()


    model_inference_app.showMaximized()  # 将窗口最大化
    sys.exit(app.exec())
