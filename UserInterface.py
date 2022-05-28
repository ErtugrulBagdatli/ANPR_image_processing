from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import QtGui, QtCore
import sys
import cv2
import NumberPlateDetection
import numpy as np

import os


class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Car Number Plate Detection and Extraction')
        self.setGeometry(160, 50, 1600, 1000)

        # self.selectedImage = cv2.imread('openImage.jpeg')
        self.selectedImage = cv2.imread(r'.\Car Images\car.png')

        self.set_image_placeholders()
        self.set_buttons()
        self.set_labels()

        self.detect_plate()

        '''
            Original
            Grayscale
            Enhanced
            Increasing Contrast(Add Weight)
            GaussianBlur
            Canny Edge Detection
            All Contours
            Top 30  Contours ----------------
            Final Image With Number Plate Detected
            Cropped Plate ------------
        '''

        self.show()

    def set_labels(self):
        self.original_image_label = QLabel(parent=self)
        self.original_image_label.setText("Original Image")
        self.original_image_label.setFont(QFont('verdana', 10))
        self.original_image_label.move(150, 10)

        self.grayscale_label = QLabel(parent=self)
        self.grayscale_label.setText("Grayscale")
        self.grayscale_label.setFont(QFont('verdana', 10))
        self.grayscale_label.move(550, 10)

        self.enhanced_label = QLabel(parent=self)
        self.enhanced_label.setText("Enhanced")
        self.enhanced_label.setFont(QFont('verdana', 10))
        self.enhanced_label.move(950, 10)

        self.contrast_increased_label = QLabel(parent=self)
        self.contrast_increased_label.setText("Contrast Increased")
        self.contrast_increased_label.setFont(QFont('verdana', 10))
        self.contrast_increased_label.move(1350, 10)

        self.gauss_blur_label = QLabel(parent=self)
        self.gauss_blur_label.setText("Gaussian Blur")
        self.gauss_blur_label.setFont(QFont('verdana', 10))
        self.gauss_blur_label.move(150, 440)

        self.canny_edge_label = QLabel(parent=self)
        self.canny_edge_label.setText("Canny Edge Detected")
        self.canny_edge_label.setFont(QFont('verdana', 10))
        self.canny_edge_label.move(550, 440)

        self.all_contours_label = QLabel(parent=self)
        self.all_contours_label.setText("All Contours")
        self.all_contours_label.setFont(QFont('verdana', 10))
        self.all_contours_label.move(950, 440)

        self.final_img_label = QLabel(parent=self)
        self.final_img_label.setText("Original Image With Plate")
        self.final_img_label.setFont(QFont('verdana', 10))
        self.final_img_label.move(1300, 440)

        self.result = QLabel(parent=self)
        self.result.setFont(QFont('verdana', 12))
        self.result.move(650, 890)
        self.result.setGeometry(650, 890, 400, 50)

    def set_image_placeholders(self):
        self.original_image = QLabel(parent=self)
        self.original_image.setGeometry(0, 30, 400, 400)

        self.grayscale = QLabel(parent=self)
        self.grayscale.setGeometry(400, 30, 400, 400)

        self.enhanced = QLabel(parent=self)
        self.enhanced.setGeometry(800, 30, 400, 400)

        self.contrast_increased = QLabel(parent=self)
        self.contrast_increased.setGeometry(1200, 30, 400, 400)

        self.gauss_blur = QLabel(parent=self)
        self.gauss_blur.setGeometry(0, 460, 400, 400)

        self.canny_edge = QLabel(parent=self)
        self.canny_edge.setGeometry(400, 460, 400, 400)

        self.all_contours = QLabel(parent=self)
        self.all_contours.setGeometry(800, 460, 400, 400)

        self.final_img = QLabel(parent=self)
        self.final_img.setGeometry(1200, 460, 400, 400)

    def set_buttons(self):
        self.openImage = QPushButton(text='Open Image To Read', parent=self)
        self.detect_number_plate = QPushButton(text='Detect Number Plate', parent=self)

        self.openImage.setFont(QFont('verdana', 10))
        self.detect_number_plate.setFont(QFont('verdana', 10))

        self.openImage.move(550, 950)
        self.detect_number_plate.move(800, 950)

        self.openImage.clicked.connect(self.open_image_fun)
        self.detect_number_plate.clicked.connect(self.detect_plate)

    def set_image(self, shown_image, placeholder):
        shownImage = cv2.resize(shown_image, (400, 400), interpolation=cv2.INTER_NEAREST)
        shownImage = cv2.cvtColor(shownImage, cv2.COLOR_BGR2RGBA)

        # shownImage = cv2.cvtColor(shownImage, cv2.COLOR_BGR2BGRA)
        shownImage = QtGui.QImage(shownImage, shownImage.shape[1], shownImage.shape[0],
                                  QtGui.QImage.Format_RGB32).rgbSwapped()
        placeholder.setPixmap(QtGui.QPixmap.fromImage(shownImage))
        placeholder.setAlignment(QtCore.Qt.AlignCenter)

    def open_image_fun(self):
        file_name = QFileDialog.getOpenFileName(self, 'Open Image File', r"",
                                                "Image files (*.jpg *.jpeg *.png)")

        if file_name[0] != "":
            self.selectedImage = cv2.imread(file_name[0])
            self.setWindowTitle(file_name[1])
            self.set_image(self.selectedImage, self.original_image)
            self.imageUri = file_name[0]

        self.grayscale.clear()
        self.enhanced.clear()
        self.contrast_increased.clear()
        self.gauss_blur.clear()
        self.canny_edge.clear()
        self.all_contours.clear()
        self.final_img.clear()
        self.result.clear()

    def detect_plate(self):
        image, gray, enhanced_img, weighted, blurred, edged, cnts, final_image, detected_result = NumberPlateDetection.detect_plate(
            self.selectedImage)

        self.set_image(image, self.original_image)
        self.set_image(gray, self.grayscale)
        self.set_image(enhanced_img, self.enhanced)
        self.set_image(weighted, self.contrast_increased)
        self.set_image(blurred, self.gauss_blur)
        self.set_image(edged, self.canny_edge)
        self.set_image(cnts, self.all_contours)
        self.set_image(final_image, self.final_img)

        self.result.setText("Result : " + detected_result)


app = QApplication(sys.argv)
window = Window()
sys.exit(app.exec_())
