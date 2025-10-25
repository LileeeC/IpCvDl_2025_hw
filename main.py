import sys
import cv2
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog
from interface import Ui_MainWindow

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        self.image1 = None
        self.image2 = None
        
        self.sobel_x = None
        self.sobel_y = None
        self.normalized_combination = None

        self.ui.pushButton_LoadImage1.clicked.connect(self.load_image_1)
        self.ui.pushButton_LoadImage2.clicked.connect(self.load_image_2)
        
        self.ui.pushButton_ColorSeperation.clicked.connect(self.run_q1_1)
        self.ui.pushButton_ColorTransformation.clicked.connect(self.run_q1_2)

        self.ui.pushButton_GaussionBlur.clicked.connect(self.run_q2_1)
        self.ui.pushButton_BilateralFilter.clicked.connect(self.run_q2_2)
        self.ui.pushButton_MedianFilter.clicked.connect(self.run_q2_3)
        
        self.ui.pushButton_SobelX.clicked.connect(self.run_q3_1)
        self.ui.pushButton_SobleY.clicked.connect(self.run_q3_2)
        self.ui.pushButton_ComAndThreshold.clicked.connect(self.run_q3_3)
        self.ui.pushButton_GradientAngle.clicked.connect(self.run_q3_4)
        
        self.ui.pushButton_Transformation.clicked.connect(self.run_q4_1)

        self.ui.pushButton_GlobalThreshold.clicked.connect(self.run_q5_1)
        self.ui.pushButton_LocalThreshold.clicked.connect(self.run_q5_2)

        self.ui.lineEdit_Rotation.setText("0")
        self.ui.lineEdit_Scaling.setText("1.0")
        self.ui.lineEdit_Tx.setText("0")
        self.ui.lineEdit_Ty.setText("0")

    def load_image_1(self):
        # QFileDialog.getOpenFileName
        # "self" is PyQt window
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Image 1", "", "Image Files (*.png *.jpg *.bmp)")
        
        if file_path:
            self.image1 = cv2.imread(file_path)
            if self.image1 is None:
                print(f"Error: Unable to load image from {file_path}")
            else:
                print(f"Loaded Image 1 from: {file_path}")
                # 順便顯示一下載入的影像 (可選)
                cv2.imshow("Loaded Image 1", self.image1)

    def load_image_2(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Image 2", "", "Image Files (*.png *.jpg *.bmp)")
        
        if file_path:
            self.image2 = cv2.imread(file_path)
            if self.image2 is None:
                print(f"Error: Unable to load image from {file_path}")
            else:
                print(f"Loaded Image 2 from: {file_path}")
                cv2.imshow("Loaded Image 2", self.image2)

    def run_q1_1(self):
        if self.image1 is None:
            print("請先載入 Image 1")
            return
            
        # split channels
        b, g, r = cv2.split(self.image1)
        zeros = np.zeros_like(b)

        # single color images
        b_image = cv2.merge([b, zeros, zeros])
        g_image = cv2.merge([zeros, g, zeros])
        r_image = cv2.merge([zeros, zeros, r])

        cv2.imshow("Blue Channel", b_image)
        cv2.imshow("Green Channel", g_image)
        cv2.imshow("Red Channel", r_image)

    def run_q1_2(self):
        if self.image1 is None:
            print("請先載入 Image 1")
            return

        # Q1
        cv_gray = cv2.cvtColor(self.image1, cv2.COLOR_BGR2GRAY)

        # Q2
        b, g, r = cv2.split(self.image1)
        # float for accurate division
        avg_gray_f = (b.astype(float) / 3) + (g.astype(float) / 3) + (r.astype(float) / 3)
        # convert back to np.uint8
        avg_gray = avg_gray_f.astype(np.uint8)

        cv2.imshow("CV_Gray (Perceptual)", cv_gray)
        cv2.imshow("Avg_Gray (Average)", avg_gray)

    def run_q2_1(self):
        if self.image1 is None:
            print("請先載入 Image 1 (image1.jpg)")
            return
        kernel_sizes = (11, 11)
        blur = cv2.GaussianBlur(self.image1, kernel_sizes, 0)
        cv2.imshow("Gaussian Blur (m=5)", blur)

        # def q2_1_trackbar_callback(m):
        #     if m == 0: 
        #         m = 1 # min of m is 1
        #     kernel_size = (2 * m + 1, 2 * m + 1)
        #     # sigmaX = 0 to let OpenCV compute it based on kernel size
        #     blurred_image = cv2.GaussianBlur(self.image1, kernel_size, 0)
        #     cv2.imshow(window_name, blurred_image)

        # # build trackbar, m = 1 to 5
        # cv2.createTrackbar("m (1-5)", window_name, 1, 5, q2_1_trackbar_callback)
        # # initial call
        # q2_1_trackbar_callback(1)

    def run_q2_2(self):
        if self.image1 is None:
            print("請先載入 Image 1 (image1.jpg)")
            return
        
        d = 11 # diameter of each pixel neighborhood
        sigmaColor = 90
        sigmaSpace = 90
        bilateral = cv2.bilateralFilter(self.image1, d, sigmaColor, sigmaSpace)
        cv2.imshow("Bilateral Filter (m=5)", bilateral)

    def run_q2_3(self):
        if self.image2 is None:
            print("請先載入 Image 2 (image2.jpg)")
            return

        ksize = 11
        median = cv2.medianBlur(self.image2, ksize)
        cv2.imshow("Median Filter (m=5)", median)

    def convolve2d(self, image, kernel):
        img_h, img_w = image.shape
        ker_h, ker_w = kernel.shape
        
        pad_h = ker_h // 2
        pad_w = ker_w // 2
        
        # np.pad for zero padding
        padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), 'constant', constant_values=0).astype(np.float64)
        output_image = np.zeros_like(image, dtype=np.float64)
        # perform convolution
        for y in range(img_h):
            for x in range(img_w):
                region = padded_image[y : y + ker_h, x : x + ker_w]
                output_image[y, x] = np.sum(region * kernel)
                
        return output_image

    def run_q3_1(self):
        if self.image1 is None:
            print("請先載入 Image 1 (building.jpg)")
            return

        gray = cv2.cvtColor(self.image1, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        sobel_x_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
        sobel_x_result = self.convolve2d(blur, sobel_x_kernel)
        self.sobel_x = sobel_x_result
        
        sobel_x_display = cv2.normalize(np.abs(sobel_x_result), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cv2.imshow("Sobel X", sobel_x_display)

    def run_q3_2(self):
        if self.image1 is None:
            print("請先載入 Image 1 (building.jpg)")
            return

        gray = cv2.cvtColor(self.image1, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        sobel_y_kernel = np.array([[-1, -2, -1], [ 0,  0,  0], [ 1,  2,  1]], dtype=np.float64)
        sobel_y_result = self.convolve2d(blur, sobel_y_kernel)
        self.sobel_y = sobel_y_result
        
        sobel_y_display = cv2.normalize(np.abs(sobel_y_result), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cv2.imshow("Sobel Y", sobel_y_display)

    def run_q3_3(self):
        if self.sobel_x is None or self.sobel_y is None:
            print("請先執行 3.1 和 3.2")
            return

        # Gradient = sqrt(Gx^2 + Gy^2)
        combination = cv2.magnitude(self.sobel_x, self.sobel_y)
        
        # Normalize to 0-255 
        normalized = cv2.normalize(combination, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        self.normalized_combination = normalized

        _, result_128 = cv2.threshold(normalized, 128, 255, cv2.THRESH_BINARY)
        _, result_28 = cv2.threshold(normalized, 28, 255, cv2.THRESH_BINARY)
        
        cv2.imshow("Combination Normalized", normalized)
        cv2.imshow("Threshold 128", result_128)
        cv2.imshow("Threshold 28", result_28)

    def run_q3_4(self):
        if self.sobel_x is None or self.sobel_y is None or self.normalized_combination is None:
            print("請先依序執行 3.1, 3.2, 3.3")
            return
            
        # gradient angles 
        angles_rad = np.arctan2(self.sobel_y, self.sobel_x)
        # radians -> degrees
        angles_deg = (np.degrees(angles_rad) + 360) % 360
        mask1 = cv2.inRange(angles_deg, 170, 190) 
        mask2 = cv2.inRange(angles_deg, 260, 280)
        
        result1 = cv2.bitwise_and(self.normalized_combination, self.normalized_combination, mask=mask1)
        result2 = cv2.bitwise_and(self.normalized_combination, self.normalized_combination, mask=mask2)
        
        cv2.imshow("Angle 170-190", result1)
        cv2.imshow("Angle 260-280", result2)

    def run_q4_1(self):
        if self.image1 is None:
            print("請先載入 Image 1 (burger.png)")
            return

        img = self.image1
        h, w = img.shape[:2]
        center = (240, 200) 
        
        try:
            # read user inputs
            angle = float(self.ui.lineEdit_Rotation.text())
            scale = float(self.ui.lineEdit_Scaling.text())
            tx = float(self.ui.lineEdit_Tx.text())
            ty = float(self.ui.lineEdit_Ty.text())
            
        except ValueError:
            print("輸入無效或為空，使用範例值: Angle=30, Scale=0.9, Tx=535, Ty=335")
            angle = 30.0  # 逆時針 30 度 
            scale = 0.9   
            tx = 535.0    
            ty = 335.0    

        # get rotation and scaling matrix
        # cv2.getRotationMatrix2D 的 angle 是逆時針
        M = cv2.getRotationMatrix2D(center, angle, scale)
        
        # translation
        M[0, 2] += tx
        M[1, 2] += ty
        result = cv2.warpAffine(img, M, (w, h))
        
        cv2.imshow("Transformed Image", result)

    def run_q5_1(self):
        if self.image1 is None:
            print("請先載入 Image 1 (QR.png)")
            return

        gray = cv2.cvtColor(self.image1, cv2.COLOR_BGR2GRAY)
        _, thresholded_image = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
        
        cv2.imshow("QR Original (Grayscale)", gray)
        cv2.imshow("Global Threshold Result", thresholded_image)

    def run_q5_2(self):
        if self.image1 is None:
            print("請先載入 Image 1 (QR.png)")
            return
        gray = cv2.cvtColor(self.image1, cv2.COLOR_BGR2GRAY)
        
        threshold_image = cv2.adaptiveThreshold(
            gray, 
            255, # maxValue
            cv2.ADAPTIVE_THRESH_MEAN_C, # adaptiveMethod
            cv2.THRESH_BINARY, # thresholdType
            19, # blockSize
            -1  # C
        )
        
        cv2.imshow("QR Original (Grayscale)", gray)
        cv2.imshow("Adaptive Threshold Result", threshold_image)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())