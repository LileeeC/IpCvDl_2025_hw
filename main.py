import cv2
import numpy as np
# from PyQt5 import QtWidgets, QtGui, QtCore
# from PyQt5.QtWidgets import QFileDialog

def q1_1_color_separation(self, image):
    # image is the input BGR image

    # split the channels
    b, g, r = cv2.split(image) [cite: 120]

    # build an all black channel with the same shape as one of the channels
    zeros = np.zeros_like(b) [cite: 131]

    # merge channels to create color images
    b_image = cv2.merge([b, zeros, zeros]) [cite: 127, 135]
    g_image = cv2.merge([zeros, g, zeros]) [cite: 128, 137]
    r_image = cv2.merge([zeros, zeros, r]) [cite: 128, 139]

    # need to check if the images are displayed correctly on ui
    cv2.imshow("Blue Channel", b_image)
    cv2.imshow("Green Channel", g_image)
    cv2.imshow("Red Channel", r_image)

    # return b_image, g_image, r_image

def q1_2_color_transformation(self, image):
    # Q1: 使用 cv2.cvtColor
    cv_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) [cite: 157]

    # Q2: 使用平均法
    b, g, r = cv2.split(image) [cite: 120]

    # convert to float for accurate division
    b_f = b.astype(float)
    g_f = g.astype(float)
    r_f = r.astype(float)
    avg_gray_f = (b_f / 3) + (g_f / 3) + (r_f / 3) [cite: 163]

    # convert back to np.uint8
    avg_gray = avg_gray_f.astype(np.uint8) [cite: 165]

    cv2.imshow("CV_Gray (Perceptual)", cv_gray) [cite: 176]
    cv2.imshow("Avg_Gray (Average)", avg_gray) [cite: 178]

# self.image1 is loaded from the ui button
# self.image1 = cv2.imread(...) 
def q2_1_gaussian_blur(self):
    if not hasattr(self, 'image1') or self.image1 is None:
        print("請先載入 Image 1")
        return

    window_name = "Gaussian Blur"
    cv2.namedWindow(window_name)

    # build trackbar，range m = 1~5 
    # min value of trackbar is 0, we will treat 0 as 1 in the callback
    cv2.createTrackbar("m (1-5)", window_name, 1, 5, self.q2_1_trackbar_callback) [cite: 338]

    # call the callback once to initialize
    self.q2_1_trackbar_callback(1)

def q2_1_trackbar_callback(self, m):
    if m == 0:
        m = 1

    kernel_size = (2 * m + 1, 2 * m + 1)
    blurred_image = cv2.GaussianBlur(self.image1, kernel_size, 0) # sigma X=0 means auto calculate sigma based on kernel size [cite: 333, 334]

    cv2.imshow("Gaussian Blur", blurred_image) [cite: 339]


def q2_2_bilateral_filter(self):
    if not hasattr(self, 'image1') or self.image1 is None:
        print("請先載入 Image 1")
        return

    window_name = "Bilateral Filter"
    cv2.namedWindow(window_name)

    # build trackbar，range m = 1~5
    cv2.createTrackbar("m (1-5)", window_name, 1, 5, self.q2_2_trackbar_callback)

    # call the callback once to initialize
    self.q2_2_trackbar_callback(1)

def q2_2_trackbar_callback(self, m):
    if m == 0:
        m = 1

    # d is the diameter of each pixel neighborhood
    d = 2 * m + 1

    sigma_color = 90
    sigma_space = 90
    bilateral_image = cv2.bilateralFilter(self.image1, d, sigma_color, sigma_space) [cite: 360]

    cv2.imshow("Bilateral Filter", bilateral_image)


def q2_3_median_filter(self):
    if not hasattr(self, 'image2') or self.image2 is None:
        print("請先載入 Image 2")
        return

    window_name = "Median Filter"
    cv2.namedWindow(window_name)
    cv2.createTrackbar("m (1-5)", window_name, 1, 5, self.q2_3_trackbar_callback)

    self.q2_3_trackbar_callback(1)

def q2_3_trackbar_callback(self, m):
    if m == 0:
        m = 1

    # ksize has to be odd and greater than 1
    ksize = 2 * m + 1
    median_image = cv2.medianBlur(self.image2, ksize)

    cv2.imshow("Median Filter", median_image)

# 2d convolution for q3
def manual_convolve2d(image, kernel):
    # size of image and kernel
    img_h, img_w = image.shape
    ker_h, ker_w = kernel.shape

    pad_h = ker_h // 2
    pad_w = ker_w // 2

    padded_image = np.zeros((img_h + 2 * pad_h, img_w + 2 * pad_w), dtype=np.float64)
    padded_image[pad_h:pad_h + img_h, pad_w:pad_w + img_w] = image.astype(np.float64)

    output_image = np.zeros_like(image, dtype=np.float64)

    for y in range(img_h):
        for x in range(img_w):
            # extract the region of interest 
            region = padded_image[y : y + ker_h, x : x + ker_w]
            # element-wise multiplication and sum
            output_image[y, x] = np.sum(region * kernel)

    return output_image

def q3_1_sobel_x(self, image):
    # image is "building.jpg" 

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    sobel_x_kernel = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float64) 

    sobel_x_result = manual_convolve2d(blur, sobel_x_kernel)

    # abs and normalize for display
    sobel_x_display = cv2.normalize(np.abs(sobel_x_result), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imshow("Sobel X", sobel_x_display)
    self.sobel_x = sobel_x_result

def q3_2_sobel_y(self, image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) [cite: 476]
    blur = cv2.GaussianBlur(gray, (3, 3), 0) [cite: 491]
    sobel_y_kernel = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=np.float64) 
    sobel_y_result = manual_convolve2d(blur, sobel_y_kernel)

    sobel_y_display = cv2.normalize(np.abs(sobel_y_result), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imshow("Sobel Y", sobel_y_display) 
    self.sobel_y = sobel_y_result

def q3_3_combination(self):
    if not hasattr(self, 'sobel_x') or not hasattr(self, 'sobel_y'):
        print("請先執行 3.1 Sobel X 和 3.2 Sobel Y")
        return

    # 1. 計算梯度幅度 G = sqrt(Gx^2 + Gy^2) 
    # 我們可以使用 cv2.magnitude() 或 numpy
    # combination = np.sqrt(self.sobel_x**2 + self.sobel_y**2)
    combination = cv2.magnitude(self.sobel_x, self.sobel_y)
    normalized = cv2.normalize(combination, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U) 
    self.normalized_combination = normalized

    thresh_val_1 = 128 
    max_val = 255
    _, result_128 = cv2.threshold(normalized, thresh_val_1, max_val, cv2.THRESH_BINARY)

    thresh_val_2 = 28
    _, result_28 = cv2.threshold(normalized, thresh_val_2, max_val, cv2.THRESH_BINARY)

    cv2.imshow("Combination Normalized", normalized)
    cv2.imshow("Threshold 128", result_128) 
    cv2.imshow("Threshold 28", result_28) 

def q3_4_gradient_angle(self):
    if not all(hasattr(self, attr) for attr in ['sobel_x', 'sobel_y', 'normalized_combination']):
        print("請先依序執行 3.1, 3.2, 3.3")
        return

    angles_rad = np.arctan2(self.sobel_y, self.sobel_x) [cite: 551]
    # convert radians to degrees and normalize to [0, 360)
    angles_deg = (np.degrees(angles_rad) + 360) % 360
    mask1 = cv2.inRange(angles_deg, 170, 190)
    mask2 = cv2.inRange(angles_deg, 260, 280)

    result1 = cv2.bitwise_and(self.normalized_combination, self.normalized_combination, mask=mask1) [cite: 556]
    result2 = cv2.bitwise_and(self.normalized_combination, self.normalized_combination, mask=mask2)

    cv2.imshow("Angle 170-190", result1) 
    cv2.imshow("Angle 260-280", result2) 


def q4_transforms(self):
    if not hasattr(self, 'burger_image') or self.burger_image is None:
        # 假設 'Load Image 1' 載入 burger.png
        if hasattr(self, 'image1') and self.image1 is not None:
            self.burger_image = self.image1
        else:
            print("請先載入 burger.png")
            return

    img = self.burger_image
    h, w = img.shape[:2] [cite: 665]
    center = (240, 200) 

    # 1. 從你的 PyQt UI 讀取值
    # 這裡使用 PDF 上的預設值作為範例
    try:
        # 假設你的 UI 元件名稱為 self.ui.rotation_input, ...
        # angle = float(self.ui.rotation_input.text())
        # scale = float(self.ui.scaling_input.text())
        # tx = float(self.ui.tx_input.text())
        # ty = float(self.ui.ty_input.text())

        # 範例值：
        angle = 30.0  # 逆時針 30 度 
        scale = 0.9   # 
        tx = 535.0    # 
        ty = 335.0    # 

    except ValueError:
        print("請輸入有效的數字")
        return

    # get matrix for rotation and scaling
    # positive angle 表示逆時針旋轉
    M_rs = cv2.getRotationMatrix2D(center, angle, scale)

    M_rs[0, 2] += tx [cite: 669]
    M_rs[1, 2] += ty [cite: 670]
    result = cv2.warpAffine(img, M_rs, (w, h)) [cite: 659]

    cv2.imshow("Transformed Burger", result)

def q5_1_global_threshold(self, image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thresh_val = 80
    max_val = 255
    _, thresholded_image = cv2.threshold(gray, thresh_val, max_val, cv2.THRESH_BINARY) [cite: 775]

    cv2.imshow("QR Original (Grayscale)", gray)
    cv2.imshow("Global Threshold Result", thresholded_image) 

def q5_2_local_threshold(self, image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    max_val = 255 
    block_size = 19 
    C = -1

    adaptive_image = cv2.adaptiveThreshold(
        gray, 
        max_val, 
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, 
        block_size, 
        C
    ) 

    cv2.imshow("QR Original (Grayscale)", gray)
    cv2.imshow("Adaptive Threshold Result", adaptive_image)