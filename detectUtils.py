import os
import cv2
import math
import imutils
import numpy as np

class Prm:
    def __init__(self, D, H1, H2):
        self.D = D
        self.H1 = H1
        self.H2 = H2

def YUV2BGR(root_path, sequence):
    imgY_path = os.path.join(root_path, sequence, 'src2Y.png')
    imgU_path = os.path.join(root_path, sequence, 'src2U.png')
    imgV_path = os.path.join(root_path, sequence, 'src2V.png')
    
    imgY = cv2.imread(imgY_path, cv2.IMREAD_UNCHANGED)
    imgU = cv2.imread(imgU_path, cv2.IMREAD_UNCHANGED)
    imgV = cv2.imread(imgV_path, cv2.IMREAD_UNCHANGED)
    
    if imgY is None or imgU is None or imgV is None:
        return -1
    
    height, width = imgY.shape
    imgU = cv2.resize(imgU, (width, height), interpolation=cv2.INTER_LINEAR)
    imgV = cv2.resize(imgV, (width, height), interpolation=cv2.INTER_LINEAR)
    
    yuv_img = cv2.merge([imgY, imgU, imgV])
    bgr_img = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2BGR)
    
    # bgr_img = filter_red(bgr_img)
    
    return bgr_img

def greenBlobDetection(image, refCoord=[1632, 1232]):
    green_channel = image[:, :, 1]
    
    # show_image(green_channel, "green", size=[800,600], position=[500,200])
    # cv2.imwrite("green.png", green_channel)
    
    edge_img = cv2.Canny(green_channel, 80, 180, apertureSize=3, L2gradient=True)
    # show_image(edge_img, "canny", size=[800,600], position=[500,200])
    
    _, binary_image = cv2.threshold(green_channel, 100, 255, cv2.THRESH_BINARY)
    
    # show_image(binary_image, "binary", size=[800,600], position=[500,200])
    # cv2.imwrite("binary.png", binary_image)

    params = cv2.SimpleBlobDetector_Params()
    
    params.filterByArea = True
    params.maxArea = 10000
    params.minArea = 30
    
    params.filterByCircularity = True
    params.maxCircularity = 1
    params.minCircularity = 0.7
    
    params.filterByConvexity = True
    params.maxConvexity = 1
    params.minConvexity = 0.7

    params.filterByInertia = True
    params.minInertiaRatio = 0.5

    params.filterByColor = True
    params.blobColor = 255
    
    params.minThreshold = 150
    params.maxThreshold = 255

    detector = cv2.SimpleBlobDetector_create(params)

    keypoints = detector.detect(binary_image)

    img_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    points = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    
    points = sorted(points, key=lambda x: np.linalg.norm(x - refCoord), reverse=True)
    
    return points, img_with_keypoints
    
def iPb_uv2pyr(keypoints, prm:Prm):
    pyrs = [0.0, 0.0, 0.0, 0.0]
    
    keypoints[2] = [np.mean([keypoints[2][0], keypoints[3][0]]), np.mean([keypoints[2][1], keypoints[3][1]])]
    
    # compute (u1,v1) and (u2,v2), coordinates of P1 and P2 in P3 frame
    u1 = keypoints[0][0] - keypoints[1][0]
    v1 = -(keypoints[0][1] - keypoints[1][1])
    u2 = keypoints[2][0] - keypoints[1][0]
    v2 = -(keypoints[2][1] - keypoints[1][1])

    # compute roll for general cases (in degree)
    roll = (math.atan2((u1 - u2), (v1 - v2))) * 180 / np.pi
    sr = math.sin(roll * np.pi / 180)
    cr = math.cos(roll * np.pi / 180)

    m = math.sqrt(((u1 - u2) ** 2 + (v1 - v2) ** 2) / (prm.H1 - prm.H2) ** 2)
    n = (sr * v1 - cr * u1) / prm.D
    k = (sr * u1 + cr * v1 - prm.H1 * m) / prm.D

    if abs(n) < 0.00001:
        temp1 = m ** 2 + n ** 2 + k ** 2
        temp2 = m ** 2 * n ** 2
        ss = (temp1 - math.sqrt(temp1 ** 2 - 4 * temp2)) / (2 * temp2)
        scale = math.sqrt(ss)
    # when Probe is vertical
    else:
        scale = 1 / math.sqrt(m ** 2 + k ** 2)

    # compute pitch and yaw (in degree)
    yaw = (math.asin(n * scale)) * 180 / np.pi
    pitch = (math.asin(k * scale / math.cos((yaw / 180) * np.pi))) * 180 / np.pi

    pyrs[0] = pitch
    pyrs[1] = yaw
    pyrs[2] = roll
    pyrs[3] = scale
    
    return pyrs

# def color_detection(img, hsv_min, hsv_max, ref_coord, size_ts):    
#     hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     mask = cv2.inRange(hsv_img, hsv_min, hsv_max)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

#     # Find contours in the mask
#     contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     contours = imutils.grab_contours(contours)
    
#     centroids_and_areas = []
#     filtered_contours = []
    
#     for c in contours:
#         # Compute the center of the contour
#         M = cv2.moments(c)
#         # filter out noises
#         if M["m00"] > size_ts and 0.7 <= circularity(c) <= 1.0:
#             cX = M["m10"] / M["m00"]
#             cY = M["m01"] / M["m00"]
            
#             filtered_contours.append(c)
#             area = cv2.contourArea(c)
#             centroids_and_areas.append((np.array([cX, cY]), area))

#     # if len(centroids_and_areas) == 3:
#     centroids_and_areas.sort(key=lambda x: np.linalg.norm(x[0] - ref_coord), reverse=True)
#     centroids = [i[0] for i in centroids_and_areas]
#     areas = [i[1] for i in centroids_and_areas]
    
#     cv2.drawContours(img, filtered_contours, -1, (0, 0, 255), thickness=1)
#     keypoints = [cv2.KeyPoint(x=pt[0], y=pt[1], size=10) for pt in centroids]
#     img = cv2.drawKeypoints(img, keypoints, None, color=(0, 0, 255))
#     return centroids, areas, img
    
def find_red_areas(img):
    smr_img = img.copy()
    hsv_img = cv2.cvtColor(smr_img, cv2.COLOR_BGR2HSV)
    
    mask1 = cv2.inRange(hsv_img, (0, 120, 100), (10, 255, 255))
    mask2 = cv2.inRange(hsv_img, (170, 120, 100), (180, 255, 255))
    mask = cv2.bitwise_or(mask1, mask2)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    
    centroids = []
    selected_contours = []
    
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] > 200:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centroids.append(np.array([cX, cY]))
            selected_contours.append(c)
    
    return centroids[0]

def filter_red(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([30, 255, 255])
    lower_red2 = np.array([160, 50, 50])
    upper_red2 = np.array([190, 255, 255])

    mask1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
    mask = mask1 + mask2

    filtered_img = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask))
    return filtered_img
    
def circularity(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0
    circularity = 4 * np.pi * (area / (perimeter * perimeter))
    return circularity

def show_image(img, name, size, position):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, size[0], size[1])
    cv2.moveWindow(name, position[0], position[1])
    cv2.imshow(name, img)
    cv2.waitKey(0)

def visualize_hsv_range(hsv_min, hsv_max):
    gradient = np.zeros((100, 500, 3), dtype=np.uint8)

    for i in range(gradient.shape[1]):
        ratio = i / gradient.shape[1]
        hsv_color = [
            int(hsv_min[j] * (1 - ratio) + hsv_max[j] * ratio)
            for j in range(3)
        ]
        gradient[:, i, :] = hsv_color

    bgr_gradient = cv2.cvtColor(gradient, cv2.COLOR_HSV2BGR)

    cv2.imshow('HSV Visualization', bgr_gradient)
    cv2.waitKey(0)
    
# def dist2ROI_size(distance, json_file_path):
#     with open(json_file_path) as f:
#         data = json.load(f)
    
#     distances = np.array(data["Distance"])
#     roi_sizes = np.array(data["ROI_size"])
#     led_sizes = np.array(data["LED_size"])
    
#     if distance < distances.min() or distance > distances.max():
#         raise ValueError("Distance is out of the range.")
    
#     idx_1 = np.max(np.where(distances <= distance))
#     idx_2 = np.min(np.where(distances >= distance))
    
#     if idx_1 == idx_2:
#         return roi_sizes[idx_1], led_sizes[idx_1]
#     else:
#         d1, d2 = distance[idx_1], distance[idx_2]
#         roi1, roi2 = roi_sizes[idx_1], roi_sizes[idx_2]
#         led1, led2 = led_sizes[idx_1], led_sizes[idx_2]

#         return (roi2 - roi1) * (distance - d1) / (d2 - d1) + roi1, (led2 - led1) * (distance - d1) / (d2 - d1) + led1
    
# def crop_image(img, center, length):
#     x_start = max(int(center[0] - length), 0)
#     y_start = max(int(center[1] - length), 0)
#     x_end = min(int(center[0] + length), img.shape[1])
#     y_end = min(int(center[1] + length), img.shape[0])

#     cropped_img = img[y_start:y_end, x_start:x_end]

#     return cropped_img



'''Geo Cal funcs'''
