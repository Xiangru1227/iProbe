import os
import cv2
import json
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R

def YUV2BGR(root_path, sequence):
    imgY_path = os.path.join(root_path, sequence, 'src1Y.png')
    imgU_path = os.path.join(root_path, sequence, 'src1U.png')
    imgV_path = os.path.join(root_path, sequence, 'src1V.png')
    
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
    
    return bgr_img

def show_image(img, name, size, position):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, size[0], size[1])
    cv2.moveWindow(name, position[0], position[1])
    cv2.imshow(name, img)
    # cv2.imwrite("new_iprobe_test/9.5m.png", img)
    cv2.waitKey(0)

def show_image_with_points(img, points, title, color=(0, 255, 255)):
    img_copy = img.copy()

    for i, (x, y) in enumerate(points):
        cv2.circle(img_copy, (int(x), int(y)), 5, color, -1)
        cv2.putText(img_copy, str(i), (int(x) + 20, int(y) - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    show_image(img_copy, title, size=[800,600], position=[500,200])
    # cv2.imwrite(f"new_iprobe_test/{title}.png", img_copy)
    
# def greenBlobDetection(image):
#     green_channel = image[:, :, 1]
#     _, binary_image = cv2.threshold(green_channel, 100, 255, cv2.THRESH_BINARY)
#     # show_image(binary_image, "BGR image", size=[800,600], position=[500,200])
    
#     params = cv2.SimpleBlobDetector_Params()
    
#     params.filterByArea = True
#     params.maxArea = 5000
#     params.minArea = 50
    
#     params.filterByCircularity = True
#     params.maxCircularity = 1
#     params.minCircularity = 0.3
    
#     params.filterByConvexity = True
#     params.maxConvexity = 1
#     params.minConvexity = 0.7

#     params.filterByInertia = True
#     params.maxInertiaRatio = 1
#     params.minInertiaRatio = 0.3

#     params.filterByColor = True
#     params.blobColor = 255
    
#     params.minThreshold = 200
#     params.maxThreshold = 255

#     detector = cv2.SimpleBlobDetector_create(params)
#     keypoints = detector.detect(binary_image)
    
#     if keypoints:
#         sizes = np.array([kp.size for kp in keypoints])
#         median_size = np.median(sizes)
#         threshold_size = 1.5 * median_size
        
#         filtered_keypoints = [kp for kp in keypoints if kp.size < threshold_size]
#     else:
#         filtered_keypoints = []
    
#     img_with_keypoints = cv2.drawKeypoints(image, filtered_keypoints, np.array([]), (0, 0, 255),
#                                           cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
#     # points = np.array([kp.pt for kp in keypoints], dtype=np.float32)
#     points = np.array([kp.pt for kp in filtered_keypoints], dtype=np.float32)
    
#     return points, img_with_keypoints

def greenBlobDetection(image):
    green_channel = image[:, :, 1]

    blurred = cv2.GaussianBlur(green_channel, (5, 5), 0)
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
    
    kernel = np.ones((5, 5), np.uint8)
    processed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = filter_contours(contours)
    # show_image(processed, "BGR image", size=[800,600], position=[500,200])

    centroids = []
    img_with_contours = image.copy()

    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            centroids.append((cx, cy))
            
        cv2.drawContours(img_with_contours, [contour], -1, (0, 0, 255), 1)

    centroids = np.array(centroids, dtype=np.float32)
    # show_image(img_with_contours, "img_with_contours", size=[800,600], position=[500,200])

    return centroids, img_with_contours

def filter_contours(contours, min_area=200, max_area=5000, min_circularity=0.8, max_circularity=1.0,
                    min_convexity=0.7, max_convexity=1.0, min_inertia_ratio=0.3):
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not (min_area <= area <= max_area):
            continue
        
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        if not (min_circularity <= circularity <= max_circularity):
            continue
        
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        convexity = area / hull_area
        if not (min_convexity <= convexity <= max_convexity):
            continue
        
        _, (w, h), _ = cv2.minAreaRect(contour)
        inertia_ratio = min(w, h) / max(w, h) if max(w, h) > 0 else 0
        if inertia_ratio < min_inertia_ratio:
            continue
        
        filtered_contours.append(contour)

    return filtered_contours

def get_smr_coord(distance, camera_matrix):
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    
    with open("cam_calibration.json", "r") as file:
        data = json.load(file)
    
    distances = np.array(data["Parallax"]["Distance"])
    azimuths = np.array(data["Parallax"]["X"])
    elevations = np.array(data["Parallax"]["Y"])
    
    interp_az = interp1d(distances, azimuths, kind='linear', fill_value="extrapolate")
    interp_el = interp1d(distances, elevations, kind='linear', fill_value="extrapolate")
    
    az = interp_az(distance)
    el = interp_el(distance)
    
    x = fx * np.tan(az) + cx
    y = cy - fy * np.tan(el)
    
    return (x, y)

def find_p23(smr_coord, p1, centroids):
    if len(centroids) == 0:
        print("No remaining centroids.")
        return None, None
    
    v = np.array(p1) - np.array(smr_coord)
    v_norm = np.linalg.norm(v)

    left_points = []
    right_points = []
    left_distances = []
    right_distances = []

    for point in centroids:
        point = np.array(point)
        
        perpendicular_dist = np.abs((point[0] - smr_coord[0]) * v[1] - (point[1] - smr_coord[1]) * v[0]) / v_norm
        cross_product = (point[0] - smr_coord[0]) * v[1] - (point[1] - smr_coord[1]) * v[0]

        if cross_product > 0:
            left_points.append(point)
            left_distances.append(perpendicular_dist)
        elif cross_product < 0:
            right_points.append(point)
            right_distances.append(perpendicular_dist)
    
    p2 = tuple(left_points[np.argmax(left_distances)]) if left_points else None
    p3 = tuple(right_points[np.argmax(right_distances)]) if right_points else None

    return p2, p3

def find_nearest_n_points(ref_point, centroids, n=3):
    distances = np.linalg.norm(centroids - np.array(ref_point), axis=1)
    nearest_indices = np.argsort(distances)[:n]

    for i in nearest_indices:
        yield centroids[i]

def rvec_to_pyr(rvec):
    Rot, _ = cv2.Rodrigues(rvec)
    R_transform = np.array([
        [1,  0,  0],  
        [0,  0,  1],  
        [0, -1,  0]   
    ], dtype=np.float32)
    Rot = R_transform @ Rot
    r = R.from_matrix(Rot)

    return r.as_euler('zxy', degrees=True)

def process_detection(root_path, sequence, distance):
    camera_matrix = np.array([[1.22762762e+04, 0.00000000e+00, 1.64486041e+03],
                              [0.00000000e+00, 1.22767125e+04, 1.11097133e+03],
                              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    
    dist_coeffs = np.array([-0.08800051, -0.48058723, -0.00252425, -0.00254014, 0])
    
    objectPoints = np.array([[0,	    54,	    88],
                             [0,	    54,	    -88],
                             [-88,	    54,	    0],
                             [88,	    54,	    0],
                             [0,	    42.76,	62.41],
                             [0,	    26.89,	43.09],
                             [0,	    11.01,	23.78],
                             [0,	    42.76,	-62.41],
                             [0,	    26.89,	-43.09],
                             [0,	    11.01,	-23.78],
                             [-62.41,	42.76,	0],
                             [-43.09,	26.89,	0],
                             [-23.78,	11.01,	0],
                             [62.41,	42.76,	0],
                             [43.09,	26.89,	0],
                             [23.78,	11.01,	0],
                             [0,	    0,	    0],
                            ])
    
    bgr_img = YUV2BGR(root_path, sequence)
    # show_image(bgr_img, "BGR image", size=[800,600], position=[500,200])
    centroids_blob, img_blobs = greenBlobDetection(bgr_img)
    
    smr_coord = get_smr_coord(distance / 1000, camera_matrix)
    cv2.circle(img_blobs, (int(smr_coord[0]), int(smr_coord[1])), 10, (0, 0, 255), -1)
    
    '''get p1(top)'''
    centroids_sorted = []
    coord_distances = np.linalg.norm(centroids_blob - np.array(smr_coord), axis=1)
    p1_index = np.argmax(coord_distances)
    p1 = tuple(centroids_blob[p1_index])
    centroids_blob = np.delete(centroids_blob, p1_index, axis=0)
    centroids_sorted.append(p1)
    
    '''get p2(bot)'''
    coord_distances = np.linalg.norm(centroids_blob - np.array(smr_coord), axis=1)
    p2_index = np.argmin(coord_distances)
    p2 = tuple(centroids_blob[p2_index])
    centroids_blob = np.delete(centroids_blob, p2_index, axis=0)
    centroids_sorted.append(p2)
    
    '''get p2p3(left,right)'''
    p3, p4 = find_p23(smr_coord, p1, centroids_blob)
    centroids_blob = np.array([p for p in centroids_blob if not np.allclose(p, p3) and not np.allclose(p, p4)])
    centroids_sorted.append(p3)
    centroids_sorted.append(p4)
    
    for p in np.array([p1, p2, p3, p4]):
        for nearest_point in find_nearest_n_points(p, centroids_blob, n=3):
            centroids_sorted.append(nearest_point)
            centroids_blob = np.array([p for p in centroids_blob if not np.allclose(p, nearest_point)])
    
    if len(centroids_blob) == 1:
        centroids_sorted.append(centroids_blob[0])
    
    show_image_with_points(img_blobs, centroids_sorted, 'test', color=(0, 255, 255))
    # show_image(img_blobs, "Blob detection", size=[800,600], position=[500,200])
    
    imagePoints = np.array(centroids_sorted)
    success, rvec, _ = cv2.solvePnP(objectPoints, imagePoints, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if success:
        yaw, pitch, roll = rvec_to_pyr(rvec)
    
    return yaw, pitch, roll, centroids_sorted

def main():
    root_path = 'new_iprobe_test/4.8'
    # sequence = ''
    distance = 4127
    
    list_yaw = []
    list_pitch = []
    list_roll = []
    list_img_pnts = []
    
    for i in range(9):
        sequence = str(i+1)
        yaw, pitch, roll, image_points = process_detection(root_path, sequence, distance)
        
        list_yaw.append(yaw)
        list_pitch.append(pitch)
        list_roll.append(roll)
        list_img_pnts.append(image_points)
            
    # print(f"points rep: \n{np.std(list_img_pnts, axis=0)}")
    print(f"\nmean (deg) -> pitch: {np.mean(list_pitch):.4f}, yaw: {np.mean(list_yaw):.4f}, roll: {np.mean(list_roll):.4f}")
    print(f"repeatability (deg) -> pitch: {np.std(list_pitch):.4f}, yaw: {np.std(list_yaw):.4f}, roll: {np.std(list_roll):.4f}")
    
if __name__ == '__main__':
    main()