from detectUtils import *

def main():
    D = 63.5
    H1 = 77.0
    H2 = -77.0
    prm = Prm(D, H1, H2)
    
    root_path = '12.20'
    sequence = 'straight'
    
    bgr_img = YUV2BGR(root_path, sequence)
    # show_image(bgr_img, "BGR image", size=[800,600], position=[500,200])
    # cv2.imwrite(root_path + "/bgr_img.png", bgr_img)
    
    # bgr_img = cv2.rotate(bgr_img, cv2.ROTATE_180)
    
    '''Find SMR by searching in the image'''
    SMR_coord = find_red_areas(bgr_img)
    
    
    '''Blob detection'''
    img_blobs = bgr_img.copy()
    centroids_blob, img_blobs = greenBlobDetection(img_blobs, SMR_coord)
    # centroids_blob, img_blobs = greenBlobDetection(img_blobs, SMR_coord)
    
    print("Coordinates of BLOB (P1, P3, P2):")
    for c in centroids_blob:
        print(c)
        
    # print(f"{centroids_blob[0][0]}, {centroids_blob[0][1]}")
    
    # cv2.circle(img_blobs, (SMR_coord[0], SMR_coord[1]), radius=5, color=(255, 0, 0), thickness=5)
    # cv2.circle(img_blobs, (int(centroids_blob[2][0]), int(centroids_blob[2][1])), radius=10, color=(255, 0, 0), thickness=10)
    show_image(img_blobs, "Blob detection", size=[800,600], position=[500,200])
    
    # cv2.imwrite(root_path + "/Blob_detection.png", img_blobs)
    
    
    '''Find contour'''
    # hsv_min = (50, 50, 50)
    # hsv_max = (100, 255, 255)
    # # visualize_hsv_range(hsv_min, hsv_max)
    
    # img_contoured = bgr_img.copy()
    # centroids_cnt, area, img_contoured = color_detection(img_contoured, hsv_min, hsv_max, SMR_coord, 100)
    
    # print("Coordinates and areas of CENTROID (P1, P3, P2):")
    # for i in range(len(centroids_cnt)):
    #     print(f"{centroids_cnt[i]}\t{area[i]}")
    
    # cv2.circle(img_contoured, (SMR_coord[0], SMR_coord[1]), radius=10, color=(255, 0, 0), thickness=5)
    # show_image(img_contoured, "Image with contour", size=[800,600], position=[500,200])
    
    # cv2.imwrite(root_path + "/contoured_img.png", img_contoured)
    
    
    '''calculate pyrs'''
    pyrs = iPb_uv2pyr(centroids_blob, prm)
    print("Pitch, Yaw, Roll and Scale:")
    print(pyrs)

if __name__ == "__main__":
    main()
