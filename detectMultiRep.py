from detectUtils import *

def main():
    D = 63.5
    H1 = 77.0
    H2 = -77.0
    prm = Prm(D, H1, H2)
    
    _draw = True
    hsv_min = (50, 50, 50)
    hsv_max = (100, 255, 255)
    
    # Modify this to input different datasets
    root_path = '12.23'
    num = 21
    
    list_of_centroids = [[] for _ in range(4)]
    list_of_area = [[] for _ in range(3)]
    list_of_pyrs = [[] for _ in range(4)]
    
    # img_output_path = os.path.join(root_path, 'img_output')
    # os.makedirs(img_output_path, exist_ok=True)
    # cnt_output_path = os.path.join(root_path, 'cnt_output.txt')
    
    for i in range(1, num):
        bgr_img = YUV2BGR(root_path, str(i))
        # print(bgr_img.dtype)
        
        SMR_img = bgr_img.copy()
        SMR_coord, SMR_img = find_red_areas(SMR_img)

        centroids, _ = greenBlobDetection(bgr_img, SMR_coord)
        # centroids, areas, img_contoured = color_detection(bgr_img, hsv_min, hsv_max, SMR_coord, 100, draw=_draw)
        # img_output = os.path.join(img_output_path, f'{i}.png')
        
        # cv2.imwrite(img_output, img_contoured)
        # with open(cnt_output_path, 'a') as f:
        #     f.write(f"Image #{i}\n")
        #     f.write(f"Number of color blocks detected: {len(centroids)}\n")
        #     f.write("Coordinates and areas of centroids (sorted by position from high to low):\n")
        #     for i in range(len(centroids)):
        #         f.write(f"{centroids[i]}\t{areas[i]}\n")
        #     f.write('\n')
        
        for j in range(4):
            list_of_centroids[j].append(centroids[j])
            # list_of_area[j].append(areas[j])
        
        pyrs = iPb_uv2pyr(centroids, prm)
        list_of_pyrs[0].append(pyrs[0])
        list_of_pyrs[1].append(pyrs[1])
        list_of_pyrs[2].append(pyrs[2])
        list_of_pyrs[3].append(pyrs[3])
        
        print(f"processed: {i}/{num - 1}")
    
    std_dev = []
    for row in list_of_centroids:
        std_dev.append((np.std([point[0] for point in row]), np.std([point[1] for point in row])))
    
    # print(root_path)
    
    print("Standard deviation of coordinates (P1, P3, P2a, P2b):")
    for i in range(4):
        print(f"{std_dev[i]}")
    print()
    
    print("Standard deviation of pitch, yaw, roll and scale:")
    for i in range(len(pyrs)):
        print(np.std(list_of_pyrs[i]))
    
if __name__ == "__main__":
    main()
