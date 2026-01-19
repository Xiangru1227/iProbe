import os
import numpy as np
import matplotlib.pyplot as plt
from calUtils import *
from scipy.spatial.transform import Rotation as R

def getFrameTransform(centroids, iprobeParam:Prm, solidCubePrm:solidCube, trackerAngle:trackerAngle, tip1_pos, tip2_pos, tip3_pos, smr_pos):
    fpinft, smr_pos, pyr_rect, _ = get_fpinft_robot(centroids, iprobeParam, solidCubePrm, trackerAngle, smr_pos)
    
    fs_z = tip2_pos - tip3_pos
    fs_z = fs_z / np.linalg.norm(fs_z)
    fs_y = np.cross(-fs_z, tip1_pos - tip2_pos)
    fs_y = fs_y / np.linalg.norm(fs_y)
    fs_x = np.cross(fs_y, fs_z)
    fsinft = np.column_stack((fs_x, fs_y, fs_z))
    
    fsinft_inv = fsinft.T
    fpinfs = np.dot(fsinft_inv, fpinft)
    
    return fpinfs, pyr_rect

def process_file(i, iprobeParam, solidCubePrm, folder):
    camera_matrix = np.array([[12113.182017, 0.000000, 1357.831370], 
                              [0.000000, 12131.370018, 1193.464636],
                              [0.000000, 0.000000, 1.000000]])
    dist_coeffs = np.array([0.031260, -3.609172, 0.001807, -0.008724, 68.695971])
    
    file_path = str(i+1) + '.txt'
    with open(os.path.join(folder, file_path), 'r') as f:
        lines = f.readlines()
    tracker_az, tracker_el = [float(x) for x in lines[0].split()]
    smr_pos = np.array([float(x) for x in lines[1].split()])
    centroids = np.array([float(x) for x in lines[2].split()]).reshape(4, 2)
    tip1_pos = np.array([float(x) for x in lines[3].split()])
    tip2_pos = np.array([float(x) for x in lines[4].split()])
    tip3_pos = np.array([float(x) for x in lines[5].split()])
    '''1 is top-left, 2 is top-right, 3 is bottom right'''
    
    tracker_angle = trackerAngle(tracker_az, tracker_el)
    centroids_undistorted = cv2.undistortPoints(centroids, camera_matrix, dist_coeffs)
    centroids_undistorted = cv2.perspectiveTransform(centroids_undistorted, camera_matrix).reshape(4, 2)

    return getFrameTransform(centroids, iprobeParam, solidCubePrm, tracker_angle, tip1_pos, tip2_pos, tip3_pos, smr_pos)

def mat_to_euler_std(fpinfs_list):
    euler_angles = np.array([R.from_matrix(mat).as_euler('xyz', degrees=True) for mat in fpinfs_list])
    std_angles = np.std(euler_angles, axis=0)

    return euler_angles, std_angles

def main():
    file_num = 18
    folder = 'iProbeCalibration/geoCalRobot3'
    iprobe_prm_input = Prm(64.16833680294536, 75.53259048641841, -77.92479728461925)
    solidCube_prm_input = solidCube(0.02938707937765631, 0.37415545856852334, -0.007216594687467497, 
                                    5.499818818409724, -0.0069071979096573775, 0.0008926791868134888)
    list_of_fpinfs = []
    list_of_pyr = []
    for i in range(file_num):
        mat, pyr = process_file(i, iprobe_prm_input, solidCube_prm_input, folder)
        list_of_fpinfs.append(mat)
        list_of_pyr.append(pyr)
        
    # for mat in list_of_fpinfs:
    #     print(mat)
    #     print()
    
    with open("new_iprobe_test/3.4.txt", "a") as f:
        f.write("Math data:\n")
    for i, pyr in enumerate(list_of_pyr):
        print(f"PYR {i+1}: Pitch = {pyr.pitch:.4f} deg\tYaw = {pyr.yaw:.4f} deg\tRoll = {pyr.roll:.4f} deg")
    
        with open("new_iprobe_test/3.4.txt", "a") as f:
            f.write(f"{pyr.pitch:.4f}, \t{pyr.yaw:.4f}, \t{pyr.roll:.4f}\n")
    
    euler_angles, std_angles = mat_to_euler_std(list_of_fpinfs)

    # print("Mean Euler Angles (degrees): [Pitch, Yaw, Roll]")
    # print(np.mean(euler_angles, axis=0))

    # print("\nEuler Angles Standard Deviation (degrees): [Pitch, Yaw, Roll]")
    # print(std_angles)
    
    plt.figure(figsize=(10, 25))

    plt.subplot(3, 2, 1)
    plt.scatter([pyr.pitch for pyr in list_of_pyr], euler_angles[:, 0], label="Pitch (probe in 3 SMR frame)", color="r")
    plt.scatter([pyr.pitch for pyr in list_of_pyr], euler_angles[:, 1], label="Yaw (probe in 3 SMR frame)", color="g")
    plt.scatter([pyr.pitch for pyr in list_of_pyr], euler_angles[:, 2], label="Roll (probe in 3 SMR frame)", color="b")
    plt.xlabel("Pitch (probe in tracker frame)")
    plt.ylabel("Euler Angles pitch (degrees)")
    plt.legend()
    # plt.title("Euler Angles vs. PYR")
    
    plt.subplot(3, 2, 2)
    plt.scatter([pyr.yaw for pyr in list_of_pyr], euler_angles[:, 0], label="Pitch (probe in 3 SMR frame)", color="r")
    plt.scatter([pyr.yaw for pyr in list_of_pyr], euler_angles[:, 1], label="Yaw (probe in 3 SMR frame)", color="g")
    plt.scatter([pyr.yaw for pyr in list_of_pyr], euler_angles[:, 2], label="Roll (probe in 3 SMR frame)", color="b")
    plt.xlabel("Yaw (probe in tracker frame)")
    plt.ylabel("Euler Angles yaw (degrees)")
    plt.legend()
    # plt.title("Euler Angles vs. PYR")
    
    plt.subplot(3, 2, 3)
    plt.scatter([pyr.roll for pyr in list_of_pyr], euler_angles[:, 0], label="Pitch (probe in 3 SMR frame)", color="r")
    plt.scatter([pyr.roll for pyr in list_of_pyr], euler_angles[:, 1], label="Yaw (probe in 3 SMR frame)", color="g")
    plt.scatter([pyr.roll for pyr in list_of_pyr], euler_angles[:, 2], label="Roll (probe in 3 SMR frame)", color="b")
    plt.xlabel("Roll (probe in tracker frame)")
    plt.ylabel("Euler Angles roll (degrees)")
    plt.legend()
    # plt.title("Euler Angles vs. PYR")
    
    plt.subplot(3, 2, 5)
    plt.scatter(np.arange(len(euler_angles)), euler_angles[:, 0], label="Pitch (probe in 3 SMR frame)", color="r")
    plt.scatter(np.arange(len(euler_angles)), euler_angles[:, 1], label="Yaw (probe in 3 SMR frame)", color="g")
    plt.scatter(np.arange(len(euler_angles)), euler_angles[:, 2], label="Roll (probe in 3 SMR frame)", color="b")
    plt.xlabel("Sequence")
    plt.ylabel("Euler Angles (degrees)")
    plt.legend()
    # plt.title("Euler Angles over Sequence")
    
    plt.subplot(3, 2, 6)
    plt.scatter(np.arange(len(euler_angles)), [pyr.pitch for pyr in list_of_pyr], label="Pitch (probe in tracker frame)", color="r")
    plt.scatter(np.arange(len(euler_angles)), [pyr.yaw for pyr in list_of_pyr], label="Yaw (probe in tracker frame)", color="g")
    plt.scatter(np.arange(len(euler_angles)), [pyr.roll for pyr in list_of_pyr], label="Roll (probe in tracker frame)", color="b")
    plt.xlabel("Sequence")
    plt.ylabel("Euler Angles (degrees)")
    plt.legend()
    # plt.title("Probe PYR over Sequence")

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    plt.subplots_adjust(bottom=0.1)
    plt.show()

if __name__ == '__main__':
    main()