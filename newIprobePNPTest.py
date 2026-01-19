import os
import cv2
import copy
import numpy as np
from calUtils import Prm, trackerAngle, PYR, solidCube, clamp
from scipy.spatial.transform import Rotation as R

def process_file_pnp(iprobeParam, camera_matrix, dist_coeffs, i, folder):
    file_path = str(i+1) + '.txt'
    with open(os.path.join(folder, file_path), 'r') as f:
        lines = f.readlines()
    tracker_az, tracker_el = [float(x) for x in lines[0].split()]
    smr_pos = np.array([float(x) for x in lines[1].split()])
    centroids = np.array([float(x) for x in lines[2].split()]).reshape(4, 2)
    tip1_pos = np.array([float(x) for x in lines[3].split()])
    tip2_pos = np.array([float(x) for x in lines[4].split()])
    tip3_pos = np.array([float(x) for x in lines[5].split()])
    centroids_undistorted = cv2.undistortPoints(centroids, camera_matrix, dist_coeffs)
    centroids_undistorted = cv2.perspectiveTransform(centroids_undistorted, camera_matrix).reshape(4, 2)
    
    pyr = iprobeParam.getPYR(centroids_undistorted)
    swap = False
    if -45 < pyr.roll <= 45 and centroids_undistorted[2][0] < centroids_undistorted[3][0]:
        swap = True
    elif 45 < pyr.roll <= 135 and centroids_undistorted[2][1] < centroids_undistorted[3][1]:
        swap = True
    elif -180 < pyr.roll <= -135 or 135 < pyr.roll <= 180:
        if centroids_undistorted[2][0] > centroids_undistorted[3][0]:
            swap = True
    elif -135 < pyr.roll <= -45 and centroids_undistorted[2][1] > centroids_undistorted[3][1]:
        swap = True
    if swap:
        centroids_undistorted[2], centroids_undistorted[3] = copy.deepcopy(centroids_undistorted[3]), copy.deepcopy(centroids_undistorted[2])

    tracker_angle = trackerAngle(tracker_az, tracker_el)
    return centroids_undistorted, tracker_angle, tip1_pos, tip2_pos, tip3_pos, smr_pos

def get_centroids(iprobeParam, camera_matrix, dist_coeffs, idx, folder):
    list_of_centroids = []
    list_of_tracker_angle = []
    list_of_smr = []
    list_of_tip1 = []
    list_of_tip2 = []
    list_of_tip3 = []
    for i in range(idx):
        centroids, tracker_angle, tip1_pos, tip2_pos, tip3_pos, smr_pos = process_file_pnp(iprobeParam, camera_matrix, dist_coeffs, i, folder)
        list_of_centroids.append(centroids)
        list_of_tracker_angle.append(tracker_angle)
        list_of_smr.append(smr_pos)
        list_of_tip1.append(tip1_pos)
        list_of_tip2.append(tip2_pos)
        list_of_tip3.append(tip3_pos)
        
    return list_of_centroids, list_of_tracker_angle, list_of_smr, list_of_tip1, list_of_tip2, list_of_tip3

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

def main():
    folder = 'iProbeCalibration/geoCalRobot3'
    idx = 18
    iprobeParam = Prm(64.16833680294536, 75.53259048641841, -77.92479728461925)
    solidCubePrm = solidCube(0.03399944385623841, 0.3906381271096312, -0.007798730079162356, 
                                5.504853410217059, -0.009268601010029149, 0.0009609381535233711)
    # iprobeParam = Prm(63.5, 77, -77)
    
    camera_matrix = np.array([[12531.015758, 0.000000, 1631.500000],
                              [0.000000, 12537.280590, 1231.500000],
                              [0.000000, 0.000000, 1.000000]])
    dist_coeffs = np.array([0.009151, 0.000000, -0.000824, -0.003733, 0.000000])
    
    centroids_list, tracker_angle_list, smr_list, tip1_list, tip2_list, tip3_list = get_centroids(iprobeParam, camera_matrix, dist_coeffs, idx, folder)
    
    objectPoints = np.array([[0, iprobeParam.D, iprobeParam.H1], 
                             [0, 0, 0], 
                             [ 30, iprobeParam.D, iprobeParam.H2], 
                             [-30, iprobeParam.D, iprobeParam.H2]], 
                             dtype=np.float32)
    # with open("new_iprobe_test/3.4.txt", "a") as f:
    #     f.write("solvePnP data:\n")
    
    tipOffset1_fp_list = []
    tipOffset2_fp_list = []
    tipOffset3_fp_list = []
    
    dist_coeffs = np.zeros((5,1), dtype=np.float32)
    for i, imagePoints in enumerate(centroids_list):
        imagePoints = np.array(imagePoints, dtype=np.float32)
        success, rvec, tvec = cv2.solvePnP(
            objectPoints, imagePoints, camera_matrix, dist_coeffs, 
            flags=cv2.SOLVEPNP_AP3P
        )
        
        if success:
            yaw, pitch, roll = rvec_to_pyr(rvec)
            pyr = PYR(pitch, yaw, roll)
            ry = R.from_euler('y', pyr.roll,  degrees=True).as_matrix()
            rx = R.from_euler('x', pyr.pitch, degrees=True).as_matrix()
            rz = R.from_euler('z', pyr.yaw,   degrees=True).as_matrix()
            fpinfc = np.dot(np.dot(ry, rx), rz)
            fbinft = tracker_angle_list[i].getFbInFt()
            fcinfb = R.from_euler('y', 0.16, degrees=True).as_matrix()
            fpinfb = np.dot(fcinfb, fpinfc)
            fpinft = np.dot(fbinft, fpinfb)
            
            smr_norm = fpinfb[:, 1]
            v_beam = np.array([0, 1, 0])
            temp = np.dot(v_beam, smr_norm)
            temp = clamp(temp, -1.0, 1.0)
            combo_angle = np.arccos(temp) * 180 / np.pi
            err_lat  = solidCubePrm.a_lat[0]  + solidCubePrm.a_lat[1]  * combo_angle + solidCubePrm.a_lat[2]  * combo_angle ** 2
            err_long = solidCubePrm.a_long[0] + solidCubePrm.a_long[1] * combo_angle + solidCubePrm.a_long[2] * combo_angle ** 2
            vTemp = np.cross(smr_norm, v_beam)
            vLatInFb = np.cross(v_beam, vTemp)
            compFb = err_lat * vLatInFb + err_long * np.array([0, -1, 0])
            compFt = np.dot(fbinft, compFb)
            smr_pos = smr_list[i] + compFt
            
            tipOffset1 = tip1_list[i] - smr_pos
            tipOffset2 = tip2_list[i] - smr_pos
            tipOffset3 = tip3_list[i] - smr_pos
            tipOffset1_fp = np.linalg.inv(fpinft) @ np.transpose(tipOffset1)
            tipOffset2_fp = np.linalg.inv(fpinft) @ np.transpose(tipOffset2)
            tipOffset3_fp = np.linalg.inv(fpinft) @ np.transpose(tipOffset3)
            tipOffset1_fp_list.append(tipOffset1_fp)
            tipOffset2_fp_list.append(tipOffset2_fp)
            tipOffset3_fp_list.append(tipOffset3_fp)
            
            # print(f"PYR {i+1}: Pitch = {pitch:.4f} deg\tYaw = {yaw:.4f} deg\tRoll = {roll:.4f} deg")
            # with open("new_iprobe_test/3.4.txt", "a") as f:
            #     f.write(f"{pitch:.4f}, \t{yaw:.4f}, \t{roll:.4f}\n")
        else:
            print(f"PYR {i+1}: solvePnP failed")
            
    for tip in tipOffset1_fp_list:
        print(f"[{tip[0]:.4f},\t{tip[1]:.4f},\t{tip[2]:.4f}]")
    tip_offset_avg = np.mean(tipOffset1_fp_list, axis=0)
    print(f"Average tip offset: [{tip_offset_avg[0]:.4f},\t{tip_offset_avg[1]:.4f},\t{tip_offset_avg[2]:.4f}]\n")
    
    for tip in tipOffset2_fp_list:
        print(f"[{tip[0]:.4f},\t{tip[1]:.4f},\t{tip[2]:.4f}]")
    tip_offset_avg = np.mean(tipOffset2_fp_list, axis=0)
    print(f"Average tip offset: [{tip_offset_avg[0]:.4f},\t{tip_offset_avg[1]:.4f},\t{tip_offset_avg[2]:.4f}]\n")
    
    for tip in tipOffset3_fp_list:
        print(f"[{tip[0]:.4f},\t{tip[1]:.4f},\t{tip[2]:.4f}]")
    tip_offset_avg = np.mean(tipOffset3_fp_list, axis=0)
    print(f"Average tip offset: [{tip_offset_avg[0]:.4f},\t{tip_offset_avg[1]:.4f},\t{tip_offset_avg[2]:.4f}]\n")
    
    offset1_std = np.std(tipOffset1_fp_list, axis=0)
    offset2_std = np.std(tipOffset2_fp_list, axis=0)
    offset3_std = np.std(tipOffset3_fp_list, axis=0)
    
    print(f"scalar error: {np.sqrt(np.sum(offset1_std ** 2 + offset2_std ** 2 + offset3_std ** 2))}")
    print(f"vector error: {np.concatenate([offset1_std, offset2_std, offset3_std])}")
        
if __name__ == '__main__':
    main()