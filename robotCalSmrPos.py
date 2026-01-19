import os
import numpy as np
from calUtils import *
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares

def getFrameTransform(centroids, iprobeParam:Prm, solidCubePrm:solidCube, trackerAngle:trackerAngle, tip1_pos, tip2_pos, tip3_pos, smr_pos):
    fs_z = tip2_pos - tip3_pos
    fs_z = fs_z / np.linalg.norm(fs_z)
    fs_y = np.cross(-fs_z, tip1_pos - tip2_pos)
    fs_y = fs_y / np.linalg.norm(fs_y)
    fs_x = np.cross(fs_y, fs_z)
    fsinft = np.column_stack((fs_x, fs_y, fs_z))
    
    fbinft = trackerAngle.getFbInFt()
    ftinfb = np.linalg.inv(fbinft)
    fsinfb = np.dot(ftinfb, fsinft)
    
    
    _, _, _, fpinfb = get_fpinft_robot(centroids, iprobeParam, solidCubePrm, trackerAngle, smr_pos)
    
    
    # smr_norm = fsinfb[:, 1]
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
    smr_pos = smr_pos + compFt    
    
    fsinft_inv = fsinft.T
    smr_in_fs = np.dot(fsinft_inv, (smr_pos - tip2_pos))
    
    return smr_in_fs

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
    tracker_angle = trackerAngle(tracker_az, tracker_el)
    centroids_undistorted = cv2.undistortPoints(centroids, camera_matrix, dist_coeffs)
    centroids_undistorted = cv2.perspectiveTransform(centroids_undistorted, camera_matrix).reshape(4, 2)

    return getFrameTransform(centroids_undistorted, iprobeParam, solidCubePrm, tracker_angle, tip1_pos, tip2_pos, tip3_pos, smr_pos)

def getError(solidCubePrm, iprobeParam):
    file_num = 18
    folder = 'iProbeCalibration/geoCalRobot3'
    iprobe_prm_input = iprobeParam
    solidCube_prm_input = solidCube(solidCubePrm[0], 
                                    solidCubePrm[1], 
                                    solidCubePrm[2], 
                                    solidCubePrm[3], 
                                    solidCubePrm[4], 
                                    solidCubePrm[5])
    
    list_of_smrinfs = []
    for i in range(file_num):
        smr = process_file(i, iprobe_prm_input, solidCube_prm_input, folder)
        list_of_smrinfs.append(smr)
        
    return np.std(np.array(list_of_smrinfs), axis=0)

def solidCube_opt(iprobe_prm_input, solidCube_prm_input, tol=1e-7, max_iter=1000):
    prev_error = np.inf
    print(f"Error befor cal: {np.linalg.norm(getError(iprobeParam=iprobe_prm_input, solidCubePrm=solidCube_prm_input))}")
    for iteration in range(max_iter):
        result_solidCube = least_squares(fun=getError, 
                                         x0=solidCube_prm_input, 
                                         args=(iprobe_prm_input, ), 
                                         ftol=tol, 
                                         verbose=0, 
                                         max_nfev=max_iter)

        solidCube_prm_input = result_solidCube.x
        current_error = np.linalg.norm(result_solidCube.fun)
        
        print(f"Iteration {iteration+1}, Error: {current_error:.8f}")
        
        if abs(prev_error - current_error) < tol:
            print("Converged.")
            break
        elif current_error >= prev_error:
            print("Converged due to increasing error.")
            return prev_solidCube_prm, prev_error
        
        prev_error = current_error
        prev_solidCube_prm = solidCube_prm_input
    
    return solidCube_prm_input, current_error

def main():
    iprobe_prm_input = Prm(64.14325663027542, 75.53741941845027, -77.94238277637696)
    solidCube_prm_input = [-2.2266e-03, 6.3928e-02, -5.1277e-05, 
                           5.4682e+00, 7.0481e-04, 4.8222e-04]
    # solidCube_prm_input = [0.036237121115058175, 0.3990319807616971, -0.00807028859567638, 
    #                        5.5066640354616645, -0.009644563529585903, 0.0009676617839816809]
    final_solidCube_prm, final_error = solidCube_opt(iprobe_prm_input, solidCube_prm_input)
    
    print("\nFinal optimized parameters:")
    print(f"SolidCube Parameters: [{final_solidCube_prm[0]}, {final_solidCube_prm[1]}, {final_solidCube_prm[2]}, {final_solidCube_prm[3]}, {final_solidCube_prm[4]}, {final_solidCube_prm[5]}]")
    print("Final Error:", final_error)
    
if __name__ == '__main__':
    main()