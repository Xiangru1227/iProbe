import pandas as pd
import matplotlib.pyplot as plt
from calUtils import *

def test(probeRef, probeData, tipData, iprobePrm, solidCubePrm):
    tipAvg = tipData.iloc[-1]
    tipPos = np.array([tipAvg['X'], tipAvg['Y'], tipAvg['Z']])

    probeAvg = probeRef.iloc[2]
    trkAngleAvg = trackerAngle(probeAvg['AZ'], probeAvg['EL'])
    smrPosAvg = [probeAvg['X'], probeAvg['Y'], probeAvg['Z']]
    centroidsAvg = [[probeAvg['C1x'], probeAvg['C1y']], 
                    [probeAvg['C2x'], probeAvg['C2y']], 
                    [probeAvg['C3x'], probeAvg['C3y']], 
                    [probeAvg['C4x'], probeAvg['C4y']]]
    pyrAvg = iprobePrm.getPYR(centroidsAvg)
    ry = R.from_euler('y', pyrAvg.roll,  degrees=True).as_matrix()
    rx = R.from_euler('x', pyrAvg.pitch, degrees=True).as_matrix()
    rz = R.from_euler('z', pyrAvg.yaw,   degrees=True).as_matrix()
    fpinfcAvg = np.dot(np.dot(ry, rx), rz)

    fbinftAvg = trkAngleAvg.getFbInFt()
    fcinfbAvg = R.from_euler('y', 0.16, degrees=True).as_matrix()
    fpinfbAvg = np.dot(fcinfbAvg, fpinfcAvg)
    fpinftAvg = np.dot(fbinftAvg, fpinfbAvg)

    smr_normAvg = fpinfbAvg[:, 1]
    v_beamAvg = np.array([0, 1, 0])

    tempAvg = np.dot(v_beamAvg, smr_normAvg)
    tempAvg = clamp(tempAvg, -1.0, 1.0)
    combo_angleAvg = np.arccos(tempAvg) * 180 / np.pi

    err_latAvg  = solidCubePrm.a_lat[0]  + solidCubePrm.a_lat[1]  * combo_angleAvg + solidCubePrm.a_lat[2]  * combo_angleAvg ** 2
    err_longAvg = solidCubePrm.a_long[0] + solidCubePrm.a_long[1] * combo_angleAvg + solidCubePrm.a_long[2] * combo_angleAvg ** 2

    vTempAvg = np.cross(smr_normAvg, v_beamAvg)
    vLatInFbAvg = np.cross(v_beamAvg, vTempAvg)

    compFbAvg = err_latAvg * vLatInFbAvg + err_longAvg * np.array([0, -1, 0])
    compFtAvg = np.dot(fbinftAvg, compFbAvg)

    tipOffsetRef = np.linalg.inv(fpinftAvg) @ np.transpose(np.array(tipPos) - np.array(smrPosAvg) - np.array(compFtAvg))
    print(f"tip offset (ref): {tipOffsetRef[0]:.6f}, {tipOffsetRef[1]:.6f}, {tipOffsetRef[2]:.6f}")

    tracker_angle = []
    smr_pos = []
    centroids = []
    pyrs = []
    tipOffsets = []

    for _, row in probeData.iloc[1:-1].iterrows():
        tracker_angle.append(trackerAngle(row['AZ'], row['EL']))
        smr_pos.append([row['X'], row['Y'], row['Z']])
        centroids.append([[row['C1x'], row['C1y']], 
                        [row['C2x'], row['C2y']], 
                        [row['C3x'], row['C3y']], 
                        [row['C4x'], row['C4y']]])
        
    for i in range(len(centroids)):
        pyr = iprobePrm.getPYR(centroids[i])
        pyrs.append(pyr)
        ry = R.from_euler('y', pyr.roll,  degrees=True).as_matrix()
        rx = R.from_euler('x', pyr.pitch, degrees=True).as_matrix()
        rz = R.from_euler('z', pyr.yaw,   degrees=True).as_matrix()
        fpinfc = np.dot(np.dot(ry, rx), rz)
        
        fbinft = tracker_angle[i].getFbInFt()
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
        
        tipOffset = np.linalg.inv(fpinft) @ np.transpose(np.array(tipPos) - np.array(smr_pos[i]) - np.array(compFt))
        tipOffsets.append(tipOffset)

    mean = np.mean(tipOffsets, axis=0)
    print(f"Mean: ({mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f})")
    std = np.std(tipOffsets, axis=0)
    print(f"Std: ({std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f})")
    peak_to_peak = np.ptp(tipOffsets, axis=0)
    print(f"peak_to_peak: ({peak_to_peak[0]:.4f}, {peak_to_peak[1]:.4f}, {peak_to_peak[2]:.4f})")
    meanErr = np.mean(tipOffsets - tipOffsetRef, axis=0)
    print(f"Mean error (with ref): ({meanErr[0]:.6f}, {meanErr[1]:.6f}, {meanErr[2]:.6f})")
        
    pitch_list = [pyr.pitch for pyr in pyrs]
    yaw_list = [pyr.yaw for pyr in pyrs]
    roll_list = [pyr.roll for pyr in pyrs]
    
    Err = [np.sqrt(np.sum(np.square(tipOffset - tipOffsetRef))) for tipOffset in tipOffsets]

    plt.subplot(3, 1, 1)
    plt.scatter(pitch_list, Err)
    plt.xlabel("Pitch")
    plt.ylabel("Error")

    plt.subplot(3, 1, 2)
    plt.scatter(yaw_list, Err)
    plt.xlabel("Yaw")
    plt.ylabel("Error")

    plt.subplot(3, 1, 3)
    plt.scatter(roll_list, Err)
    plt.xlabel("Roll")
    plt.ylabel("Error")

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    probeRef = pd.read_excel('geoCal1/verification/probe.xlsx')
    probeData = pd.read_excel('geoCal1/verification/probe.xlsx')
    tipData = pd.read_excel('geoCal1/verification/tip.xlsx')

    # iprobePrm = Prm(63.5, 77, -77)
    # solidCubePrm = solidCube(0, 0, 0, 0, 0, 0)
    
    iprobePrm = Prm(66.20844612, 74.07940723, -77.77190349)
    solidCubePrm = solidCube(5.75713355e-03, 1.65454347e-01, -2.51927934e-03, 
                             5.47618023e+00, 6.33467621e-02, -4.00017302e-04)
    print('geoCal3:')
    test(probeRef, probeData, tipData, iprobePrm, solidCubePrm)
    
    # iprobePrm = Prm(66.09355661, 74.33674123, -77.60860147)
    # solidCubePrm = solidCube(-2.22466626e-03, 6.39618190e-02, -5.08201694e-04, 
    #                          5.46820210e+00, 7.12379253e-04, 1.75473031e-03)
    # print('geoCal1:')
    # test(probeRef, probeData, tipData, iprobePrm, solidCubePrm)