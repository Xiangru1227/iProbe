from calUtils import *

def getDiff_robot(iprobeParam, solidCubePrm, idx, folder):
    tipOffset1_fp_list = []
    tipOffset2_fp_list = []
    tipOffset3_fp_list = []
    pyr_list = []
    
    for i in range(idx):
        tipOffset1, tipOffset2, tipOffset3 = process_file_robot(i, iprobeParam, solidCubePrm, folder)
        centroids = np.loadtxt(os.path.join(folder, f"{i+1}.txt"), skiprows=2, max_rows=1).reshape(4, 2)
        pyr = iprobeParam.getPYR(centroids)

        tipOffset1_fp_list.append(tipOffset1)
        tipOffset2_fp_list.append(tipOffset2)
        tipOffset3_fp_list.append(tipOffset3)
        pyr_list.append([pyr.pitch, pyr.yaw, pyr.roll])
    
    tipRef1 = tipOffset1_fp_list[0]
    tipRef2 = tipOffset2_fp_list[0]
    tipRef3 = tipOffset3_fp_list[0]
    
    tipDiff1 = np.linalg.norm(np.array(tipOffset1_fp_list) - tipRef1, axis=1)
    tipDiff2 = np.linalg.norm(np.array(tipOffset2_fp_list) - tipRef2, axis=1)
    tipDiff3 = np.linalg.norm(np.array(tipOffset3_fp_list) - tipRef3, axis=1)

    pyr_array = np.array(pyr_list)

    labels = ['Pitch (°)', 'Yaw (°)', 'Roll (°)']
    tip_labels = ['Tip 1', 'Tip 2', 'Tip 3']
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    
    roll_indices = np.arange(1, 6)
    yaw_indices = np.arange(6, 12)
    pitch_indices = np.arange(12, 18)
    for tip_idx, tipDiff in enumerate([tipDiff1, tipDiff2, tipDiff3]):
        for pyr_idx, indices in enumerate([pitch_indices, yaw_indices, roll_indices]):
            ax = axes[tip_idx, pyr_idx]
            ax.scatter(pyr_array[indices, pyr_idx], tipDiff[indices], alpha=0.7)
            ax.set_xlabel(labels[pyr_idx])
            ax.set_ylabel("Offset Difference (mm)")
            ax.set_title(f"{tip_labels[tip_idx]} vs {labels[pyr_idx]} (Group {indices[0]+1}-{indices[-1]+1})")
    
    # for tip_idx, tipDiff in enumerate([tipDiff1, tipDiff2, tipDiff3]):
    #     for pyr_idx in range(3):
    #         ax = axes[tip_idx, pyr_idx]
    #         ax.scatter(pyr_array[:, pyr_idx], tipDiff, alpha=0.7)
    #         ax.set_xlabel(labels[pyr_idx])
    #         ax.set_ylabel("Offset Difference (mm)")
    #         ax.set_title(f"{tip_labels[tip_idx]} vs {labels[pyr_idx]}")
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    iprobe_prm_input = Prm(65.09435929303649, 74.20233322262244, -78.42560752762888)
    solidCube_prm_input = solidCube(0.4785964150348852, 0.3046415434970332, -0.003916539620741264, 
                                        5.949882180933179, -0.1232472204813352, 0.0029416898060632144)

    getDiff_robot(iprobe_prm_input, solidCube_prm_input, 18, 'iProbeCalibration/geoCalRobot3')