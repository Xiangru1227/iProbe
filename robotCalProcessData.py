import pandas as pd
import glob
import os

folder_path = "iProbeCalibration/geoCalRobot3/raw"
write_path = "iProbeCalibration/geoCalRobot3"

for i in range(1, 19):
    probe_file = os.path.join(folder_path, f"probe{i}.xlsx")
    probe_df = pd.read_excel(probe_file)
    
    az_el_avg = probe_df[['AZ', 'EL']].mean().tolist()
    xyz_avg = probe_df[['X', 'Y', 'Z']].mean().tolist()
    c_avg = probe_df[['C1x', 'C1y', 'C2x', 'C2y', 'C3x', 'C3y', 'C4x', 'C4y']].mean().tolist()
    
    tip_data = []
    for j in range(1, 4):
        tip_file = os.path.join(folder_path, f"tip{i}_{j}.xlsx")
        tip_df = pd.read_excel(tip_file)
        tip_avg = tip_df[['X', 'Y', 'Z']].mean().tolist()
        tip_data.append("\t".join(map(str, tip_avg)))
        
    output_lines = [
        "\t".join(map(str, az_el_avg)),
        "\t".join(map(str, xyz_avg)),
        "\t".join(map(str, c_avg)),
        "\n".join(tip_data),
    ]
    
    output_file = os.path.join(write_path, f"{i}.txt")
    with open(output_file, "w") as f:
        f.write("\n".join(output_lines))

print("Processing done")
