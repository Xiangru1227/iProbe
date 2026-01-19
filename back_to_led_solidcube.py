import numpy as np
import math
import os
import re
from scipy.spatial.transform import Rotation as R

class Prm:
    def __init__(self, D, H1, H2):
        self.D = D
        self.H1 = H1
        self.H2 = H2
        
class SMR:
    def __init__(self, az, el, x, y, z):
        self.az = az
        self.el = el
        self.x = x
        self.y = y
        self.z = z
        
class Data:
    def __init__(self, smr: SMR, raw_centroid_xy, rvec=None, tvec=None):
        self.smr = smr
        self.centroids = []
        for i in range(len(raw_centroid_xy) // 2):
            self.centroids.append([raw_centroid_xy[2 * i], raw_centroid_xy[2 * i + 1]])
        # Optional extrinsics parsed directly from dump (solvePnP results)
        self.rvec = None if rvec is None else np.asarray(rvec, dtype=np.float64).reshape(3,)
        self.tvec = None if tvec is None else np.asarray(tvec, dtype=np.float64).reshape(3,)
      
def read_ref_smr_from_txt(file_path):
    smrs = []
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines()[:3]]
    data = [list(map(float, line.split(','))) for line in lines]
    for row in data:
        smrs.append(SMR(None, None, row[0], row[1], row[2]))
    return smrs

def build_fsinft_from_three_points(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """
    Build FS frame (FS in FT) from three reference points:
      fs_z = p1 - p2; fs_y = cross(-fs_z, p3 - p2); fs_x = cross(fs_y, fs_z)
    Returns 3x3 fsinft with columns [fs_x, fs_y, fs_z]. All computations in float64.
    """
    p1 = np.asarray(p1, dtype=np.float64).reshape(3,)
    p2 = np.asarray(p2, dtype=np.float64).reshape(3,)
    p3 = np.asarray(p3, dtype=np.float64).reshape(3,)
    
    eps = 1e-12

    z_vec = p2 - p1
    nz = float(np.linalg.norm(z_vec))
    if nz <= eps:
        raise ValueError("Degenerate FS construction: |p1 - p2| too small")
    z_hat = z_vec / nz

    y_vec = np.cross(-z_hat, (p2 - p3))
    ny = float(np.linalg.norm(y_vec))
    if ny <= eps:
        raise ValueError("Degenerate FS construction: y axis norm too small (points nearly colinear)")
    y_hat = y_vec / ny

    x_vec = np.cross(y_hat, z_hat)
    nx = float(np.linalg.norm(x_vec))
    if nx <= eps:
        raise ValueError("Degenerate FS construction: x axis norm too small")
    x_hat = x_vec / nx

    # Re-orthogonalize y to ensure perfect orthonormal right-handed frame
    y_hat = np.cross(z_hat, x_hat)
    y_norm = float(np.linalg.norm(y_hat))
    if y_norm > eps:
        y_hat = y_hat / y_norm

    fsinft = np.column_stack((x_hat, y_hat, z_hat)).astype(np.float64)
    
    return fsinft

def read_iprobe_data_from_dat(file_path):
    data_list = []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        header_line = f.readline().rstrip('\n\r')
        if not header_line:
            raise ValueError("Empty file or missing header")
        header_tokens = header_line.split('\t') if '\t' in header_line else re.split(r'\s+', header_line.strip())
        token_to_idx = {tok: i for i, tok in enumerate(header_tokens)}
        # Build a normalized lookup: remove spaces/underscores/dots and lowercase, e.g., "RVec X" -> "rvecx"
        def _norm(tok: str) -> str:
            return re.sub(r'[^0-9a-zA-Z]+', '', tok).lower()
        norm_to_tok = {_norm(tok): tok for tok in header_tokens}

        def need(tok: str) -> int:
            if tok not in token_to_idx:
                raise KeyError(f"Header missing required column '{tok}'")
            return token_to_idx[tok]

        def need_any(toks):
            for t in toks:
                if t in token_to_idx:
                    return token_to_idx[t]
            raise KeyError(f"Header missing required columns (any of): {toks}")

        # Required columns
        az_idx = need("AZ")
        el_idx = need("EL")
        bx_idx = need_any(["Before Comp. X", "Before Comp X", "Before Comp.X", "BeforeComp X"])
        by_idx = need_any(["Before Comp. Y", "Before Comp Y", "Before Comp.Y", "BeforeComp Y"])
        bz_idx = need_any(["Before Comp Z.", "Before Comp. Z", "Before Comp Z", "BeforeComp Z"])

        # Optional: rvec/tvec columns from MatchedDataDump if available
        def _find_optional_vec(prefix: str):
            # Try normalized keys like rvecx, rvecy, rvecz or tvecx, tvecy, tvecz
            idxs = []
            for axis in ('x', 'y', 'z'):
                key_norm = f"{prefix}{axis}"
                if key_norm in norm_to_tok:
                    tok = norm_to_tok[key_norm]
                    idxs.append(token_to_idx[tok])
                else:
                    idxs.append(None)
            # If any missing, treat as unavailable
            if any(i is None for i in idxs):
                return None
            return tuple(idxs)

        rvec_idx_triplet = _find_optional_vec("rvec")
        tvec_idx_triplet = _find_optional_vec("tvec")

        # Collect centroid indices
        cx_idx = []
        cy_idx = []
        for i in range(1, 18):
            cx_tok = f"CX{i}"
            cy_tok = f"CY{i}"
            if cx_tok in token_to_idx and cy_tok in token_to_idx:
                cx_idx.append(token_to_idx[cx_tok])
                cy_idx.append(token_to_idx[cy_tok])
        if len(cx_idx) != 17 or len(cy_idx) != 17:
            raise KeyError("Header missing some centroid columns CX1..CX17/CY1..CY17")

        # Parse rows
        for line in f:
            s = line.rstrip('\n\r')
            if not s:
                continue
            toks = s.split('\t') if '\t' in s else re.split(r'\s+', s.strip())
            max_needed = max([az_idx, el_idx, bx_idx, by_idx, bz_idx] + cx_idx + cy_idx)
            if len(toks) <= max_needed:
                continue
            try:
                az = float(toks[az_idx])
                el = float(toks[el_idx])
                x = float(toks[bx_idx])
                y = float(toks[by_idx])
                z = float(toks[bz_idx])
                # flatten 17 centroid pairs into 34-length list
                raw_centroid_xy = []
                for k in range(17):
                    raw_centroid_xy.append(float(toks[cx_idx[k]]))
                    raw_centroid_xy.append(float(toks[cy_idx[k]]))
                smr = SMR(az, el, x, y, z)
                # Optional rvec/tvec parsing
                rvec = None
                tvec = None
                if rvec_idx_triplet is not None:
                    try:
                        rvec = [float(toks[rvec_idx_triplet[0]]), float(toks[rvec_idx_triplet[1]]), float(toks[rvec_idx_triplet[2]])]
                    except Exception:
                        rvec = None
                if tvec_idx_triplet is not None:
                    try:
                        tvec = [float(toks[tvec_idx_triplet[0]]), float(toks[tvec_idx_triplet[1]]), float(toks[tvec_idx_triplet[2]])]
                    except Exception:
                        tvec = None
                data_list.append(Data(smr, raw_centroid_xy, rvec=rvec, tvec=tvec))
            except ValueError:
                continue

    if not data_list:
        raise ValueError(f"No valid rows parsed from {file_path}")
    return data_list

def iPb_uv2pyr(keypoints, prm:Prm):
    ypr = [0.0, 0.0, 0.0]
    
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

    ypr[0] = yaw
    ypr[1] = pitch
    ypr[2] = roll
    
    return ypr

def _deg2rad(deg: float) -> float:
    return float(deg) * np.pi / 180.0


def _rotz_deg(angle_deg: float) -> np.ndarray:
    a = _deg2rad(angle_deg)
    cz = float(np.cos(a)); sz = float(np.sin(a))
    return np.array([[cz, -sz, 0.0],
                     [sz,  cz, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def _rotx_deg(angle_deg: float) -> np.ndarray:
    a = _deg2rad(angle_deg)
    cx = float(np.cos(a)); sx = float(np.sin(a))
    return np.array([[1.0, 0.0, 0.0],
                     [0.0,  cx, -sx],
                     [0.0,  sx,  cx]], dtype=np.float64)


def _roty_deg(angle_deg: float) -> np.ndarray:
    a = _deg2rad(angle_deg)
    cy = float(np.cos(a)); sy = float(np.sin(a))
    return np.array([[ cy, 0.0, sy],
                     [0.0, 1.0, 0.0],
                     [-sy, 0.0, cy]], dtype=np.float64)


def _euler_zxy_deg_to_rmat(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Build rotation matrix from zxy Euler (degrees): yaw(z), pitch(x), roll(y)."""
    return R.from_euler('zxy', [yaw_deg, pitch_deg, roll_deg], degrees=True).as_matrix()

def _rmat_to_euler_zxy_deg(Rm: np.ndarray) -> tuple[float, float, float]:
    """Extract zxy Euler angles (degrees) from rotation matrix using SciPy; returns (yaw, pitch, roll)."""
    Rm = np.asarray(Rm, dtype=np.float64)
    yaw, pitch, roll = R.from_matrix(Rm).as_euler('zxy', degrees=True)
    return float(yaw), float(pitch), float(roll)

def process_single_pose(subdir_num, fcinfb, prm, retro_normal_in_fp, solid_cube_param):
    # Locate target path back_to_led/1
    script_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(script_dir, "back_to_led", str(subdir_num))

    # Read three SMR coordinates from Single_Point_Data_*.txt (unchanged logic)
    smr_txt = None
    for name in os.listdir(target_dir):
        if name.startswith("Single_Point_Data_") and name.endswith(".txt"):
            smr_txt = os.path.join(target_dir, name)
            break
    if smr_txt is None:
        raise FileNotFoundError("Single_Point_Data_*.txt not found in back_to_led/1")

    smrs = read_ref_smr_from_txt(smr_txt)
    if len(smrs) < 3:
        raise ValueError("Need three SMR coordinates in Single_Point_Data_*.txt")

    # Extract three SMR positions and compute fsinft
    p1 = np.array([smrs[0].x, smrs[0].y, smrs[0].z], dtype=np.float64)
    p2 = np.array([smrs[1].x, smrs[1].y, smrs[1].z], dtype=np.float64)
    p3 = np.array([smrs[2].x, smrs[2].y, smrs[2].z], dtype=np.float64)

    # print(f"SMR1: ({p1[0]}, {p1[1]}, {p1[2]})")
    # print(f"SMR2: ({p2[0]}, {p2[1]}, {p2[2]})")
    # print(f"SMR3: ({p3[0]}, {p3[1]}, {p3[2]})")

    fsinft = build_fsinft_from_three_points(p1, p2, p3)
    
    ypr_fsinft = _rmat_to_euler_zxy_deg(fsinft)
    # print(f"fsinft YPR (deg): yaw={ypr_fsinft[0]:.6f}, pitch={ypr_fsinft[1]:.6f}, roll={ypr_fsinft[2]:.6f}")

    # Read MatchedDataDump.dat via helper and print first four centroid pairs per row
    mdd_path = os.path.join(target_dir, "MatchedDataDump.dat")
    if not os.path.exists(mdd_path):
        raise FileNotFoundError("MatchedDataDump.dat not found in back_to_led/1")

    data_list = read_iprobe_data_from_dat(mdd_path)
    # Per-row: print first four centroid pairs and compute YPR using 3rd, 4th, 2nd centroids
    iprobe_prm = Prm(prm[0], prm[1], prm[2])
    smr_in_fs_comp_list = []
    for d in data_list:
        c = d.centroids
        if len(c) >= 4:
            # print(f"Centroids: ({c[0][0]}, {c[0][1]}), ({c[1][0]}, {c[1][1]}), ({c[2][0]}, {c[2][1]}), ({c[3][0]}, {c[3][1]})")
            pts4 = [c[0], c[1], c[2], c[3]]
            drop_idx = min(range(4), key=lambda i: pts4[i][1])
            remaining = [pts4[i] for i in range(4) if i != drop_idx]
            keypoints = sorted(remaining, key=lambda p: p[0], reverse=True)
            # print(f"Keypoints: {keypoints}")
            ypr = iPb_uv2pyr(keypoints, iprobe_prm)
            # print(f"fpinfc YPR bef (deg): yaw={ypr[0]:.6f}, pitch={ypr[1]:.6f}, roll={ypr[2]:.6f}")
            fpinfc = _euler_zxy_deg_to_rmat(ypr[0], ypr[1], ypr[2])

            az_deg = d.smr.az
            el_deg = d.smr.el
            Rz = _rotz_deg(float(az_deg))
            Rx = _rotx_deg(float(el_deg))
            fbinft = Rz @ Rx
            
            fpinfb = fcinfb @ fpinfc
            fpinft = fbinft @ fpinfb
            fpinft_pyr = _rmat_to_euler_zxy_deg(fpinft)
            # print(f"fpinft YPR (deg): yaw={fpinft_pyr[0]:.6f}, pitch={fpinft_pyr[1]:.6f}, roll={fpinft_pyr[2]:.6f}")
            
            fpinfs = np.linalg.solve(fsinft, fpinft)
            fpinfs_pyr = _rmat_to_euler_zxy_deg(fpinfs)
            # print(f"fpinfs YPR (deg): yaw={fpinfs_pyr[0]:.6f}, pitch={fpinfs_pyr[1]:.6f}, roll={fpinfs_pyr[2]:.6f}")
            
        smr_before_comp = [d.smr.x, d.smr.y, d.smr.z]
        
        scp = np.asarray(solid_cube_param, dtype=np.float64).reshape(-1,)
        a_lat = scp[0:4]
        a_long = scp[4:8]
        
        smr_norm_fb = fpinfb @ retro_normal_in_fp
        nrm = float(np.linalg.norm(smr_norm_fb))
        if nrm > 0.0:
            smr_norm_fb = smr_norm_fb / nrm
        else:
            raise ValueError("smr_norm_fb norm is zero or negative, cannot normalize.")

        v_beam_fb = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        temp = float(np.clip(float(np.dot(v_beam_fb, smr_norm_fb)), -1.0, 1.0))
        combo_angle_deg = float(np.degrees(np.arccos(temp)))

        err_lat = float(a_lat[0] + a_lat[1] * combo_angle_deg + a_lat[2] * (combo_angle_deg ** 2) + a_lat[3] * (combo_angle_deg ** 3))
        err_long = float(a_long[0] + a_long[1] * combo_angle_deg + a_long[2] * (combo_angle_deg ** 2) + a_long[3] * (combo_angle_deg ** 3))

        v_temp = np.cross(smr_norm_fb, v_beam_fb)
        v_lat_in_fb = np.cross(v_beam_fb, v_temp)
        n = float(np.linalg.norm(v_lat_in_fb))
        if n > 0.0:
            v_lat_in_fb = v_lat_in_fb / n
        else:
            raise ValueError("v_lat_in_fb norm is zero or negative, cannot normalize.")
            
        comp_fb = err_lat * v_lat_in_fb + err_long * np.array([0.0, -1.0, 0.0], dtype=np.float64)
        comp_ft = fbinft @ comp_fb
        smr_xyz_ft_comp = smr_before_comp + comp_ft
        
        smr_in_fs_comp = np.linalg.solve(fsinft, smr_xyz_ft_comp - p2)
        # print(f"smr_in_fs_comp: {smr_in_fs_comp[0]:.6f}, {smr_in_fs_comp[1]:.6f}, {smr_in_fs_comp[2]:.6f}")
        smr_in_fs_comp_list.append(smr_in_fs_comp)
        
    return np.mean(smr_in_fs_comp_list, axis=0)

def get_pyr_list(fcinfb, prm, retro_normal_in_fp, solid_cube_param):
    smr_in_fs_comp_list = []
    for i in range(1, 8):
        smr_in_fs_comp = process_single_pose(str(i), fcinfb, prm, retro_normal_in_fp, solid_cube_param)
        if smr_in_fs_comp is not None:
            smr_in_fs_comp_list.append(smr_in_fs_comp)
            
    if smr_in_fs_comp_list:
        arr = np.asarray(smr_in_fs_comp_list, dtype=np.float64)
        stds = np.std(arr, axis=0)
        ptps = np.ptp(arr, axis=0)
        print(f"SMR_in_FS std: x={stds[0]:.6f}, y={stds[1]:.6f}, z={stds[2]:.6f}")
        print(f"SMR_in_FS ptp: x={ptps[0]:.6f}, y={ptps[1]:.6f}, z={ptps[2]:.6f}")
        return stds
    
def main():
    fcinfb = np.array([[ 9.99601553e-01, -1.57124273e-04, -2.82260604e-02],
                       [ 3.05766202e-04,  9.99986109e-01,  5.26188484e-03],
                       [ 2.82248415e-02, -5.26841884e-03,  9.99587716e-01]], dtype=np.float64)
    prm = [66.9458282000292, 79.5609903283944, -79.14074678394307]
    retro_normal_in_fp = np.array([0, 1, 0], dtype=np.float64)
    solid_cube_param = [0.03337562528583654, -0.03232965420056112, 0.0034968402730924026, -9.89459226909839e-05,
		                9.85132442276245, 0.003432913527927372, -0.0006334825927563309, 2.1592940299138004e-05]
    get_pyr_list(fcinfb, prm, retro_normal_in_fp, solid_cube_param)

if __name__ == "__main__":
    main()


# =============================
# Optimization utilities (new)
# =============================

def _evaluate_smr_across_folders(fcinfb: np.ndarray, prm: list[float],
                                 retro_normal_in_fp: np.ndarray, solid_cube_param: list[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-folder mean smr_in_fs_comp and overall std/ptp across folders 1..7.
    Returns (means_arr[N,3], stds[3], ptps[3]).
    """
    means = []
    for i in range(1, 8):
        mean_xyz = process_single_pose(str(i), fcinfb, prm, retro_normal_in_fp, solid_cube_param)
        if mean_xyz is not None:
            means.append(mean_xyz)
    if not means:
        return np.zeros((0, 3), dtype=np.float64), np.array([np.inf, np.inf, np.inf], dtype=np.float64), np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    means_arr = np.asarray(means, dtype=np.float64)
    stds = np.std(means_arr, axis=0)
    ptps = np.ptp(means_arr, axis=0)
    return means_arr, stds, ptps


def optimize_retro_normal_with_minimize(fcinfb: np.ndarray, prm: list[float], solid_cube_param: list[float],
                                        retro_init: np.ndarray | None = None,
                                        max_tilt_deg: float = 10.0):
    """
    Optimize retro_normal_in_fp using scipy.optimize.minimize.
    Parameterization: spherical angles (theta, phi) to keep unit norm.
    Objective: minimize sum of squared std of smr_in_fs_comp across folders.
    """
    from scipy.optimize import minimize

    def angles_to_unit(theta: float, phi: float) -> np.ndarray:
        ct = float(np.cos(theta))
        st = float(np.sin(theta))
        cp = float(np.cos(phi))
        sp = float(np.sin(phi))
        return np.array([st * cp, ct, st * sp], dtype=np.float64)

    if retro_init is None:
        retro_init = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    retro_init = retro_init / (np.linalg.norm(retro_init) + 1e-12)
    theta0 = float(np.arccos(np.clip(retro_init[1], -1.0, 1.0)))
    phi0 = float(np.arctan2(retro_init[2], retro_init[0]))

    def objective(x: np.ndarray) -> float:
        retro = angles_to_unit(x[0], x[1])
        _, stds, _ = _evaluate_smr_across_folders(fcinfb, prm, retro, solid_cube_param)
        return float(np.sum(stds * stds))

    max_tilt = float(np.deg2rad(max_tilt_deg))
    res = minimize(
        objective,
        x0=np.array([theta0, phi0], dtype=np.float64),
        method="L-BFGS-B",
        bounds=[(0.0, max_tilt), (-np.pi, np.pi)],
        options={"maxiter": 100, "ftol": 1e-12, "gtol": 1e-12, "disp": True},
    )
    retro_opt = angles_to_unit(res.x[0], res.x[1])
    return retro_opt, res


def optimize_solid_cube_with_least_squares(fcinfb: np.ndarray, prm: list[float], retro_normal_in_fp: np.ndarray,
                                           solid_cube_init: list[float]):
    """
    Optimize solid_cube_param using scipy.optimize.least_squares.
    Residuals: per-folder mean smr_in_fs_comp minus overall mean to reduce cross-folder variance.
    """
    from scipy.optimize import least_squares

    def residuals(x: np.ndarray) -> np.ndarray:
        means_arr, _, _ = _evaluate_smr_across_folders(fcinfb, prm, retro_normal_in_fp, x.tolist())
        if means_arr.shape[0] == 0:
            return np.array([1e6, 1e6, 1e6], dtype=np.float64)
        overall_mean = np.mean(means_arr, axis=0)
        return (means_arr - overall_mean).reshape(-1)

    x0 = np.asarray(solid_cube_init, dtype=np.float64)
    solid_cube_bounds = [(-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1),
                         (8.0, 11.0), (-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1)]
    lb = np.array([b[0] for b in solid_cube_bounds], dtype=np.float64)
    ub = np.array([b[1] for b in solid_cube_bounds], dtype=np.float64)
    res = least_squares(
        residuals,
        x0,
        bounds=(lb, ub),
        ftol=1e-9,
        xtol=1e-9,
        gtol=1e-9,
        loss="soft_l1",
        f_scale=0.5,
        max_nfev=500,
        verbose=2,
    )
    return res.x.tolist(), res


def optimize_retro_and_solidcube(fcinfb: np.ndarray, prm: list[float],
                                 retro_init: np.ndarray | None = None, solid_cube_init: list[float] | None = None,
                                 iters: int = 2):
    """
    Alternating optimization:
    1) Fix solid_cube, optimize retro_normal (minimize)
    2) Fix retro_normal, optimize solid_cube (least_squares)
    """
    retro_init = np.array([0.0, 1.0, 0.0], dtype=np.float64) if retro_init is None else np.asarray(retro_init, dtype=np.float64)
    solid_cube_init = [0.0] * 8 if solid_cube_init is None else [float(v) for v in solid_cube_init]

    for _ in range(max(1, int(iters))):
        retro, _ = optimize_retro_normal_with_minimize(fcinfb, prm, solid_cube_init, retro_init)
        solid_cube, _ = optimize_solid_cube_with_least_squares(fcinfb, prm, retro_init, solid_cube_init)

    # Initial report
    _, stds0, ptps0 = _evaluate_smr_across_folders(fcinfb, prm, retro_init, solid_cube_init)
    print("Initial SMR_in_FS std:", stds0)
    print("Initial SMR_in_FS ptp:", ptps0)

    _, stds, ptps = _evaluate_smr_across_folders(fcinfb, prm, retro, solid_cube)
    print("Optimized retro_normal_in_fp:", retro)
    print("Optimized solid_cube_param:", solid_cube)
    print("SMR_in_FS std:", stds)
    print("SMR_in_FS ptp:", ptps)
    return retro, solid_cube, stds, ptps

fcinfb = np.array([[ 9.99601553e-01, -1.57124273e-04, -2.82260604e-02],
                   [ 3.05766202e-04,  9.99986109e-01,  5.26188484e-03],
                   [ 2.82248415e-02, -5.26841884e-03,  9.99587716e-01]], dtype=np.float64)
prm = [66.9458282000292, 79.5609903283944, -79.14074678394307]
retro_normal_in_fp = np.array([0, 1, 0], dtype=np.float64)
solid_cube_param = [0.03337562528583654, -0.03232965420056112, 0.0034968402730924026, -9.89459226909839e-05,
                    9.85132442276245, 0.003432913527927372, -0.0006334825927563309, 2.1592940299138004e-05]
# optimize_retro_and_solidcube(fcinfb, prm, retro_normal_in_fp, solid_cube_param)