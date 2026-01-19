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

def process_single_pose(subdir_num, fcinfb, prm):
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
    fpinft_pyr_accum = []
    fpinfs_pyr_accum = []
    for d in data_list:
        c = d.centroids
        if len(c) >= 4:
            # print(f"Centroids: ({c[0][0]}, {c[0][1]}), ({c[1][0]}, {c[1][1]}), ({c[2][0]}, {c[2][1]}), ({c[3][0]}, {c[3][1]})")
            # Select first four, drop the one with the smallest y, then sort remaining by x desc
            pts4 = [c[0], c[1], c[2], c[3]]
            drop_idx = min(range(4), key=lambda i: pts4[i][1])
            remaining = [pts4[i] for i in range(4) if i != drop_idx]
            keypoints = sorted(remaining, key=lambda p: p[0], reverse=True)
            # print(f"Keypoints: {keypoints}")
            ypr = iPb_uv2pyr(keypoints, iprobe_prm)  # [yaw, pitch, roll] in degrees
            # print(f"fpinfc YPR bef (deg): yaw={ypr[0]:.6f}, pitch={ypr[1]:.6f}, roll={ypr[2]:.6f}")

            # Build fpinfc from YPR (zxy Euler), then apply R_transform to get true fpinfc
            fpinfc = _euler_zxy_deg_to_rmat(ypr[0], ypr[1], ypr[2])
            # fpinfc_pyr = _rmat_to_euler_zxy_deg(fpinfc)
            # print(f"fpinfc YPR aft (deg): yaw={fpinfc_pyr[0]:.6f}, pitch={fpinfc_pyr[1]:.6f}, roll={fpinfc_pyr[2]:.6f}")
            # R_transform = np.array([[1, 0, 0],
            #                         [0, 0, 1],
            #                         [0, -1, 0]], dtype=np.float64)
            # fpinfc = R_transform @ fpinfc

            # Build fbinft from az/el (from dump)
            az_deg = d.smr.az
            el_deg = d.smr.el
            Rz = _rotz_deg(float(az_deg))
            Rx = _rotx_deg(float(el_deg))
            fbinft = Rz @ Rx
            
            fpinfb = fcinfb @ fpinfc
            fpinft = fbinft @ fpinfb
            fpinft_pyr = _rmat_to_euler_zxy_deg(fpinft)
            # print(f"fpinft YPR (deg): yaw={fpinft_pyr[0]:.6f}, pitch={fpinft_pyr[1]:.6f}, roll={fpinft_pyr[2]:.6f}")
            fpinft_pyr_accum.append([fpinft_pyr[0], fpinft_pyr[1], fpinft_pyr[2]])
            
            fpinfs = np.linalg.solve(fsinft, fpinft)
            fpinfs_pyr = _rmat_to_euler_zxy_deg(fpinfs)
            # print(f"fpinfs YPR (deg): yaw={fpinfs_pyr[0]:.6f}, pitch={fpinfs_pyr[1]:.6f}, roll={fpinfs_pyr[2]:.6f}")
            fpinfs_pyr_accum.append([fpinfs_pyr[0], fpinfs_pyr[1], fpinfs_pyr[2]])

            # # Print YPR for fpinfc and fbinft
            # ypr_fpinfc = _rmat_to_euler_zxy_deg(fpinfc)
            # ypr_fbinft = _rmat_to_euler_zxy_deg(fbinft)
            # print(f"fpinfc YPR (deg): yaw={ypr_fpinfc[0]:.6f}, pitch={ypr_fpinfc[1]:.6f}, roll={ypr_fpinfc[2]:.6f}")
            # print(f"fbinft YPR (deg): yaw={ypr_fbinft[0]:.6f}, pitch={ypr_fbinft[1]:.6f}, roll={ypr_fbinft[2]:.6f}")

    if fpinft_pyr_accum:
        arr = np.asarray(fpinft_pyr_accum, dtype=np.float64)
        fpinft_pyr_means = np.mean(arr, axis=0)
        # print(f"fpinft YPR mean (deg): yaw={fpinft_pyr_means[0]:.6f}, pitch={fpinft_pyr_means[1]:.6f}, roll={fpinft_pyr_means[2]:.6f}")
        
    # Aggregate and print statistics for fpinft YPR
    if fpinfs_pyr_accum:
        arr = np.asarray(fpinfs_pyr_accum, dtype=np.float64)
        fpinfs_pyr_means = np.mean(arr, axis=0)
        # fpinfs_pyr_stds = np.std(arr, axis=0)
        # fpinfs_pyr_ptps = np.ptp(arr, axis=0)
        # print(f"fpinfs YPR mean (deg): yaw={fpinfs_pyr_means[0]:.6f}, pitch={fpinfs_pyr_means[1]:.6f}, roll={fpinfs_pyr_means[2]:.6f}")
        # print(f"fpinfs YPR std  (deg): yaw={fpinfs_pyr_stds[0]:.6f}, pitch={fpinfs_pyr_stds[1]:.6f}, roll={fpinfs_pyr_stds[2]:.6f}")
        # print(f"fpinfs YPR ptp  (deg): yaw={fpinfs_pyr_ptps[0]:.6f}, pitch={fpinfs_pyr_ptps[1]:.6f}, roll={fpinfs_pyr_ptps[2]:.6f}")
        
        return fpinfs_pyr_means
    

def get_pyr_list(fcinfb, prm):
    ypr_list = []
    for i in range(1, 8):
        ypr = process_single_pose(str(i), fcinfb, prm)
        if ypr is not None:
            ypr_list.append(ypr)
            # print(f"{i}: yaw={ypr[0]:.6f}, pitch={ypr[1]:.6f}, roll={ypr[2]:.6f}")
            
    if ypr_list:
        arr = np.asarray(ypr_list, dtype=np.float64)
        stds = np.std(arr, axis=0)
        ptps = np.ptp(arr, axis=0)
        print(f"YPR std (deg): yaw={stds[0]:.6f}, pitch={stds[1]:.6f}, roll={stds[2]:.6f}")
        print(f"YPR ptp (deg): yaw={ptps[0]:.6f}, pitch={ptps[1]:.6f}, roll={ptps[2]:.6f}")
        return stds
    
def main():
    fcinfb = np.array([[ 9.99601368e-01, -1.84102328e-04, -2.82324569e-02],
                       [ 3.32774427e-04,  9.99986103e-01,  5.26139129e-03],
                       [ 2.82310959e-02, -5.26868897e-03,  9.99587538e-01]], dtype=np.float64)
    prm = [66.12387441012505, 78.5838407426541, -78.17207267392216]
    get_pyr_list(fcinfb, prm)

if __name__ == "__main__":
    main()


# =============================
# Optimization utilities (new)
# =============================

def _evaluate_ypr_across_folders(fcinfb: np.ndarray, prm_list: list[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute YPR means (per folder), overall std, and ptp across folders 1..7 using current fcinfb and prm.
    Returns (means_arr[N,3], stds[3], ptps[3]).
    """
    means = []
    for i in range(1, 8):
        ypr_mean = process_single_pose(str(i), fcinfb, prm_list)
        if ypr_mean is not None:
            means.append(ypr_mean)
    if not means:
        return np.zeros((0, 3), dtype=np.float64), np.array([np.inf, np.inf, np.inf], dtype=np.float64), np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    means_arr = np.asarray(means, dtype=np.float64)
    stds = np.std(means_arr, axis=0)
    ptps = np.ptp(means_arr, axis=0)
    return means_arr, stds, ptps


def optimize_fcinfb_with_minimize(fcinfb_init: np.ndarray, prm_fixed: list[float], bounds_rad: float = np.deg2rad(10.0)):
    """
    Optimize fcinfb using scipy.optimize.minimize (variables are rotation vector, radians).
    Objective: minimize sum of squared std of YPR means across folders (stability objective).
    """
    from scipy.optimize import minimize

    def rvec_to_rmat(rvec: np.ndarray) -> np.ndarray:
        return R.from_rotvec(rvec).as_matrix()

    def objective(rvec: np.ndarray) -> float:
        fcinfb_try = rvec_to_rmat(rvec)
        _, stds, _ = _evaluate_ypr_across_folders(fcinfb_try, prm_fixed)
        # scalar objective: sum of squares (yaw/pitch/roll)
        return float(np.sum(stds * stds))

    # start from current fcinfb: convert to rotvec
    rvec0 = R.from_matrix(fcinfb_init).as_rotvec()
    bnds = [(-bounds_rad, bounds_rad), (-bounds_rad, bounds_rad), (-bounds_rad, bounds_rad)]

    res = minimize(
        objective,
        rvec0,
        method="L-BFGS-B",
        bounds=bnds,
        options={
            "maxiter": 200,
            "ftol": 1e-12,
            "gtol": 1e-12,
            "eps": 1e-9,
            "disp": False,
        },
    )
    fcinfb_opt = R.from_rotvec(res.x).as_matrix()
    return fcinfb_opt, res


def optimize_prm_with_least_squares(prm_init: list[float], fcinfb_fixed: np.ndarray,
                                    d_bounds=(30.0, 120.0), h1_bounds=(0.0, 150.0), h2_bounds=(-150.0, 0.0)):
    """
    Optimize probe geometry parameters prm=[D, H1, H2] using scipy.optimize.least_squares.
    Residuals: for folders 1..7, stack (ypr_i - overall_mean_ypr) to drive cross-folder consistency (reduce std).
    """
    from scipy.optimize import least_squares

    def residuals(x: np.ndarray) -> np.ndarray:
        prm_try = [float(x[0]), float(x[1]), float(x[2])]
        means_arr, _, _ = _evaluate_ypr_across_folders(fcinfb_fixed, prm_try)
        if means_arr.shape[0] == 0:
            # No data; return large residuals
            return np.array([1e6, 1e6, 1e6], dtype=np.float64)
        overall_mean = np.mean(means_arr, axis=0)
        res = (means_arr - overall_mean).reshape(-1)  # flatten (N*3,)
        return res.astype(np.float64)

    x0 = np.asarray(prm_init, dtype=np.float64)
    lb = np.array([d_bounds[0], h1_bounds[0], h2_bounds[0]], dtype=np.float64)
    ub = np.array([d_bounds[1], h1_bounds[1], h2_bounds[1]], dtype=np.float64)

    res = least_squares(
        residuals,
        x0,
        bounds=(lb, ub),
        loss="soft_l1",
        f_scale=0.5,
        ftol=1e-9,
        xtol=1e-9,
        gtol=1e-9,
        max_nfev=500,
        verbose=0,
    )
    prm_opt = res.x.tolist()
    return prm_opt, res


def optimize_fcinfb_and_prm(fcinfb_init: np.ndarray | None = None, prm_init: list[float] | None = None,
                            iters: int = 2):
    """
    Alternating optimization:
    1) Fix prm, optimize fcinfb with minimize (reduce YPR std across folders)
    2) Fix fcinfb, optimize prm with least_squares (residuals = per-folder YPR minus overall mean)
    Repeat for a small number of outer iterations.
    """
    fcinfb = np.eye(3, dtype=np.float64) if fcinfb_init is None else np.asarray(fcinfb_init, dtype=np.float64)
    prm = [65.0, 80.0, -80.0] if prm_init is None else [float(prm_init[0]), float(prm_init[1]), float(prm_init[2])]

    # Print initial error before optimization
    means_arr0, stds0, ptps0 = _evaluate_ypr_across_folders(fcinfb, prm)
    if means_arr0.shape[0] > 0:
        print("Initial YPR std (deg):", stds0)
        print("Initial YPR ptp (deg):", ptps0)

    for _ in range(max(1, int(iters))):
        fcinfb, _ = optimize_fcinfb_with_minimize(fcinfb, prm)
        prm, _ = optimize_prm_with_least_squares(prm, fcinfb)

    # Final report
    means_arr, stds, ptps = _evaluate_ypr_across_folders(fcinfb, prm)
    print("Optimized fcinfb:")
    print(fcinfb)
    print("Optimized prm [D, H1, H2]:", prm)
    if means_arr.shape[0] > 0:
        print("YPR std (deg):", stds)
        print("YPR ptp (deg):", ptps)
    return fcinfb, prm, stds, ptps

fcinfb = np.array([[ 9.99601368e-01, -1.84102328e-04, -2.82324569e-02],
                   [ 3.32774427e-04,  9.99986103e-01,  5.26139129e-03],
                   [ 2.82310959e-02, -5.26868897e-03,  9.99587538e-01]], dtype=np.float64)
prm = [66.12387441012505, 78.5838407426541, -78.17207267392216]
# optimize_fcinfb_and_prm(fcinfb, prm)