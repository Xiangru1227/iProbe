import math
import numpy as np

class Prm:
    def __init__(self, D, H1, H2):
        self.D = D
        self.H1 = H1
        self.H2 = H2

def iPb_uv2pyr(keypoints, prm:Prm):
    pyrs = [0.0, 0.0, 0.0, 0.0]
    
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

    pyrs[0] = pitch
    pyrs[1] = yaw
    pyrs[2] = roll
    pyrs[3] = scale
    
    return pyrs