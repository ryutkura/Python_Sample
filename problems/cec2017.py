# problems/cec2017.py
# CEC2017 ベンチマーク関数スイート完全版
# C++コード (cec17_test_func.cpp) の完全なPython移植

import numpy as np
import os
import math
from .base_problem import BaseProblem

# --- データキャッシュ ---
_data_cache = {}
_temp_arrays_cache = {}

# --- 定数 ---
INF = 1.0e99
EPS = 1.0e-14
E = math.e
PI = math.pi
CF_NUM = 10

def _get_temp_arrays(nx):
    """グローバル配列 y, z の取得または作成"""
    if nx not in _temp_arrays_cache:
        _temp_arrays_cache[nx] = {
            'y': np.zeros(nx),
            'z': np.zeros(nx)
        }
    return _temp_arrays_cache[nx]

def _load_cec_data(func_num, nx):
    """データファイルの読み込み"""
    key = (func_num, nx)
    if key in _data_cache:
        return _data_cache[key]

    # 次元数の検証
    if nx not in [2, 10, 20, 30, 50, 100]:
        raise ValueError(f"Dimension {nx} is not supported. Use 2, 10, 20, 30, 50, or 100.")
    
    # F2の検証
    if func_num == 2:
        raise ValueError("Function F2 has been deleted from CEC2017.")
    
    # D=2での制限チェック
    if nx == 2 and (17 <= func_num <= 22 or func_num in [29, 30]):
        raise ValueError(f"Function F{func_num} is not defined for D=2.")

    base_path = "input_data"
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"'{base_path}' directory not found. Please copy it to the project root.")

    M = None
    OShift = None
    SS = None

    # --- M (Rotation Matrix) の読み込み ---
    filename = os.path.join(base_path, f"M_{func_num}_D{nx}.txt")
    try:
        M_flat = np.loadtxt(filename)
        if func_num < 20:
            M = M_flat.reshape(nx, nx)
        else:
            M = M_flat.reshape(CF_NUM, nx, nx)
    except Exception as e:
        print(f"Warning: Could not load M data for F{func_num} D{nx}. {e}")
        if func_num < 20:
            M = np.eye(nx)
        else:
            M = np.array([np.eye(nx) for _ in range(CF_NUM)])

    # --- OShift (Shift Vector) の読み込み ---
    filename = os.path.join(base_path, f"shift_data_{func_num}.txt")
    try:
        if func_num < 20:
            OShift = np.loadtxt(filename)[:nx]
        else:
            with open(filename, 'r') as f:
                lines = f.readlines()
                OShift = []
                for i in range(CF_NUM):
                    if i < len(lines):
                        values = [float(x) for x in lines[i].split()]
                        OShift.extend(values[:nx])
                OShift = np.array(OShift)
    except Exception as e:
        print(f"Warning: Could not load OShift data for F{func_num}. {e}")
        OShift = np.zeros(nx * (CF_NUM if func_num >= 20 else 1))

    # --- SS (Shuffle Data) の読み込み ---
    if (11 <= func_num <= 20) or (func_num == 29) or (func_num == 30):
        filename = os.path.join(base_path, f"shuffle_data_{func_num}_D{nx}.txt")
        try:
            SS_flat = np.loadtxt(filename, dtype=int)
            if func_num == 29 or func_num == 30:
                SS = SS_flat.reshape(CF_NUM, nx)
            else:
                SS = SS_flat
            SS = SS - 1  # 1-based to 0-based
        except Exception as e:
            print(f"Warning: Could not load SS data for F{func_num} D{nx}. {e}")
            if func_num == 29 or func_num == 30:
                SS = np.array([np.arange(nx) for _ in range(CF_NUM)])
            else:
                SS = np.arange(nx)

    data = {'M': M, 'OShift': OShift, 'SS': SS}
    _data_cache[key] = data
    return data

# --- ヘルパー関数 ---

def _shiftfunc(x, os_vec):
    """シフト変換"""
    return x - os_vec

def _rotatefunc(y, mr_mat):
    """回転変換"""
    return mr_mat.dot(y)

def _sr_func(x, os_vec, mr_mat, sh_rate, s_flag, r_flag, nx):
    """シフト＆回転変換"""
    temp_arrays = _get_temp_arrays(nx)
    y = temp_arrays['y']
    z = temp_arrays['z']
    
    if s_flag == 1:
        y = _shiftfunc(x, os_vec)
        y = y * sh_rate
    else:
        y = x * sh_rate

    if r_flag == 1:
        z = _rotatefunc(y, mr_mat)
        return z.copy()
    else:
        z[:] = y
        return z.copy()

def _asyfunc(x, nx, beta):
    """非対称変換"""
    xasy = x.copy()
    indices = x > 0
    xasy[indices] = np.power(x[indices], 1.0 + beta * np.arange(nx)[indices] / (nx - 1) * np.sqrt(x[indices]))
    return xasy

def _oszfunc(x, nx):
    """振動変換"""
    xosz = x.copy()
    for i in [0, nx - 1]:
        if x[i] != 0:
            xx = math.log(abs(x[i]))
            if x[i] > 0:
                c1, c2 = 10, 7.9
            else:
                c1, c2 = 5.5, 3.1
            
            sx = np.sign(x[i])
            xosz[i] = sx * math.exp(xx + 0.049 * (math.sin(c1 * xx) + math.sin(c2 * xx)))
    return xosz

def _cf_cal(x, nx, Os, delta, bias, fit, cf_num):
    """コンポジション関数の計算"""
    w = np.zeros(cf_num)
    w_max = 0
    
    for i in range(cf_num):
        fit[i] += bias[i]
        w_i = 0
        for j in range(nx):
            w_i += (x[j] - Os[i * nx + j]) ** 2
        
        if w_i != 0:
            w[i] = (1.0 / w_i)**0.5 * math.exp(-w_i / (2.0 * nx * (delta[i]**2)))
        else:
            w[i] = INF
        
        if w[i] > w_max:
            w_max = w[i]

    w_sum = np.sum(w)
    if w_max == 0:
        w = np.ones(cf_num)
        w_sum = cf_num

    f_val = np.sum(w / w_sum * fit)
    return f_val

# --- 基本関数 ---

def _sphere_func(x, nx, Os, Mr, s_flag, r_flag):
    """Sphere関数"""
    z = _sr_func(x, Os, Mr, 1.0, s_flag, r_flag, nx)
    return np.sum(z**2)

def _ellips_func(x, nx, Os, Mr, s_flag, r_flag):
    """Ellipsoidal関数"""
    z = _sr_func(x, Os, Mr, 1.0, s_flag, r_flag, nx)
    f = 0.0
    for i in range(nx):
        f += (10.0**(6.0 * i / (nx - 1))) * z[i]**2
    return f

def _bent_cigar_func(x, nx, Os, Mr, s_flag, r_flag):
    """Bent Cigar関数"""
    z = _sr_func(x, Os, Mr, 1.0, s_flag, r_flag, nx)
    f = z[0]**2 + 1e6 * np.sum(z[1:]**2)
    return f

def _discus_func(x, nx, Os, Mr, s_flag, r_flag):
    """Discus関数"""
    z = _sr_func(x, Os, Mr, 1.0, s_flag, r_flag, nx)
    f = 1e6 * z[0]**2 + np.sum(z[1:]**2)
    return f

def _dif_powers_func(x, nx, Os, Mr, s_flag, r_flag):
    """Different Powers関数"""
    z = _sr_func(x, Os, Mr, 1.0, s_flag, r_flag, nx)
    f = 0.0
    for i in range(nx):
        f += abs(z[i]) ** (2 + 4 * i / (nx - 1))
    return math.sqrt(f)

def _rosenbrock_func(x, nx, Os, Mr, s_flag, r_flag):
    """Rosenbrock関数"""
    z = _sr_func(x, Os, Mr, 2.048/100.0, s_flag, r_flag, nx)
    z = z + 1.0
    tmp1 = z[:-1]**2 - z[1:]
    tmp2 = z[:-1] - 1.0
    f = np.sum(100.0 * tmp1**2 + tmp2**2)
    return f

def _schaffer_F7_func(x, nx, Os, Mr, s_flag, r_flag):
    """Schaffer's F7関数"""
    z = _sr_func(x, Os, Mr, 1.0, s_flag, r_flag, nx)
    s = np.sqrt(z[:-1]**2 + z[1:]**2)
    tmp = np.sin(50.0 * s**0.2)
    f = np.sum(s**0.5 + s**0.5 * tmp**2)
    return (f / (nx - 1))**2

def _ackley_func(x, nx, Os, Mr, s_flag, r_flag):
    """Ackley関数"""
    z = _sr_func(x, Os, Mr, 1.0, s_flag, r_flag, nx)
    sum1 = np.sum(z**2)
    sum2 = np.sum(np.cos(2.0 * PI * z))
    sum1 = -0.2 * math.sqrt(sum1 / nx)
    sum2 /= nx
    return E - 20.0 * math.exp(sum1) - math.exp(sum2) + 20.0

def _weierstrass_func(x, nx, Os, Mr, s_flag, r_flag):
    """Weierstrass関数"""
    z = _sr_func(x, Os, Mr, 0.5/100.0, s_flag, r_flag, nx)
    a, b, k_max = 0.5, 3.0, 20
    f = 0.0
    sum2 = 0.0
    
    for j in range(k_max + 1):
        sum2 += (a**j) * math.cos(2.0 * PI * (b**j) * 0.5)
    
    for i in range(nx):
        sum_val = 0.0
        for j in range(k_max + 1):
            sum_val += (a**j) * math.cos(2.0 * PI * (b**j) * (z[i] + 0.5))
        f += sum_val
    
    return f - nx * sum2

def _griewank_func(x, nx, Os, Mr, s_flag, r_flag):
    """Griewank関数"""
    z = _sr_func(x, Os, Mr, 600.0/100.0, s_flag, r_flag, nx)
    s = np.sum(z**2)
    p = np.prod(np.cos(z / np.sqrt(np.arange(1, nx + 1))))
    return 1.0 + s / 4000.0 - p

def _rastrigin_func(x, nx, Os, Mr, s_flag, r_flag):
    """Rastrigin関数"""
    z = _sr_func(x, Os, Mr, 5.12/100.0, s_flag, r_flag, nx)
    return np.sum(z**2 - 10.0 * np.cos(2.0 * PI * z) + 10.0)

def _step_rastrigin_func(x, nx, Os, Mr, s_flag, r_flag):
    """Step Rastrigin関数"""
    y = x.copy()
    for i in range(nx):
        if abs(y[i] - Os[i]) > 0.5:
            y[i] = Os[i] + math.floor(2 * (y[i] - Os[i]) + 0.5) / 2
            
    z = _sr_func(y, Os, Mr, 5.12/100.0, s_flag, r_flag, nx)
    return np.sum(z**2 - 10.0 * np.cos(2.0 * PI * z) + 10.0)

def _schwefel_func(x, nx, Os, Mr, s_flag, r_flag):
    """Schwefel関数"""
    z = _sr_func(x, Os, Mr, 1000.0/100.0, s_flag, r_flag, nx)
    f = 0.0
    for i in range(nx):
        zi = z[i] + 4.209687462275036e+002
        if zi > 500:
            f -= (500.0 - math.fmod(zi, 500)) * math.sin(math.sqrt(abs(500.0 - math.fmod(zi, 500))))
            tmp = (zi - 500.0) / 100
            f += tmp * tmp / nx
        elif zi < -500:
            f -= (-500.0 + math.fmod(abs(zi), 500)) * math.sin(math.sqrt(abs(500.0 - math.fmod(abs(zi), 500))))
            tmp = (zi + 500.0) / 100
            f += tmp * tmp / nx
        else:
            f -= zi * math.sin(math.sqrt(abs(zi)))
    f += 4.189828872724338e+002 * nx
    return f

def _katsuura_func(x, nx, Os, Mr, s_flag, r_flag):
    """Katsuura関数"""
    z = _sr_func(x, Os, Mr, 5.0/100.0, s_flag, r_flag, nx)
    f = 1.0
    tmp3 = (nx ** 1.2)
    
    for i in range(nx):
        temp = 0.0
        for j in range(1, 33):
            tmp1 = 2.0 ** j
            tmp2 = tmp1 * z[i]
            temp += abs(tmp2 - math.floor(tmp2 + 0.5)) / tmp1
        f *= (1.0 + (i + 1) * temp) ** (10.0 / tmp3)
    
    tmp1 = 10.0 / nx / nx
    return f * tmp1 - tmp1

def _bi_rastrigin_func(x, nx, Os, Mr, s_flag, r_flag):
    """Bi-Rastrigin関数"""
    mu0 = 2.5
    d = 1.0
    s = 1.0 - 1.0 / (2.0 * math.sqrt(nx + 20.0) - 8.2)
    mu1 = -math.sqrt((mu0**2 - d) / s)
    
    if s_flag == 1:
        y = _shiftfunc(x, Os)
    else:
        y = x.copy()
    y *= 10.0 / 100.0

    tmpx = 2 * y
    tmpx[Os < 0] *= -1.0
    
    z = tmpx.copy()
    tmpx = tmpx + mu0
    
    tmp1 = np.sum((tmpx - mu0)**2)
    tmp2 = np.sum((tmpx - mu1)**2)
    tmp2 = s * tmp2 + d * nx
    
    if r_flag == 1:
        y_rot = _rotatefunc(z, Mr)
        tmp = np.sum(np.cos(2.0 * PI * y_rot))
    else:
        tmp = np.sum(np.cos(2.0 * PI * z))
        
    f_val = min(tmp1, tmp2) + 10.0 * (nx - tmp)
    return f_val

def _grie_rosen_func(x, nx, Os, Mr, s_flag, r_flag):
    """Griewank-Rosenbrock関数"""
    z = _sr_func(x, Os, Mr, 5.0/100.0, s_flag, r_flag, nx)
    z = z + 1.0
    f = 0.0
    
    for i in range(nx - 1):
        tmp1 = z[i]**2 - z[i + 1]
        tmp2 = z[i] - 1.0
        temp = 100.0 * tmp1**2 + tmp2**2
        f += (temp**2) / 4000.0 - math.cos(temp) + 1.0
    
    tmp1 = z[nx - 1]**2 - z[0]
    tmp2 = z[nx - 1] - 1.0
    temp = 100.0 * tmp1**2 + tmp2**2
    f += (temp**2) / 4000.0 - math.cos(temp) + 1.0
    
    return f

def _escaffer6_func(x, nx, Os, Mr, s_flag, r_flag):
    """Expanded Schaffer's F6関数"""
    z = _sr_func(x, Os, Mr, 1.0, s_flag, r_flag, nx)
    f = 0.0
    
    for i in range(nx - 1):
        temp1 = math.sin(math.sqrt(z[i]**2 + z[i + 1]**2))
        temp1 = temp1**2
        temp2 = 1.0 + 0.001 * (z[i]**2 + z[i + 1]**2)
        f += 0.5 + (temp1 - 0.5) / (temp2**2)
    
    temp1 = math.sin(math.sqrt(z[nx - 1]**2 + z[0]**2))
    temp1 = temp1**2
    temp2 = 1.0 + 0.001 * (z[nx - 1]**2 + z[0]**2)
    f += 0.5 + (temp1 - 0.5) / (temp2**2)
    
    return f

def _happycat_func(x, nx, Os, Mr, s_flag, r_flag):
    """HappyCat関数"""
    z = _sr_func(x, Os, Mr, 5.0/100.0, s_flag, r_flag, nx)
    z = z - 1.0
    alpha = 1.0 / 8.0
    r2 = np.sum(z**2)
    sum_z = np.sum(z)
    return abs(r2 - nx) ** (2 * alpha) + (0.5 * r2 + sum_z) / nx + 0.5

def _hgbat_func(x, nx, Os, Mr, s_flag, r_flag):
    """HGBat関数"""
    z = _sr_func(x, Os, Mr, 5.0/100.0, s_flag, r_flag, nx)
    z = z - 1.0
    alpha = 1.0 / 4.0
    r2 = np.sum(z**2)
    sum_z = np.sum(z)
    return abs(r2**2 - sum_z**2) ** (2 * alpha) + (0.5 * r2 + sum_z) / nx + 0.5

def _zakharov_func(x, nx, Os, Mr, s_flag, r_flag):
    """Zakharov関数"""
    z = _sr_func(x, Os, Mr, 1.0, s_flag, r_flag, nx)
    sum1 = np.sum(z**2)
    sum2 = np.sum(0.5 * (np.arange(1, nx + 1)) * z)
    return sum1 + sum2**2 + sum2**4

def _levy_func(x, nx, Os, Mr, s_flag, r_flag):
    """Levy関数"""
    z = _sr_func(x, Os, Mr, 1.0, s_flag, r_flag, nx)
    w = 1.0 + (z - 1.0) / 4.0
    
    term1 = math.sin(PI * w[0])**2
    term3 = (w[nx - 1] - 1)**2 * (1 + math.sin(2 * PI * w[nx - 1])**2)
    
    wi = w[:-1]
    sum_val = np.sum((wi - 1)**2 * (1 + 10 * np.sin(PI * wi + 1)**2))
    
    return term1 + sum_val + term3

# --- Hybrid関数 (F11-F20) ---

def _hf01(x, nx, Os, Mr, SS, s_flag, r_flag):
    """Hybrid Function 1"""
    cf_num = 3
    Gp = [0.2, 0.4, 0.4]
    G_nx = [int(np.ceil(Gp[i] * nx)) for i in range(cf_num - 1)]
    G_nx.append(nx - sum(G_nx))
    G = [0]
    for i in range(1, cf_num):
        G.append(G[i - 1] + G_nx[i - 1])
    
    z = _sr_func(x, Os, Mr, 1.0, s_flag, r_flag, nx)
    y = z[SS]
    
    fit = np.zeros(cf_num)
    fit[0] = _zakharov_func(y[G[0]:G[0] + G_nx[0]], G_nx[0], np.zeros(G_nx[0]), np.eye(G_nx[0]), 0, 0)
    fit[1] = _rosenbrock_func(y[G[1]:G[1] + G_nx[1]], G_nx[1], np.zeros(G_nx[1]), np.eye(G_nx[1]), 0, 0)
    fit[2] = _rastrigin_func(y[G[2]:G[2] + G_nx[2]], G_nx[2], np.zeros(G_nx[2]), np.eye(G_nx[2]), 0, 0)
    
    return np.sum(fit)

def _hf02(x, nx, Os, Mr, SS, s_flag, r_flag):
    """Hybrid Function 2"""
    cf_num = 3
    Gp = [0.3, 0.3, 0.4]
    G_nx = [int(np.ceil(Gp[i] * nx)) for i in range(cf_num - 1)]
    G_nx.append(nx - sum(G_nx))
    G = [0]
    for i in range(1, cf_num):
        G.append(G[i - 1] + G_nx[i - 1])
    
    z = _sr_func(x, Os, Mr, 1.0, s_flag, r_flag, nx)
    y = z[SS]
    
    fit = np.zeros(cf_num)
    fit[0] = _ellips_func(y[G[0]:G[0] + G_nx[0]], G_nx[0], np.zeros(G_nx[0]), np.eye(G_nx[0]), 0, 0)
    fit[1] = _schwefel_func(y[G[1]:G[1] + G_nx[1]], G_nx[1], np.zeros(G_nx[1]), np.eye(G_nx[1]), 0, 0)
    fit[2] = _bent_cigar_func(y[G[2]:G[2] + G_nx[2]], G_nx[2], np.zeros(G_nx[2]), np.eye(G_nx[2]), 0, 0)
    
    return np.sum(fit)

def _hf03(x, nx, Os, Mr, SS, s_flag, r_flag):
    """Hybrid Function 3"""
    cf_num = 3
    Gp = [0.3, 0.3, 0.4]
    G_nx = [int(np.ceil(Gp[i] * nx)) for i in range(cf_num - 1)]
    G_nx.append(nx - sum(G_nx))
    G = [0]
    for i in range(1, cf_num):
        G.append(G[i - 1] + G_nx[i - 1])
    
    z = _sr_func(x, Os, Mr, 1.0, s_flag, r_flag, nx)
    y = z[SS]
    
    fit = np.zeros(cf_num)
    fit[0] = _bent_cigar_func(y[G[0]:G[0] + G_nx[0]], G_nx[0], np.zeros(G_nx[0]), np.eye(G_nx[0]), 0, 0)
    fit[1] = _rosenbrock_func(y[G[1]:G[1] + G_nx[1]], G_nx[1], np.zeros(G_nx[1]), np.eye(G_nx[1]), 0, 0)
    fit[2] = _bi_rastrigin_func(y[G[2]:G[2] + G_nx[2]], G_nx[2], np.zeros(G_nx[2]), np.eye(G_nx[2]), 0, 0)
    
    return np.sum(fit)

def _hf04(x, nx, Os, Mr, SS, s_flag, r_flag):
    """Hybrid Function 4"""
    cf_num = 4
    Gp = [0.2, 0.2, 0.2, 0.4]
    G_nx = [int(np.ceil(Gp[i] * nx)) for i in range(cf_num - 1)]
    G_nx.append(nx - sum(G_nx))
    G = [0]
    for i in range(1, cf_num):
        G.append(G[i - 1] + G_nx[i - 1])
    
    z = _sr_func(x, Os, Mr, 1.0, s_flag, r_flag, nx)
    y = z[SS]
    
    fit = np.zeros(cf_num)
    fit[0] = _ellips_func(y[G[0]:G[0] + G_nx[0]], G_nx[0], np.zeros(G_nx[0]), np.eye(G_nx[0]), 0, 0)
    fit[1] = _ackley_func(y[G[1]:G[1] + G_nx[1]], G_nx[1], np.zeros(G_nx[1]), np.eye(G_nx[1]), 0, 0)
    fit[2] = _schaffer_F7_func(y[G[2]:G[2] + G_nx[2]], G_nx[2], np.zeros(G_nx[2]), np.eye(G_nx[2]), 0, 0)
    fit[3] = _rastrigin_func(y[G[3]:G[3] + G_nx[3]], G_nx[3], np.zeros(G_nx[3]), np.eye(G_nx[3]), 0, 0)
    
    return np.sum(fit)

def _hf05(x, nx, Os, Mr, SS, s_flag, r_flag):
    """Hybrid Function 5"""
    cf_num = 4
    Gp = [0.2, 0.2, 0.3, 0.3]
    G_nx = [int(np.ceil(Gp[i] * nx)) for i in range(cf_num - 1)]
    G_nx.append(nx - sum(G_nx))
    G = [0]
    for i in range(1, cf_num):
        G.append(G[i - 1] + G_nx[i - 1])
    
    z = _sr_func(x, Os, Mr, 1.0, s_flag, r_flag, nx)
    y = z[SS]
    
    fit = np.zeros(cf_num)
    fit[0] = _bent_cigar_func(y[G[0]:G[0] + G_nx[0]], G_nx[0], np.zeros(G_nx[0]), np.eye(G_nx[0]), 0, 0)
    fit[1] = _hgbat_func(y[G[1]:G[1] + G_nx[1]], G_nx[1], np.zeros(G_nx[1]), np.eye(G_nx[1]), 0, 0)
    fit[2] = _rastrigin_func(y[G[2]:G[2] + G_nx[2]], G_nx[2], np.zeros(G_nx[2]), np.eye(G_nx[2]), 0, 0)
    fit[3] = _rosenbrock_func(y[G[3]:G[3] + G_nx[3]], G_nx[3], np.zeros(G_nx[3]), np.eye(G_nx[3]), 0, 0)
    
    return np.sum(fit)

def _hf06(x, nx, Os, Mr, SS, s_flag, r_flag):
    """Hybrid Function 6"""
    cf_num = 4
    Gp = [0.2, 0.2, 0.3, 0.3]
    G_nx = [int(np.ceil(Gp[i] * nx)) for i in range(cf_num - 1)]
    G_nx.append(nx - sum(G_nx))
    G = [0]
    for i in range(1, cf_num):
        G.append(G[i - 1] + G_nx[i - 1])
    
    z = _sr_func(x, Os, Mr, 1.0, s_flag, r_flag, nx)
    y = z[SS]
    
    fit = np.zeros(cf_num)
    fit[0] = _escaffer6_func(y[G[0]:G[0] + G_nx[0]], G_nx[0], np.zeros(G_nx[0]), np.eye(G_nx[0]), 0, 0)
    fit[1] = _hgbat_func(y[G[1]:G[1] + G_nx[1]], G_nx[1], np.zeros(G_nx[1]), np.eye(G_nx[1]), 0, 0)
    fit[2] = _rosenbrock_func(y[G[2]:G[2] + G_nx[2]], G_nx[2], np.zeros(G_nx[2]), np.eye(G_nx[2]), 0, 0)
    fit[3] = _schwefel_func(y[G[3]:G[3] + G_nx[3]], G_nx[3], np.zeros(G_nx[3]), np.eye(G_nx[3]), 0, 0)
    
    return np.sum(fit)

def _hf07(x, nx, Os, Mr, SS, s_flag, r_flag):
    """Hybrid Function 7"""
    cf_num = 5
    Gp = [0.1, 0.2, 0.2, 0.2, 0.3]
    G_nx = [int(np.ceil(Gp[i] * nx)) for i in range(cf_num - 1)]
    G_nx.append(nx - sum(G_nx))
    G = [0]
    for i in range(1, cf_num):
        G.append(G[i - 1] + G_nx[i - 1])
    
    z = _sr_func(x, Os, Mr, 1.0, s_flag, r_flag, nx)
    y = z[SS]
    
    fit = np.zeros(cf_num)
    fit[0] = _katsuura_func(y[G[0]:G[0] + G_nx[0]], G_nx[0], np.zeros(G_nx[0]), np.eye(G_nx[0]), 0, 0)
    fit[1] = _ackley_func(y[G[1]:G[1] + G_nx[1]], G_nx[1], np.zeros(G_nx[1]), np.eye(G_nx[1]), 0, 0)
    fit[2] = _grie_rosen_func(y[G[2]:G[2] + G_nx[2]], G_nx[2], np.zeros(G_nx[2]), np.eye(G_nx[2]), 0, 0)
    fit[3] = _schwefel_func(y[G[3]:G[3] + G_nx[3]], G_nx[3], np.zeros(G_nx[3]), np.eye(G_nx[3]), 0, 0)
    fit[4] = _rastrigin_func(y[G[4]:G[4] + G_nx[4]], G_nx[4], np.zeros(G_nx[4]), np.eye(G_nx[4]), 0, 0)
    
    return np.sum(fit)

def _hf08(x, nx, Os, Mr, SS, s_flag, r_flag):
    """Hybrid Function 8"""
    cf_num = 5
    Gp = [0.2, 0.2, 0.2, 0.2, 0.2]
    G_nx = [int(np.ceil(Gp[i] * nx)) for i in range(cf_num - 1)]
    G_nx.append(nx - sum(G_nx))
    G = [0]
    for i in range(1, cf_num):
        G.append(G[i - 1] + G_nx[i - 1])
    
    z = _sr_func(x, Os, Mr, 1.0, s_flag, r_flag, nx)
    y = z[SS]
    
    fit = np.zeros(cf_num)
    fit[0] = _ellips_func(y[G[0]:G[0] + G_nx[0]], G_nx[0], np.zeros(G_nx[0]), np.eye(G_nx[0]), 0, 0)
    fit[1] = _ackley_func(y[G[1]:G[1] + G_nx[1]], G_nx[1], np.zeros(G_nx[1]), np.eye(G_nx[1]), 0, 0)
    fit[2] = _rastrigin_func(y[G[2]:G[2] + G_nx[2]], G_nx[2], np.zeros(G_nx[2]), np.eye(G_nx[2]), 0, 0)
    fit[3] = _hgbat_func(y[G[3]:G[3] + G_nx[3]], G_nx[3], np.zeros(G_nx[3]), np.eye(G_nx[3]), 0, 0)
    fit[4] = _discus_func(y[G[4]:G[4] + G_nx[4]], G_nx[4], np.zeros(G_nx[4]), np.eye(G_nx[4]), 0, 0)
    
    return np.sum(fit)

def _hf09(x, nx, Os, Mr, SS, s_flag, r_flag):
    """Hybrid Function 9"""
    cf_num = 5
    Gp = [0.2, 0.2, 0.2, 0.2, 0.2]
    G_nx = [int(np.ceil(Gp[i] * nx)) for i in range(cf_num - 1)]
    G_nx.append(nx - sum(G_nx))
    G = [0]
    for i in range(1, cf_num):
        G.append(G[i - 1] + G_nx[i - 1])
    
    z = _sr_func(x, Os, Mr, 1.0, s_flag, r_flag, nx)
    y = z[SS]
    
    fit = np.zeros(cf_num)
    fit[0] = _bent_cigar_func(y[G[0]:G[0] + G_nx[0]], G_nx[0], np.zeros(G_nx[0]), np.eye(G_nx[0]), 0, 0)
    fit[1] = _rastrigin_func(y[G[1]:G[1] + G_nx[1]], G_nx[1], np.zeros(G_nx[1]), np.eye(G_nx[1]), 0, 0)
    fit[2] = _grie_rosen_func(y[G[2]:G[2] + G_nx[2]], G_nx[2], np.zeros(G_nx[2]), np.eye(G_nx[2]), 0, 0)
    fit[3] = _weierstrass_func(y[G[3]:G[3] + G_nx[3]], G_nx[3], np.zeros(G_nx[3]), np.eye(G_nx[3]), 0, 0)
    fit[4] = _escaffer6_func(y[G[4]:G[4] + G_nx[4]], G_nx[4], np.zeros(G_nx[4]), np.eye(G_nx[4]), 0, 0)
    
    return np.sum(fit)

def _hf10(x, nx, Os, Mr, SS, s_flag, r_flag):
    """Hybrid Function 10"""
    cf_num = 6
    Gp = [0.1, 0.1, 0.2, 0.2, 0.2, 0.2]
    G_nx = [int(np.ceil(Gp[i] * nx)) for i in range(cf_num - 1)]
    G_nx.append(nx - sum(G_nx))
    G = [0]
    for i in range(1, cf_num):
        G.append(G[i - 1] + G_nx[i - 1])
    
    z = _sr_func(x, Os, Mr, 1.0, s_flag, r_flag, nx)
    y = z[SS]
    
    fit = np.zeros(cf_num)
    fit[0] = _hgbat_func(y[G[0]:G[0] + G_nx[0]], G_nx[0], np.zeros(G_nx[0]), np.eye(G_nx[0]), 0, 0)
    fit[1] = _katsuura_func(y[G[1]:G[1] + G_nx[1]], G_nx[1], np.zeros(G_nx[1]), np.eye(G_nx[1]), 0, 0)
    fit[2] = _ackley_func(y[G[2]:G[2] + G_nx[2]], G_nx[2], np.zeros(G_nx[2]), np.eye(G_nx[2]), 0, 0)
    fit[3] = _rastrigin_func(y[G[3]:G[3] + G_nx[3]], G_nx[3], np.zeros(G_nx[3]), np.eye(G_nx[3]), 0, 0)
    fit[4] = _schwefel_func(y[G[4]:G[4] + G_nx[4]], G_nx[4], np.zeros(G_nx[4]), np.eye(G_nx[4]), 0, 0)
    fit[5] = _schaffer_F7_func(y[G[5]:G[5] + G_nx[5]], G_nx[5], np.zeros(G_nx[5]), np.eye(G_nx[5]), 0, 0)
    
    return np.sum(fit)

# --- Composition関数 (F21-F30) ---

def _cf01(x, nx, Os, Mr, r_flag):
    """Composition Function 1"""
    cf_num = 3
    delta = np.array([10, 20, 30])
    bias = np.array([0, 100, 200])
    fit = np.zeros(cf_num)
    
    fit[0] = _rosenbrock_func(x, nx, Os[0:nx], Mr[0].reshape(nx, nx) if Mr.ndim == 3 else Mr, 1, r_flag)
    fit[1] = _ellips_func(x, nx, Os[nx:2*nx], Mr[1].reshape(nx, nx) if Mr.ndim == 3 else Mr, 1, r_flag)
    fit[1] = 10000 * fit[1] / 1e+10
    fit[2] = _rastrigin_func(x, nx, Os[2*nx:3*nx], Mr[2].reshape(nx, nx) if Mr.ndim == 3 else Mr, 1, r_flag)
    
    return _cf_cal(x, nx, Os, delta, bias, fit, cf_num)

def _cf02(x, nx, Os, Mr, r_flag):
    """Composition Function 2"""
    cf_num = 3
    delta = np.array([10, 20, 30])
    bias = np.array([0, 100, 200])
    fit = np.zeros(cf_num)
    
    fit[0] = _rastrigin_func(x, nx, Os[0:nx], Mr[0].reshape(nx, nx) if Mr.ndim == 3 else Mr, 1, r_flag)
    fit[1] = _griewank_func(x, nx, Os[nx:2*nx], Mr[1].reshape(nx, nx) if Mr.ndim == 3 else Mr, 1, r_flag)
    fit[1] = 1000 * fit[1] / 100
    fit[2] = _schwefel_func(x, nx, Os[2*nx:3*nx], Mr[2].reshape(nx, nx) if Mr.ndim == 3 else Mr, 1, r_flag)
    
    return _cf_cal(x, nx, Os, delta, bias, fit, cf_num)

def _cf03(x, nx, Os, Mr, r_flag):
    """Composition Function 3"""
    cf_num = 4
    delta = np.array([10, 20, 30, 40])
    bias = np.array([0, 100, 200, 300])
    fit = np.zeros(cf_num)
    
    fit[0] = _rosenbrock_func(x, nx, Os[0:nx], Mr[0].reshape(nx, nx) if Mr.ndim == 3 else Mr, 1, r_flag)
    fit[1] = _ackley_func(x, nx, Os[nx:2*nx], Mr[1].reshape(nx, nx) if Mr.ndim == 3 else Mr, 1, r_flag)
    fit[1] = 1000 * fit[1] / 100
    fit[2] = _schwefel_func(x, nx, Os[2*nx:3*nx], Mr[2].reshape(nx, nx) if Mr.ndim == 3 else Mr, 1, r_flag)
    fit[3] = _rastrigin_func(x, nx, Os[3*nx:4*nx], Mr[3].reshape(nx, nx) if Mr.ndim == 3 else Mr, 1, r_flag)
    
    return _cf_cal(x, nx, Os, delta, bias, fit, cf_num)

def _cf04(x, nx, Os, Mr, r_flag):
    """Composition Function 4"""
    cf_num = 4
    delta = np.array([10, 20, 30, 40])
    bias = np.array([0, 100, 200, 300])
    fit = np.zeros(cf_num)
    
    fit[0] = _ackley_func(x, nx, Os[0:nx], Mr[0].reshape(nx, nx) if Mr.ndim == 3 else Mr, 1, r_flag)
    fit[0] = 1000 * fit[0] / 100
    fit[1] = _ellips_func(x, nx, Os[nx:2*nx], Mr[1].reshape(nx, nx) if Mr.ndim == 3 else Mr, 1, r_flag)
    fit[1] = 10000 * fit[1] / 1e+10
    fit[2] = _griewank_func(x, nx, Os[2*nx:3*nx], Mr[2].reshape(nx, nx) if Mr.ndim == 3 else Mr, 1, r_flag)
    fit[2] = 1000 * fit[2] / 100
    fit[3] = _rastrigin_func(x, nx, Os[3*nx:4*nx], Mr[3].reshape(nx, nx) if Mr.ndim == 3 else Mr, 1, r_flag)
    
    return _cf_cal(x, nx, Os, delta, bias, fit, cf_num)

def _cf05(x, nx, Os, Mr, r_flag):
    """Composition Function 5"""
    cf_num = 5
    delta = np.array([10, 20, 30, 40, 50])
    bias = np.array([0, 100, 200, 300, 400])
    fit = np.zeros(cf_num)
    
    fit[0] = _rastrigin_func(x, nx, Os[0:nx], Mr[0].reshape(nx, nx) if Mr.ndim == 3 else Mr, 1, r_flag)
    fit[0] = 10000 * fit[0] / 1e+3
    fit[1] = _happycat_func(x, nx, Os[nx:2*nx], Mr[1].reshape(nx, nx) if Mr.ndim == 3 else Mr, 1, r_flag)
    fit[1] = 1000 * fit[1] / 1e+3
    fit[2] = _ackley_func(x, nx, Os[2*nx:3*nx], Mr[2].reshape(nx, nx) if Mr.ndim == 3 else Mr, 1, r_flag)
    fit[2] = 1000 * fit[2] / 100
    fit[3] = _discus_func(x, nx, Os[3*nx:4*nx], Mr[3].reshape(nx, nx) if Mr.ndim == 3 else Mr, 1, r_flag)
    fit[3] = 10000 * fit[3] / 1e+10
    fit[4] = _rosenbrock_func(x, nx, Os[4*nx:5*nx], Mr[4].reshape(nx, nx) if Mr.ndim == 3 else Mr, 1, r_flag)
    
    return _cf_cal(x, nx, Os, delta, bias, fit, cf_num)

def _cf06(x, nx, Os, Mr, r_flag):
    """Composition Function 6"""
    cf_num = 5
    delta = np.array([10, 20, 20, 30, 40])
    bias = np.array([0, 100, 200, 300, 400])
    fit = np.zeros(cf_num)
    
    fit[0] = _escaffer6_func(x, nx, Os[0:nx], Mr[0].reshape(nx, nx) if Mr.ndim == 3 else Mr, 1, r_flag)
    fit[0] = 10000 * fit[0] / 2e+7
    fit[1] = _schwefel_func(x, nx, Os[nx:2*nx], Mr[1].reshape(nx, nx) if Mr.ndim == 3 else Mr, 1, r_flag)
    fit[2] = _griewank_func(x, nx, Os[2*nx:3*nx], Mr[2].reshape(nx, nx) if Mr.ndim == 3 else Mr, 1, r_flag)
    fit[2] = 1000 * fit[2] / 100
    fit[3] = _rosenbrock_func(x, nx, Os[3*nx:4*nx], Mr[3].reshape(nx, nx) if Mr.ndim == 3 else Mr, 1, r_flag)
    fit[4] = _rastrigin_func(x, nx, Os[4*nx:5*nx], Mr[4].reshape(nx, nx) if Mr.ndim == 3 else Mr, 1, r_flag)
    fit[4] = 10000 * fit[4] / 1e+3
    
    return _cf_cal(x, nx, Os, delta, bias, fit, cf_num)

def _cf07(x, nx, Os, Mr, r_flag):
    """Composition Function 7"""
    cf_num = 6
    delta = np.array([10, 20, 30, 40, 50, 60])
    bias = np.array([0, 100, 200, 300, 400, 500])
    fit = np.zeros(cf_num)
    
    fit[0] = _hgbat_func(x, nx, Os[0:nx], Mr[0].reshape(nx, nx) if Mr.ndim == 3 else Mr, 1, r_flag)
    fit[0] = 10000 * fit[0] / 1000
    fit[1] = _rastrigin_func(x, nx, Os[nx:2*nx], Mr[1].reshape(nx, nx) if Mr.ndim == 3 else Mr, 1, r_flag)
    fit[1] = 10000 * fit[1] / 1e+3
    fit[2] = _schwefel_func(x, nx, Os[2*nx:3*nx], Mr[2].reshape(nx, nx) if Mr.ndim == 3 else Mr, 1, r_flag)
    fit[2] = 10000 * fit[2] / 4e+3
    fit[3] = _bent_cigar_func(x, nx, Os[3*nx:4*nx], Mr[3].reshape(nx, nx) if Mr.ndim == 3 else Mr, 1, r_flag)
    fit[3] = 10000 * fit[3] / 1e+30
    fit[4] = _ellips_func(x, nx, Os[4*nx:5*nx], Mr[4].reshape(nx, nx) if Mr.ndim == 3 else Mr, 1, r_flag)
    fit[4] = 10000 * fit[4] / 1e+10
    fit[5] = _escaffer6_func(x, nx, Os[5*nx:6*nx], Mr[5].reshape(nx, nx) if Mr.ndim == 3 else Mr, 1, r_flag)
    fit[5] = 10000 * fit[5] / 2e+7
    
    return _cf_cal(x, nx, Os, delta, bias, fit, cf_num)

def _cf08(x, nx, Os, Mr, r_flag):
    """Composition Function 8"""
    cf_num = 6
    delta = np.array([10, 20, 30, 40, 50, 60])
    bias = np.array([0, 100, 200, 300, 400, 500])
    fit = np.zeros(cf_num)
    
    fit[0] = _ackley_func(x, nx, Os[0:nx], Mr[0].reshape(nx, nx) if Mr.ndim == 3 else Mr, 1, r_flag)
    fit[0] = 1000 * fit[0] / 100
    fit[1] = _griewank_func(x, nx, Os[nx:2*nx], Mr[1].reshape(nx, nx) if Mr.ndim == 3 else Mr, 1, r_flag)
    fit[1] = 1000 * fit[1] / 100
    fit[2] = _discus_func(x, nx, Os[2*nx:3*nx], Mr[2].reshape(nx, nx) if Mr.ndim == 3 else Mr, 1, r_flag)
    fit[2] = 10000 * fit[2] / 1e+10
    fit[3] = _rosenbrock_func(x, nx, Os[3*nx:4*nx], Mr[3].reshape(nx, nx) if Mr.ndim == 3 else Mr, 1, r_flag)
    fit[4] = _happycat_func(x, nx, Os[4*nx:5*nx], Mr[4].reshape(nx, nx) if Mr.ndim == 3 else Mr, 1, r_flag)
    fit[4] = 1000 * fit[4] / 1e+3
    fit[5] = _escaffer6_func(x, nx, Os[5*nx:6*nx], Mr[5].reshape(nx, nx) if Mr.ndim == 3 else Mr, 1, r_flag)
    fit[5] = 10000 * fit[5] / 2e+7
    
    return _cf_cal(x, nx, Os, delta, bias, fit, cf_num)

def _cf09(x, nx, Os, Mr, SS, r_flag):
    """Composition Function 9"""
    cf_num = 3
    delta = np.array([10, 30, 50])
    bias = np.array([0, 100, 200])
    fit = np.zeros(cf_num)
    
    fit[0] = _hf05(x, nx, Os[0:nx], Mr[0].reshape(nx, nx) if Mr.ndim == 3 else Mr, SS[0] if SS.ndim == 2 else SS, 1, r_flag)
    fit[1] = _hf06(x, nx, Os[nx:2*nx], Mr[1].reshape(nx, nx) if Mr.ndim == 3 else Mr, SS[1] if SS.ndim == 2 else SS, 1, r_flag)
    fit[2] = _hf07(x, nx, Os[2*nx:3*nx], Mr[2].reshape(nx, nx) if Mr.ndim == 3 else Mr, SS[2] if SS.ndim == 2 else SS, 1, r_flag)
    
    return _cf_cal(x, nx, Os, delta, bias, fit, cf_num)

def _cf10(x, nx, Os, Mr, SS, r_flag):
    """Composition Function 10"""
    cf_num = 3
    delta = np.array([10, 30, 50])
    bias = np.array([0, 100, 200])
    fit = np.zeros(cf_num)
    
    fit[0] = _hf05(x, nx, Os[0:nx], Mr[0].reshape(nx, nx) if Mr.ndim == 3 else Mr, SS[0] if SS.ndim == 2 else SS, 1, r_flag)
    fit[1] = _hf08(x, nx, Os[nx:2*nx], Mr[1].reshape(nx, nx) if Mr.ndim == 3 else Mr, SS[1] if SS.ndim == 2 else SS, 1, r_flag)
    fit[2] = _hf09(x, nx, Os[2*nx:3*nx], Mr[2].reshape(nx, nx) if Mr.ndim == 3 else Mr, SS[2] if SS.ndim == 2 else SS, 1, r_flag)
    
    return _cf_cal(x, nx, Os, delta, bias, fit, cf_num)

# --- メインディスパッチャ ---

def _cec17_test_func(x, nx, func_num):
    """CEC2017テスト関数のメインディスパッチャ"""
    data = _load_cec_data(func_num, nx)
    OShift = data['OShift']
    M = data['M']
    SS = data['SS']

    func_map = {
        1: lambda: _bent_cigar_func(x, nx, OShift, M, 1, 1) + 100.0,
        3: lambda: _zakharov_func(x, nx, OShift, M, 1, 1) + 300.0,
        4: lambda: _rosenbrock_func(x, nx, OShift, M, 1, 1) + 400.0,
        5: lambda: _rastrigin_func(x, nx, OShift, M, 1, 1) + 500.0,
        6: lambda: _schaffer_F7_func(x, nx, OShift, M, 1, 1) + 600.0,
        7: lambda: _bi_rastrigin_func(x, nx, OShift, M, 1, 1) + 700.0,
        8: lambda: _step_rastrigin_func(x, nx, OShift, M, 1, 1) + 800.0,
        9: lambda: _levy_func(x, nx, OShift, M, 1, 1) + 900.0,
        10: lambda: _schwefel_func(x, nx, OShift, M, 1, 1) + 1000.0,
        11: lambda: _hf01(x, nx, OShift, M, SS, 1, 1) + 1100.0,
        12: lambda: _hf02(x, nx, OShift, M, SS, 1, 1) + 1200.0,
        13: lambda: _hf03(x, nx, OShift, M, SS, 1, 1) + 1300.0,
        14: lambda: _hf04(x, nx, OShift, M, SS, 1, 1) + 1400.0,
        15: lambda: _hf05(x, nx, OShift, M, SS, 1, 1) + 1500.0,
        16: lambda: _hf06(x, nx, OShift, M, SS, 1, 1) + 1600.0,
        17: lambda: _hf07(x, nx, OShift, M, SS, 1, 1) + 1700.0,
        18: lambda: _hf08(x, nx, OShift, M, SS, 1, 1) + 1800.0,
        19: lambda: _hf09(x, nx, OShift, M, SS, 1, 1) + 1900.0,
        20: lambda: _hf10(x, nx, OShift, M, SS, 1, 1) + 2000.0,
        21: lambda: _cf01(x, nx, OShift, M, 1) + 2100.0,
        22: lambda: _cf02(x, nx, OShift, M, 1) + 2200.0,
        23: lambda: _cf03(x, nx, OShift, M, 1) + 2300.0,
        24: lambda: _cf04(x, nx, OShift, M, 1) + 2400.0,
        25: lambda: _cf05(x, nx, OShift, M, 1) + 2500.0,
        26: lambda: _cf06(x, nx, OShift, M, 1) + 2600.0,
        27: lambda: _cf07(x, nx, OShift, M, 1) + 2700.0,
        28: lambda: _cf08(x, nx, OShift, M, 1) + 2800.0,
        29: lambda: _cf09(x, nx, OShift, M, SS, 1) + 2900.0,
        30: lambda: _cf10(x, nx, OShift, M, SS, 1) + 3000.0,
    }
    
    if func_num == 2:
        raise ValueError("Function F2 has been deleted from CEC2017.")
    
    if func_num in func_map:
        return func_map[func_num]()
    else:
        raise ValueError(f"Function number {func_num} is not defined (only 1-30, excluding 2).")

# --- BaseProblemを継承した公開クラス群 ---

class CEC2017_F1(BaseProblem):
    def __init__(self, dimension: int, shift_value: float = 0.0):
        super().__init__(dimension, -100.0, 100.0, 100.0)
        self.func_num = 1
        _load_cec_data(self.func_num, dimension)

    def evaluate(self, position: np.ndarray) -> float:
        return _cec17_test_func(position, self.dimension, self.func_num)

class CEC2017_F3(BaseProblem):
    def __init__(self, dimension: int, shift_value: float = 0.0):
        super().__init__(dimension, -100.0, 100.0, 300.0)
        self.func_num = 3
        _load_cec_data(self.func_num, dimension)

    def evaluate(self, position: np.ndarray) -> float:
        return _cec17_test_func(position, self.dimension, self.func_num)

class CEC2017_F4(BaseProblem):
    def __init__(self, dimension: int, shift_value: float = 0.0):
        super().__init__(dimension, -100.0, 100.0, 400.0)
        self.func_num = 4
        _load_cec_data(self.func_num, dimension)

    def evaluate(self, position: np.ndarray) -> float:
        return _cec17_test_func(position, self.dimension, self.func_num)

class CEC2017_F5(BaseProblem):
    def __init__(self, dimension: int, shift_value: float = 0.0):
        super().__init__(dimension, -100.0, 100.0, 500.0)
        self.func_num = 5
        _load_cec_data(self.func_num, dimension)

    def evaluate(self, position: np.ndarray) -> float:
        return _cec17_test_func(position, self.dimension, self.func_num)

class CEC2017_F6(BaseProblem):
    def __init__(self, dimension: int, shift_value: float = 0.0):
        super().__init__(dimension, -100.0, 100.0, 600.0)
        self.func_num = 6
        _load_cec_data(self.func_num, dimension)

    def evaluate(self, position: np.ndarray) -> float:
        return _cec17_test_func(position, self.dimension, self.func_num)

class CEC2017_F7(BaseProblem):
    def __init__(self, dimension: int, shift_value: float = 0.0):
        super().__init__(dimension, -100.0, 100.0, 700.0)
        self.func_num = 7
        _load_cec_data(self.func_num, dimension)

    def evaluate(self, position: np.ndarray) -> float:
        return _cec17_test_func(position, self.dimension, self.func_num)

class CEC2017_F8(BaseProblem):
    def __init__(self, dimension: int, shift_value: float = 0.0):
        super().__init__(dimension, -100.0, 100.0, 800.0)
        self.func_num = 8
        _load_cec_data(self.func_num, dimension)

    def evaluate(self, position: np.ndarray) -> float:
        return _cec17_test_func(position, self.dimension, self.func_num)

class CEC2017_F9(BaseProblem):
    def __init__(self, dimension: int, shift_value: float = 0.0):
        super().__init__(dimension, -100.0, 100.0, 900.0)
        self.func_num = 9
        _load_cec_data(self.func_num, dimension)

    def evaluate(self, position: np.ndarray) -> float:
        return _cec17_test_func(position, self.dimension, self.func_num)

class CEC2017_F10(BaseProblem):
    def __init__(self, dimension: int, shift_value: float = 0.0):
        super().__init__(dimension, -100.0, 100.0, 1000.0)
        self.func_num = 10
        _load_cec_data(self.func_num, dimension)

    def evaluate(self, position: np.ndarray) -> float:
        return _cec17_test_func(position, self.dimension, self.func_num)

class CEC2017_F11(BaseProblem):
    def __init__(self, dimension: int, shift_value: float = 0.0):
        super().__init__(dimension, -100.0, 100.0, 1100.0)
        self.func_num = 11
        _load_cec_data(self.func_num, dimension)

    def evaluate(self, position: np.ndarray) -> float:
        return _cec17_test_func(position, self.dimension, self.func_num)

class CEC2017_F12(BaseProblem):
    def __init__(self, dimension: int, shift_value: float = 0.0):
        super().__init__(dimension, -100.0, 100.0, 1200.0)
        self.func_num = 12
        _load_cec_data(self.func_num, dimension)

    def evaluate(self, position: np.ndarray) -> float:
        return _cec17_test_func(position, self.dimension, self.func_num)

class CEC2017_F13(BaseProblem):
    def __init__(self, dimension: int, shift_value: float = 0.0):
        super().__init__(dimension, -100.0, 100.0, 1300.0)
        self.func_num = 13
        _load_cec_data(self.func_num, dimension)

    def evaluate(self, position: np.ndarray) -> float:
        return _cec17_test_func(position, self.dimension, self.func_num)

class CEC2017_F14(BaseProblem):
    def __init__(self, dimension: int, shift_value: float = 0.0):
        super().__init__(dimension, -100.0, 100.0, 1400.0)
        self.func_num = 14
        _load_cec_data(self.func_num, dimension)

    def evaluate(self, position: np.ndarray) -> float:
        return _cec17_test_func(position, self.dimension, self.func_num)

class CEC2017_F15(BaseProblem):
    def __init__(self, dimension: int, shift_value: float = 0.0):
        super().__init__(dimension, -100.0, 100.0, 1500.0)
        self.func_num = 15
        _load_cec_data(self.func_num, dimension)

    def evaluate(self, position: np.ndarray) -> float:
        return _cec17_test_func(position, self.dimension, self.func_num)

class CEC2017_F16(BaseProblem):
    def __init__(self, dimension: int, shift_value: float = 0.0):
        super().__init__(dimension, -100.0, 100.0, 1600.0)
        self.func_num = 16
        _load_cec_data(self.func_num, dimension)

    def evaluate(self, position: np.ndarray) -> float:
        return _cec17_test_func(position, self.dimension, self.func_num)

class CEC2017_F17(BaseProblem):
    def __init__(self, dimension: int, shift_value: float = 0.0):
        super().__init__(dimension, -100.0, 100.0, 1700.0)
        self.func_num = 17
        _load_cec_data(self.func_num, dimension)

    def evaluate(self, position: np.ndarray) -> float:
        return _cec17_test_func(position, self.dimension, self.func_num)

class CEC2017_F18(BaseProblem):
    def __init__(self, dimension: int, shift_value: float = 0.0):
        super().__init__(dimension, -100.0, 100.0, 1800.0)
        self.func_num = 18
        _load_cec_data(self.func_num, dimension)

    def evaluate(self, position: np.ndarray) -> float:
        return _cec17_test_func(position, self.dimension, self.func_num)

class CEC2017_F19(BaseProblem):
    def __init__(self, dimension: int, shift_value: float = 0.0):
        super().__init__(dimension, -100.0, 100.0, 1900.0)
        self.func_num = 19
        _load_cec_data(self.func_num, dimension)

    def evaluate(self, position: np.ndarray) -> float:
        return _cec17_test_func(position, self.dimension, self.func_num)

class CEC2017_F20(BaseProblem):
    def __init__(self, dimension: int, shift_value: float = 0.0):
        super().__init__(dimension, -100.0, 100.0, 2000.0)
        self.func_num = 20
        _load_cec_data(self.func_num, dimension)

    def evaluate(self, position: np.ndarray) -> float:
        return _cec17_test_func(position, self.dimension, self.func_num)

class CEC2017_F21(BaseProblem):
    def __init__(self, dimension: int, shift_value: float = 0.0):
        super().__init__(dimension, -100.0, 100.0, 2100.0)
        self.func_num = 21
        _load_cec_data(self.func_num, dimension)

    def evaluate(self, position: np.ndarray) -> float:
        return _cec17_test_func(position, self.dimension, self.func_num)

class CEC2017_F22(BaseProblem):
    def __init__(self, dimension: int, shift_value: float = 0.0):
        super().__init__(dimension, -100.0, 100.0, 2200.0)
        self.func_num = 22
        _load_cec_data(self.func_num, dimension)

    def evaluate(self, position: np.ndarray) -> float:
        return _cec17_test_func(position, self.dimension, self.func_num)

class CEC2017_F23(BaseProblem):
    def __init__(self, dimension: int, shift_value: float = 0.0):
        super().__init__(dimension, -100.0, 100.0, 2300.0)
        self.func_num = 23
        _load_cec_data(self.func_num, dimension)

    def evaluate(self, position: np.ndarray) -> float:
        return _cec17_test_func(position, self.dimension, self.func_num)

class CEC2017_F24(BaseProblem):
    def __init__(self, dimension: int, shift_value: float = 0.0):
        super().__init__(dimension, -100.0, 100.0, 2400.0)
        self.func_num = 24
        _load_cec_data(self.func_num, dimension)

    def evaluate(self, position: np.ndarray) -> float:
        return _cec17_test_func(position, self.dimension, self.func_num)

class CEC2017_F25(BaseProblem):
    def __init__(self, dimension: int, shift_value: float = 0.0):
        super().__init__(dimension, -100.0, 100.0, 2500.0)
        self.func_num = 25
        _load_cec_data(self.func_num, dimension)

    def evaluate(self, position: np.ndarray) -> float:
        return _cec17_test_func(position, self.dimension, self.func_num)

class CEC2017_F26(BaseProblem):
    def __init__(self, dimension: int, shift_value: float = 0.0):
        super().__init__(dimension, -100.0, 100.0, 2600.0)
        self.func_num = 26
        _load_cec_data(self.func_num, dimension)

    def evaluate(self, position: np.ndarray) -> float:
        return _cec17_test_func(position, self.dimension, self.func_num)

class CEC2017_F27(BaseProblem):
    def __init__(self, dimension: int, shift_value: float = 0.0):
        super().__init__(dimension, -100.0, 100.0, 2700.0)
        self.func_num = 27
        _load_cec_data(self.func_num, dimension)

    def evaluate(self, position: np.ndarray) -> float:
        return _cec17_test_func(position, self.dimension, self.func_num)

class CEC2017_F28(BaseProblem):
    def __init__(self, dimension: int, shift_value: float = 0.0):
        super().__init__(dimension, -100.0, 100.0, 2800.0)
        self.func_num = 28
        _load_cec_data(self.func_num, dimension)

    def evaluate(self, position: np.ndarray) -> float:
        return _cec17_test_func(position, self.dimension, self.func_num)

class CEC2017_F29(BaseProblem):
    def __init__(self, dimension: int, shift_value: float = 0.0):
        super().__init__(dimension, -100.0, 100.0, 2900.0)
        self.func_num = 29
        _load_cec_data(self.func_num, dimension)

    def evaluate(self, position: np.ndarray) -> float:
        return _cec17_test_func(position, self.dimension, self.func_num)

class CEC2017_F30(BaseProblem):
    def __init__(self, dimension: int, shift_value: float = 0.0):
        super().__init__(dimension, -100.0, 100.0, 3000.0)
        self.func_num = 30
        _load_cec_data(self.func_num, dimension)

    def evaluate(self, position: np.ndarray) -> float:
        return _cec17_test_func(position, self.dimension, self.func_num)
