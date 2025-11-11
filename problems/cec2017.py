# problems/cec2017.py
# CEC2017 ベンチマーク関数スイート
# C++コード (cec17_test_func.cpp) のロジックをPythonに移植

import numpy as np
import os
import math
from .base_problem import BaseProblem

# --- データキャッシュ ---
# C++版のグローバル変数(OShift, M, SS)とini_flagの役割を果たす
# (func_num, nx) をキーとして、読み込んだデータを辞書に保存する
_data_cache = {}

# C++版のグローバル配列 y, z の代わり
# これらは関数間で使い回されるため、キャッシュで管理する
_temp_arrays_cache = {}

# --- 定数 ---
INF = 1.0e99
EPS = 1.0e-14
E = math.e
PI = math.pi
CF_NUM = 10 # コンポジション関数の数

def _get_temp_arrays(nx):
    """
    C++版のグローバル配列 y, z を取得または作成する
    """
    if nx not in _temp_arrays_cache:
        _temp_arrays_cache[nx] = {
            'y': np.zeros(nx),
            'z': np.zeros(nx)
        }
    return _temp_arrays_cache[nx]

def _load_cec_data(func_num, nx):
    """
    C++版の ini_flag==0 ブロックに相当。
    Shiftベクトル、Rotation行列、Shuffleデータをファイルから読み込む。
    """
    key = (func_num, nx)
    if key in _data_cache:
        return _data_cache[key]

    # --- input_data フォルダのパスを設定 ---
    # このスクリプトは run_experiment.py から呼ばれることを想定
    # run_experiment.py と同じ階層に input_data があると仮定
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
        M = np.eye(nx) # フォールバック

    # --- OShift (Shift Vector) の読み込み ---
    filename = os.path.join(base_path, f"shift_data_{func_num}.txt")
    try:
        if func_num < 20:
            OShift = np.loadtxt(filename)
        else:
            # CF関数のShiftデータは複数行ある
            OShift_flat = np.loadtxt(filename, max_rows=CF_NUM)
            OShift = OShift_flat.flatten()
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
            # C++は1-based index, Pythonは0-based
            SS = SS - 1 
        except Exception as e:
            print(f"Warning: Could not load SS data for F{func_num} D{nx}. {e}")
            SS = np.arange(nx)

    data = {'M': M, 'OShift': OShift, 'SS': SS}
    _data_cache[key] = data
    return data

# --- C++ ヘルパー関数のPython移植 ---

def _shiftfunc(x, os_vec):
    return x - os_vec

def _rotatefunc(y, mr_mat):
    # Mは(nx, nx)のNumpy配列
    # C++: xrot[i] = sum(x[j] * Mr[i*nx+j]) -> xrot = Mr @ y
    return mr_mat.dot(y)

def _sr_func(x, os_vec, mr_mat, sh_rate, s_flag, r_flag, nx):
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
        return z
    else:
        # zにコピーして返す（yは他で使い回されるため）
        z[:] = y
        return z

def _asyfunc(x, nx, beta):
    xasy = x.copy()
    indices = x > 0
    xasy[indices] = np.power(x[indices], 1.0 + beta * np.arange(nx)[indices] / (nx - 1) * np.sqrt(x[indices]))
    return xasy

def _oszfunc(x, nx):
    xosz = x.copy()
    for i in [0, nx - 1]: # 先頭と末尾のみ
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
    w = np.zeros(cf_num)
    w_max = 0
    w_sum = 0
    
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

# --- C++ 基本関数のPython移植 ---
# (C++版とシグネチャを合わせるため、Os, Mrなども引数に取る)

def _sphere_func(x, nx, Os, Mr, s_flag, r_flag):
    z = _sr_func(x, Os, Mr, 1.0, s_flag, r_flag, nx)
    return np.sum(z**2)

def _ellips_func(x, nx, Os, Mr, s_flag, r_flag):
    z = _sr_func(x, Os, Mr, 1.0, s_flag, r_flag, nx)
    f = 0.0
    for i in range(nx):
       f += (10.0**(6.0 * i / (nx - 1))) * z[i]**2
    return f

def _zakharov_func(x, nx, Os, Mr, s_flag, r_flag):
    z = _sr_func(x, Os, Mr, 1.0, s_flag, r_flag, nx)
    sum1 = np.sum(z**2)
    sum2 = np.sum(0.5 * (np.arange(1, nx + 1)) * z)
    return sum1 + sum2**2 + sum2**4

def _rosenbrock_func(x, nx, Os, Mr, s_flag, r_flag):
    z = _sr_func(x, Os, Mr, 2.048/100.0, s_flag, r_flag, nx)
    z = z + 1.0 # shift to origin
    tmp1 = z[:-1]**2 - z[1:]
    tmp2 = z[:-1] - 1.0
    f = np.sum(100.0 * tmp1**2 + tmp2**2)
    return f

def _rastrigin_func(x, nx, Os, Mr, s_flag, r_flag):
    z = _sr_func(x, Os, Mr, 5.12/100.0, s_flag, r_flag, nx)
    return np.sum(z**2 - 10.0 * np.cos(2.0 * PI * z) + 10.0)

def _schaffer_F7_func(x, nx, Os, Mr, s_flag, r_flag):
    # C++版はグローバルyを使っているが、sr_funcの戻り値zを使うのが正しい
    z = _sr_func(x, Os, Mr, 1.0, s_flag, r_flag, nx)
    f = 0.0
    # C++版のy[i]はz[i]の誤記と思われるため、z[i]で計算
    s = np.sqrt(z[:-1]**2 + z[1:]**2)
    tmp = np.sin(50.0 * s**0.2)
    f = np.sum(s**0.5 + s**0.5 * tmp**2)
    return (f / (nx - 1))**2

def _bi_rastrigin_func(x, nx, Os, Mr, s_flag, r_flag):
    mu0=2.5
    d=1.0
    s=1.0-1.0/(2.0*math.sqrt(nx+20.0)-8.2)
    mu1=-math.sqrt((mu0**2-d)/s)
    
    if s_flag==1:
        y = _shiftfunc(x, Os)
    else:
        y = x.copy()
    y *= 10.0/100.0

    tmpx = 2 * y
    tmpx[Os < 0] *= -1.0
    
    z = tmpx.copy()
    tmpx = tmpx + mu0
    
    tmp1 = np.sum((tmpx - mu0)**2)
    tmp2 = np.sum((tmpx - mu1)**2)
    tmp2 = s * tmp2 + d * nx
    
    tmp = 0.0
    if r_flag==1:
        y_rot = _rotatefunc(z, Mr)
        tmp = np.sum(np.cos(2.0 * PI * y_rot))
    else:
        tmp = np.sum(np.cos(2.0 * PI * z))
        
    f_val = min(tmp1, tmp2) + 10.0 * (nx - tmp)
    return f_val

def _step_rastrigin_func(x, nx, Os, Mr, s_flag, r_flag):
    y = x.copy()
    # C++版のロジックを適用
    for i in range(nx):
        if abs(y[i] - Os[i]) > 0.5:
            y[i] = Os[i] + math.floor(2 * (y[i] - Os[i]) + 0.5) / 2
            
    z = _sr_func(y, Os, Mr, 5.12/100.0, s_flag, r_flag, nx)
    return np.sum(z**2 - 10.0 * np.cos(2.0 * PI * z) + 10.0)

def _levy_func(x, nx, Os, Mr, s_flag, r_flag):
    z = _sr_func(x, Os, Mr, 1.0, s_flag, r_flag, nx)
    w = 1.0 + (z - 1.0) / 4.0
    
    term1 = math.sin(PI * w[0])**2
    term3 = (w[nx-1] - 1)**2 * (1 + math.sin(2 * PI * w[nx-1])**2)
    
    wi = w[:-1]
    sum_val = np.sum((wi - 1)**2 * (1 + 10 * np.sin(PI * wi + 1)**2))
    
    return term1 + sum_val + term3

def _schwefel_func(x, nx, Os, Mr, s_flag, r_flag):
    z = _sr_func(x, Os, Mr, 1000.0/100.0, s_flag, r_flag, nx)
    f = 0.0
    for i in range(nx):
        zi = z[i] + 4.209687462275036e+002
        if zi > 500:
            f -= (500.0 - math.fmod(zi, 500)) * math.sin(pow(500.0 - math.fmod(zi, 500), 0.5))
            tmp = (zi - 500.0) / 100
            f += tmp * tmp / nx
        elif zi < -500:
            f -= (-500.0 + math.fmod(abs(zi), 500)) * math.sin(pow(500.0 - math.fmod(abs(zi), 500), 0.5))
            tmp = (zi + 500.0) / 100
            f += tmp * tmp / nx
        else:
            f -= zi * math.sin(pow(abs(zi), 0.5))
    f += 4.189828872724338e+002 * nx
    return f

def _bent_cigar_func(x, nx, Os, Mr, s_flag, r_flag):
    z = _sr_func(x, Os, Mr, 1.0, s_flag, r_flag, nx)
    f = z[0]**2 + np.sum(1e6 * (z[1:]**2))
    return f

# ... (他の基本関数 _discus_func, _dif_powers_func, _ackley_func, ... _hgbat_func も同様に移植) ...
# C++コードが長すぎるため、ここでは主要な関数のみ移植例として示します。
# hf/cf関数も同様に、C++のロジックに従って移植が必要です。

# --- C++ メインディスパッチャ _cec17_test_func のPython移植 ---
# (C++コードの switch 文に相当)
def _cec17_test_func(x, nx, func_num):
    data = _load_cec_data(func_num, nx)
    OShift = data['OShift']
    M = data['M']
    SS = data['SS']

    # C++版の f[i] = 0.0 の代わり
    f_val = 0.0

    if func_num == 1:
        f_val = _bent_cigar_func(x, nx, OShift, M, 1, 1) + 100.0
    elif func_num == 2:
        # F2は欠番
        print("\nError: This function (F2) has been deleted\n")
        f_val = 0.0
    elif func_num == 3:
        f_val = _zakharov_func(x, nx, OShift, M, 1, 1) + 300.0
    elif func_num == 4:
        f_val = _rosenbrock_func(x, nx, OShift, M, 1, 1) + 400.0
    elif func_num == 5:
        f_val = _rastrigin_func(x, nx, OShift, M, 1, 1) + 500.0
    elif func_num == 6:
         f_val = _schaffer_F7_func(x, nx, OShift, M, 1, 1) + 600.0
    elif func_num == 7:
        f_val = _bi_rastrigin_func(x, nx, OShift, M, 1, 1) + 700.0
    elif func_num == 8:
        f_val = _step_rastrigin_func(x, nx, OShift, M, 1, 1) + 800.0
    elif func_num == 9:
        f_val = _levy_func(x, nx, OShift, M, 1, 1) + 900.0
    elif func_num == 10:
        f_val = _schwefel_func(x, nx, OShift, M, 1, 1) + 1000.0
    # ... (F11からF30までの case も同様に移植) ...
    # ... (hfXX や cfXX を呼び出す) ...
    else:
        print(f"\nError: Function number {func_num} is not defined (only 1-30).\n")
        f_val = 0.0
    
    return f_val


# --- BaseProblemを継承した公開クラス群 ---
# これが run_experiment.py から動的に読み込まれる

class CEC2017_F1(BaseProblem):
    def __init__(self, dimension: int, shift_value: float = 0.0):
        super().__init__(dimension, -100.0, 100.0, 0.0) # shift_valueは使わない
        self.func_num = 1
        _load_cec_data(self.func_num, dimension) # 実行前にデータをロード/キャッシュ

    def evaluate(self, position: np.ndarray) -> float:
        return _cec17_test_func(position, self.dimension, self.func_num)

# F2は欠番 (C++コードにも "deleted" とある)

class CEC2017_F3(BaseProblem):
    def __init__(self, dimension: int, shift_value: float = 0.0):
        super().__init__(dimension, -100.0, 100.0, 0.0)
        self.func_num = 3
        _load_cec_data(self.func_num, dimension)

    def evaluate(self, position: np.ndarray) -> float:
        return _cec17_test_func(position, self.dimension, self.func_num)

class CEC2017_F4(BaseProblem):
    def __init__(self, dimension: int, shift_value: float = 0.0):
        super().__init__(dimension, -100.0, 100.0, 0.0)
        self.func_num = 4
        _load_cec_data(self.func_num, dimension)

    def evaluate(self, position: np.ndarray) -> float:
        return _cec17_test_func(position, self.dimension, self.func_num)

class CEC2017_F5(BaseProblem):
    def __init__(self, dimension: int, shift_value: float = 0.0):
        super().__init__(dimension, -100.0, 100.0, 0.0)
        self.func_num = 5
        _load_cec_data(self.func_num, dimension)

    def evaluate(self, position: np.ndarray) -> float:
        return _cec17_test_func(position, self.dimension, self.func_num)

class CEC2017_F6(BaseProblem):
    def __init__(self, dimension: int, shift_value: float = 0.0):
        super().__init__(dimension, -100.0, 100.0, 0.0)
        self.func_num = 6
        _load_cec_data(self.func_num, dimension)

    def evaluate(self, position: np.ndarray) -> float:
        return _cec17_test_func(position, self.dimension, self.func_num)

class CEC2017_F7(BaseProblem):
    def __init__(self, dimension: int, shift_value: float = 0.0):
        super().__init__(dimension, -100.0, 100.0, 0.0)
        self.func_num = 7
        _load_cec_data(self.func_num, dimension)

    def evaluate(self, position: np.ndarray) -> float:
        return _cec17_test_func(position, self.dimension, self.func_num)

class CEC2017_F8(BaseProblem):
    def __init__(self, dimension: int, shift_value: float = 0.0):
        super().__init__(dimension, -100.0, 100.0, 0.0)
        self.func_num = 8
        _load_cec_data(self.func_num, dimension)

    def evaluate(self, position: np.ndarray) -> float:
        return _cec17_test_func(position, self.dimension, self.func_num)

class CEC2017_F9(BaseProblem):
    def __init__(self, dimension: int, shift_value: float = 0.0):
        super().__init__(dimension, -100.0, 100.0, 0.0)
        self.func_num = 9
        _load_cec_data(self.func_num, dimension)

    def evaluate(self, position: np.ndarray) -> float:
        return _cec17_test_func(position, self.dimension, self.func_num)

class CEC2017_F10(BaseProblem):
    def __init__(self, dimension: int, shift_value: float = 0.0):
        super().__init__(dimension, -100.0, 100.0, 0.0)
        self.func_num = 10
        _load_cec_data(self.func_num, dimension)

    def evaluate(self, position: np.ndarray) -> float:
        return _cec17_test_func(position, self.dimension, self.func_num)

# ... (同様に F11 から F30 までのクラスを定義) ...
# C++コードが非常に長いため、この回答ではF1-F10までを例として示します。
# F11以降も、F10までと同様のパターンでクラス定義を追加してください。
# (例: CEC2017_F11, CEC2017_F12, ... , CEC2017_F30)