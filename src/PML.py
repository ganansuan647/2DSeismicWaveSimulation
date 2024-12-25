import numpy as np

def create_cpml_arrays(
    nz: int,
    nx: int,
    pml_n: int,
    v: np.ndarray,   # 局部速度矩阵 (nz, nx)
    dt: float,
    dz: float,
    R: float = 1e-10
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    构建 2D CPML（Convolutional PML）所需的吸收系数数组，考虑局部速度。
    """
    # 其余可调参数
    alpha_max = np.pi * 20.0
    k_max = 1.0
    m = 2  # CPML 指数

    # 初始化二维数组
    sigma_x = np.zeros((nz, nx), dtype=np.float64)
    alpha_x = np.zeros((nz, nx), dtype=np.float64)
    k_x     = np.ones((nz, nx),  dtype=np.float64)

    sigma_z = np.zeros((nz, nx), dtype=np.float64)
    alpha_z = np.zeros((nz, nx), dtype=np.float64)
    k_z     = np.ones((nz, nx),  dtype=np.float64)

    # ======================== 新增一个用于逐点计算的函数 ========================
    def pml_profile(distance: float, v_local: float) -> tuple[float, float, float]:
        """
        根据某一点与PML边界的距离(distance)和局部速度(v_local)，
        返回标量形式的 (sigma, alpha, k)。
        """
        if distance <= 0:  # 不在PML区域
            return 0.0, 0.0, 1.0

        # 归一化深度
        d_norm = distance / pml_n  # 范围 [0, 1]
        # 按照简单配方计算 sigma
        # 这里乘了 1.5 是参考某些文献可能有的经验系数，也可不乘
        # 注意：-np.log(R) * v_local / (pml_n * dz) 是“局部AA”，只针对一个点
        AA_local = -np.log(R) * v_local * 1.5 / (pml_n * dz)
        sigma_val = AA_local * (d_norm ** m)
        alpha_val = alpha_max * (1.0 - d_norm)
        k_val     = 1.0 + (k_max - 1.0) * (d_norm ** m)
        return sigma_val, alpha_val, k_val

    # ========================= 计算 Z 方向 PML 参数 =========================
    for iz in range(nz):
        for ix in range(nx):
            # 根据 iz 距离最近的Z向边界
            if iz < pml_n:
                # 距离顶部边界
                dist_z = pml_n - iz
            elif iz >= nz - pml_n:
                # 距离底部边界
                dist_z = iz - (nz - pml_n - 1)
            else:
                dist_z = 0

            # 局部速度
            v_local = v[iz, ix]

            sigma_val, alpha_val, k_val = pml_profile(dist_z, v_local)
            sigma_z[iz, ix] = sigma_val
            alpha_z[iz, ix] = alpha_val
            k_z[iz, ix]     = k_val

    # ========================= 计算 X 方向 PML 参数 =========================
    for iz in range(nz):
        for ix in range(nx):
            if ix < pml_n:
                dist_x = pml_n - ix
            elif ix >= nx - pml_n:
                dist_x = ix - (nx - pml_n - 1)
            else:
                dist_x = 0

            v_local = v[iz, ix]
            sigma_val, alpha_val, k_val = pml_profile(dist_x, v_local)
            sigma_x[iz, ix] = sigma_val
            alpha_x[iz, ix] = alpha_val
            k_x[iz, ix]     = k_val

    # ======================= 计算 a_x, b_x, a_z, b_z =======================
    b_z = np.exp(-(sigma_z / k_z + alpha_z) * dt)
    # 避免出现 0 除法
    denom_z = sigma_z + alpha_z * k_z
    denom_z[denom_z == 0] = 1e-16  # 或其它极小值
    a_z = (sigma_z * (b_z - 1.0)) / denom_z
    a_z = np.nan_to_num(a_z)

    b_x = np.exp(-(sigma_x / k_x + alpha_x) * dt)
    denom_x = sigma_x + alpha_x * k_x
    denom_x[denom_x == 0] = 1e-16
    a_x = (sigma_x * (b_x - 1.0)) / denom_x
    a_x = np.nan_to_num(a_x)

    return a_x, b_x, a_z, b_z, k_x, k_z


def safe_division(num: np.ndarray, denom: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """
    安全除法，避免除以零的情况。

    参数：
    -------
    num : np.ndarray
        分子。
    denom : np.ndarray
        分母。
    eps : float, 可选
        防止分母为 0 的极小值。

    返回：
    -------
    np.ndarray
        安全除法结果。
    """
    return num / (denom + eps)