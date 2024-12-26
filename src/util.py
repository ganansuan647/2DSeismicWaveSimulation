from numba import njit, prange
import numpy as np

def get_fd_coeff(order: int) -> np.ndarray:
    """
    获取指定阶数的有限差分系数。

    参数：
    -------
    order : int
        差分阶数，必须为偶数且在预定义的支持范围内。

    返回：
    -------
    coeffs : np.ndarray
        指定阶数的有限差分系数。
    """
    # 定义有限差分系数
    coefficients_map = {
        2: np.array([-0.5, 0.0, 0.5]),
        4: np.array([-1/12,  2/3,  0.0, -2/3,  1/12]),
        6: np.array([ 1/60, -3/20,  3/4,  0.0, -3/4,  3/20, -1/60]),
        8: np.array([-1/280,  4/105, -1/5,  4/5,  0.0, -4/5,  1/5, -4/105,  1/280]),
        10: np.array([-1/1260,  5/504, -5/84,  5/21,  0.0, -5/21,  5/84, -5/504,  1/1260])
    }
    if order not in coefficients_map:
        raise ValueError(f"Unsupported order: {order}. Supported orders: {list(coefficients_map.keys())}.")
    return coefficients_map[order]

@njit(parallel=True, fastmath=True)
def fd_derivative(p: np.ndarray,
                  coeff: np.ndarray,
                  step: float,
                  axis: int) -> np.ndarray:
    """
    计算标量场 p 在指定轴上的有限差分一阶导数。

    参数：
    -------
    p : np.ndarray
        标量场 (nz, nx)。
    coeff : np.ndarray
        有限差分系数。
    step : float
        网格步长 (dx 或 dz)。
    axis : int
        0 表示对 Z 方向差分，1 表示对 X 方向差分。

    返回：
    -------
    dp : np.ndarray
        一阶导数结果 (nz, nx)。
    """
    half_len = len(coeff) // 2
    nz, nx = p.shape
    dp = np.zeros_like(p)
    for i in prange(nz):
        for j in prange(nx):
            val = 0.0
            for k in range(len(coeff)):
                if axis == 0:  # Z 向差分
                    ii = i + (k - half_len)
                    if 0 <= ii < nz:
                        val += coeff[k] * p[ii, j]
                else:  # X 向差分
                    jj = j + (k - half_len)
                    if 0 <= jj < nx:
                        val += coeff[k] * p[i, jj]
            dp[i, j] = val / step
    return dp

@njit(parallel=True, fastmath=True)
def update_step(p: np.ndarray,
                p_old: np.ndarray,
                v: np.ndarray,
                dt: float,
                dx: float,
                dz: float,
                coeff: np.ndarray,
                a_x: np.ndarray,
                b_x: np.ndarray,
                a_z: np.ndarray,
                b_z: np.ndarray,
                k_x: np.ndarray,
                k_z: np.ndarray,
                phix: np.ndarray,
                phiz: np.ndarray,
                source_positions: list,
                source_val: float,
                time_order: int
                ) -> tuple[np.ndarray, np.ndarray]:
    """
    完成时间步内的波场更新，支持 CPML 边界处理并加震源。

    参数：
    -------
    p : np.ndarray
        当前时刻的波场 (nz, nx)。
    p_old : np.ndarray
        上一时刻的波场 (nz, nx)。
    v : np.ndarray
        波速模型 (nz, nx)。
    dt : float
        时间步长 (s)。
    dx : float
        X 方向网格步长 (m)。
    dz : float
        Z 方向网格步长 (m)。
    coeff : np.ndarray
        有限差分系数。
    a_x, b_x, a_z, b_z, k_x, k_z : np.ndarray
        CPML 吸收系数和介质参数。
    phix, phiz : np.ndarray
        CPML 中的辅助记忆变量。
    source_positions : list
        震源位置及权重列表 [(z_idx, x_idx, weight), ...]。
    source_val : float
        当前时刻震源子波数值。
    time_order : int
        时间阶数 (1 或 2)，用于决定差分形式。

    返回：
    -------
    p_new : np.ndarray
        更新后的波场 (nz, nx)。
    p : np.ndarray
        更新前的波场，即下一时刻的 p_old。
    """
    # 先计算空间一阶导数
    dp_dx = fd_derivative(p, coeff, dx, axis=1)
    dp_dz = fd_derivative(p, coeff, dz, axis=0)

    nz, nx = p.shape

    # CPML 辅助量 phix, phiz 的更新
    for i in prange(nz):
        for j in prange(nx):
            phix[i, j] = b_x[i, j] * phix[i, j] + a_x[i, j] * dp_dx[i, j]
            phiz[i, j] = b_z[i, j] * phiz[i, j] + a_z[i, j] * dp_dz[i, j]

    # 加入震源
    for z_idx, x_idx, weight in source_positions:
        if 0 <= z_idx < nz and 0 <= x_idx < nx:
            p[z_idx, x_idx] += source_val * weight

    # 实际一阶导数中加上 CPML 修正项
    for i in prange(nz):
        for j in prange(nx):
            dp_dx[i, j] = (dp_dx[i, j] + phix[i, j]) / k_x[i, j]
            dp_dz[i, j] = (dp_dz[i, j] + phiz[i, j]) / k_z[i, j]

    # 再做一次导数得到 Laplacian
    d2p_dx2 = fd_derivative(dp_dx, coeff, dx, axis=1)
    d2p_dz2 = fd_derivative(dp_dz, coeff, dz, axis=0)
    lap = d2p_dx2 + d2p_dz2

    # 时间差分（1阶或2阶）
    if time_order == 1:
        p_new = p + (v ** 2) * (dt ** 2) * lap
    elif time_order == 2:
        p_new = 2.0 * p - p_old + (v ** 2) * (dt ** 2) * lap
    else:
        p_new = 2.0 * p - p_old + (v ** 2) * (dt ** 2) * lap

    return p_new, p

@njit(parallel=True, fastmath=True)
def update_acoustic_staggered(
    vx, vz, P,
    rho_vx, rho_vz, rho_p,   # 可能在各自网格上定义好的密度值
    K_p,                     # 压力点处的体积模量
    dx, dz, dt,
    coeff_x, coeff_z        # 分别用于 x 向、z 向的一阶差分系数
):
    """
    单步更新 2D 交错网格下的速度 (vx, vz) 和压力 P。
    这里的 rho_vx, rho_vz, rho_p, K_p 等都可以是二维数组, 显式包含介质变化。
    压力场P放在整点上，速度场vx, vz放在半整点上。
            j-1        j         j+1
    i+0.5   *----vz----*----vz----*
            |          |          |
    i      vx        (P)        vx
            |          |          |
    i-0.5   *----vz----*----vz----*
    """
    # --- 1) 更新 v_x (对 P 做 x 向差分) ---
    # P 在 (i,j)， v_x 在 (i,j+0.5)，所以要对 P 做插值或直接采用交错索引。
    # 这里为简化，假设数组已做好对齐(仅演示原理)，实际需注意下标偏移。
    dP_dx = fd_derivative(P, coeff_x, dx, axis=1)
    # vx^new = vx^old - dt * (1/rho_vx) * dP/dx
    for iz in prange(vx.shape[0]):
        for ix in prange(vx.shape[1]):
            vx[iz, ix] = vx[iz, ix] - dt * (1.0 / rho_vx[iz, ix]) * dP_dx[iz, ix]

    # --- 2) 更新 v_z (对 P 做 z 向差分) ---
    dP_dz = fd_derivative(P, coeff_z, dz, axis=0)
    for iz in prange(vz.shape[0]):
        for ix in prange(vz.shape[1]):
            vz[iz, ix] = vz[iz, ix] - dt * (1.0 / rho_vz[iz, ix]) * dP_dz[iz, ix]

    # --- 3) 更新 P (对 v_x, v_z 求散度) ---
    # P 在 (i,j)， v_x 在 (i,j+0.5)， v_z 在 (i+0.5,j)
    # 因此散度 approx d(vx)/dx + d(vz)/dz
    dvx_dx = fd_derivative(vx, coeff_x, dx, axis=1)
    dvz_dz = fd_derivative(vz, coeff_z, dz, axis=0)
    for iz in prange(P.shape[0]):
        for ix in prange(P.shape[1]):
            div_v = dvx_dx[iz, ix] + dvz_dz[iz, ix]
            P[iz, ix] = P[iz, ix] - dt * K_p[iz, ix] * div_v

    return vx, vz, P