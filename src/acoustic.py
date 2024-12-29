from typing import Literal
import numpy as np
from tqdm import tqdm
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from pathlib import Path

from .util import get_fd_coeff, update_step
from .PML import create_cpml_arrays
from .wavelet import wavelet


class AcousticWave2D:
    """
    2D 声波方程模拟器 (含 CPML 吸收边界)。
    使用有限差分数值方法对声波场在均匀/非均匀介质中传播进行模拟。
    """
    def __init__(
        self,
        model: np.ndarray,          # 场地矩阵nz*nx
        materials: np.ndarray,      # 声波速度模型{0: {"v": 340.0, "rho": 1.225},1: {"v": 1500.0, "rho": 1000.0}}
        wavelet_type: Literal["ricker", "berlage", "other"] = "ricker",
        dz: float = 5.0,
        dx: float = 5.0,
        dt: float = 0.0005,
        tmax: float = 2.0,
        fm: float = 20.0,
        pml_n: int = 50,
        fd_order: int = 10,
        time_order: int = 2,
        source_position: list[tuple[int, int, float]] = None,
        receiver_positions: list[tuple[int, int]] = None
    ) -> None:
        """
        初始化 2D 声波模拟器。

        参数：
        -------
        wavelet_type : Literal["ricker", "berlage", "other"]
            震源波形类型，默认为 "ricker"。
        physical_nz : int
            不含 PML 的物理区域网格数（Z 方向）。
        physical_nx : int
            不含 PML 的物理区域网格数（X 方向）。
        dz : float
            Z 方向网格步长 (m)。
        dx : float
            X 方向网格步长 (m)。
        dt : float
            时间步长 (s)。
        tmax : float
            模拟总时间 (s)。
        fm : float
            震源子波中心频率 (Hz)。
        pml_n : int
            PML 区域厚度（单侧）。
        vmax : float
            最大波速 (m/s)。
        fd_order : int
            空间有限差分阶数（必须为偶数，默认 10）。
        time_order : int
            时间差分阶数（1 或 2）。
        """
        self.wavelet_type = wavelet_type

        # 原始物理区大小
        self.model = model
        self.physical_nz, self.physical_nx = model.shape
        
        self.nz = self.physical_nz + 2 * pml_n
        self.nx = self.physical_nx + 2 * pml_n
        self.pml_n = pml_n
        
        self.model_padded = self.pad_model_with_nearest(self.model, pml_n)

        self.dz = dz
        self.dx = dx
        self.dt = dt
        self.tmax = tmax
        self.fm = fm
        self.materials = materials
        # 使用材料中的波速和密度替换模型
        def replace_v(material_id):
            return self.materials[material_id]["v"]
        
        def replace_rho(material_id):
            return self.materials[material_id]["rho"]
        
        fun_v = np.vectorize(replace_v)
        fun_rho = np.vectorize(replace_rho)
        self.v = fun_v(self.model_padded)
        self.rho = fun_rho(self.model_padded)

        self.fd_order = fd_order
        self.time_order = time_order

        # 生成子波（可选 ricker, gaussian, other）
        self.s, self.tt = wavelet(
            wavelet_type=self.wavelet_type,
            f=self.fm,
            dt=self.dt,
            T= 2./self.fm # 保证计算一个周期
        )

        # CPML 各种吸收/修正系数
        self.a_x, self.b_x, self.a_z, self.b_z, self.k_x, self.k_z = create_cpml_arrays(
            self.nz, self.nx, self.pml_n, self.v, self.dt, self.dz
        )

        # 初始化波场与 CPML 辅助量
        self.p = np.zeros((self.nz, self.nx), dtype=np.float64)
        self.p_old = np.zeros((self.nz, self.nx), dtype=np.float64)
        self.phix = np.zeros((self.nz, self.nx), dtype=np.float64)
        self.phiz = np.zeros((self.nz, self.nx), dtype=np.float64)

        # 空间差分系数
        self.fd_coeff_array = get_fd_coeff(self.fd_order)

        # # 震源位置及权重（如果需要多个震源可在此列表添加）
        # 默认震源放置在模型中心
        if source_position is not None:
            self.z0, self.x0 = source_position
            self.z0 += pml_n
            self.x0 += pml_n
        else:
            self.z0, self.x0 = (self.nz//2, self.nx//2)
        # self.source_positions = [(self.z0, self.x0, 1.0)]
        
        # 为源构造空间分布（以源点为中心，高斯分布）
        # 高斯标准差设定，根据需要调整
        sigma_src = 5 # 可以根据需要调整
        src_grid_size = 11  # 源在3x3格点范围内分布
        source_positions = []
        for iz in range(self.z0 - src_grid_size//2, self.z0 + src_grid_size//2 + 1):
            for ix in range(self.x0 - src_grid_size//2, self.x0 + src_grid_size//2 + 1):
                dist2 = ((iz - self.z0)**2 + (ix - self.x0)**2)
                w = np.exp(-dist2/(2*sigma_src**2))
                source_positions.append((iz, ix, w))
        # 归一化权重
        wsum = sum([pos[2] for pos in source_positions])
        self.source_positions = [(pos[0], pos[1], pos[2]/wsum) for pos in source_positions]
        
        self.receiver_positions = []
        if receiver_positions is not None:
            for (rz, rx) in receiver_positions:
                self.receiver_positions.append((rz + pml_n, rx + pml_n))
        else:
            # 默认在某一列布置若干检波器
            self.receiver_positions = [(i, self.nx//2) for i in range(self.nz)]

        # 用于保存检波器的波形随时间的记录
        # 形状: [n_receivers, n_times], 先全部置零
        self.n_receivers = len(self.receiver_positions)
        self.n_times = int(self.tmax / self.dt)
        self.receiver_data = np.zeros((self.n_receivers, self.n_times), dtype=np.float64)

    def pad_model_with_nearest(self, model: np.ndarray, pml_n: int):
        """
        利用最近点外延的方式扩展模型到包含 PML 的网格。
        可以根据需要实现更复杂的插值。
        """
        nz, nx = model.shape
        big_model = np.zeros((nz + 2*pml_n, nx + 2*pml_n), dtype=int)

        # 中心部分 = 原模型
        big_model[pml_n:pml_n+nz, pml_n:pml_n+nx] = model

        # 上下外延
        for iz in range(pml_n):
            big_model[iz, pml_n:-pml_n] = model[0, :]
        for iz in range(nz + pml_n, nz + 2*pml_n):
            big_model[iz, pml_n:-pml_n] = model[-1, :]

        # 左右外延
        for ix in range(pml_n):
            big_model[:, ix] = big_model[:, pml_n]
        for ix in range(nx + pml_n, nx + 2*pml_n):
            big_model[:, ix] = big_model[:, nx + pml_n - 1]

        return big_model

    def step(self, it: int) -> None:
        """
        单步更新波场, 并记录检波器的波形。
        it: 当前时间步索引
        """
        t = it * self.dt
        # 获取此刻震源子波数值
        val = 0.0
        if (t >= self.tt[0]) and (t <= self.tt[-1]):
            idx = np.searchsorted(self.tt, t)
            if idx < len(self.tt) and abs(self.tt[idx] - t) < 1e-12:
                val = self.s[idx]

        # 更新波场
        self.p, self.p_old = update_step(
            p=self.p,
            p_old=self.p_old,
            v=self.v,
            dt=self.dt,
            dx=self.dx,
            dz=self.dz,
            coeff=self.fd_coeff_array,
            a_x=self.a_x, b_x=self.b_x,
            a_z=self.a_z, b_z=self.b_z,
            k_x=self.k_x, k_z=self.k_z,
            phix=self.phix, phiz=self.phiz,
            source_positions=self.source_positions,
            source_val=val,
            time_order=self.time_order
        )

        # 保存检波器波形
        for i, (rz, rx) in enumerate(self.receiver_positions):
            self.receiver_data[i, it] = self.p[int(rz/self.dz), int(rx/self.dx)]

    def run(self, save_snapshots=False, snapshot_interval=50) -> list[np.ndarray]:
        """
        执行正演，返回波场快照（可选）。
        """
        nt = self.n_times
        snapshots = []
        self.snapshot_interval = snapshot_interval
        for it in tqdm(range(nt), desc="Simulating"):
            self.step(it)
            # 保存快照
            if save_snapshots and (it % snapshot_interval == 0):
                snapshots.append(self.p.copy())
        return snapshots

    def show_wavelet(self) -> None:
        """
        可视化震源子波。
        """
        plt.figure(figsize=(10, 6))
        # 子波和 tt 有相同长度
        plt.plot(self.tt, self.s, label=f'{self.wavelet_type.capitalize()} Wavelet (f={self.fm} Hz)')
        plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
        plt.title(f'{self.wavelet_type.capitalize()} Wavelet')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.legend()
        plt.show()

    def visualize_snapshots(self, snapshots: list[np.ndarray],dt:int = 1, boundary: list[int] = None) -> None:
        """
        将波场快照序列保存为图像，并组合成 MP4 动画。

        参数：
        -------
        snapshots : list[np.ndarray]
            波场快照列表。
        """
        filenames = []
        # 将快照导出为图片并保存
        for i, snapshot in enumerate(snapshots):
            if i % dt != 0:
                continue
            plt.figure(figsize=(7, 6))
            if boundary is not None:
                # 四个边界
                snapshot = snapshot[boundary[0]:boundary[1], boundary[2]:boundary[3]]
            plt.imshow(snapshot, cmap="RdBu", aspect="auto", origin="upper")
            plt.colorbar(label="Amplitude")
            plt.title(f"T = {self.snapshot_interval * i * self.dt * 1000:.1f} ms")
            plt.xlabel("X")
            plt.ylabel("Z")
            filename = Path(f"./output/frame_{self.snapshot_interval * i * self.dt*1000:.1f}ms.png")
            if not filename.exists():
                filename.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(filename, dpi=100)
            plt.close()
            filenames.append(filename)

        # 合成为 MP4 动画
        mp4_filename = Path("./output/acoustic_wavefield.mp4")
        if not mp4_filename.exists():
            mp4_filename.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(mp4_filename, fps=10) as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

        print(f"MP4 动画已保存为 {mp4_filename}")
        
    def save_receiver_data(self, filename="./output/receiver_data.npy"):
        """
        保存检波器记录 (n_receivers, n_times)
        """
        filename = Path(filename)
        if not filename.exists():
            filename.parent.mkdir(parents=True, exist_ok=True)
        np.save(filename, self.receiver_data)
        print(f"检波数据已保存为 {filename}")
