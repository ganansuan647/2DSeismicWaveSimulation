import bruges as bg
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from shapely.affinity import translate

from src.acoustic import AcousticWave2D
from src.model import ModelManager
from utils.dam import create_gravity_dam_mask

def add_dam(model,dz,dx,dam_polygon,dam_position,ground_h):
    nz,nx = model.shape
    dam_mask = translate(dam_polygon, xoff = dam_position[0], yoff = dam_position[1])
    minx_dam = dam_mask.bounds[0]
    minz_dam = dam_mask.bounds[1]
    maxz_dam = dam_mask.bounds[3]
    for i in range(nz):
        for j in range(nx):
            point_pos = (j*dx, ground_h-i*dz)
            if dam_mask.contains(Point(*point_pos)):
                model[i,j] = 4
            else:
                # 上游侧赋予水
                if point_pos[0] < dam_position[0]:
                    if point_pos[1] > minz_dam and point_pos[1] < maxz_dam:
                        model[i,j] = 1
    return model
    


if __name__ == "__main__":
    materials = {
            0: {"v": 340, "rho": 1.225},      # 空气
            1: {"v": 1500.0, "rho": 1000.0},    # 水
            2: {"v": 1800.0, "rho": 2300.0},    # 岩石 1
            3: {"v": 2500.0, "rho": 3200.0},    # 岩石 2
            4: {"v": 3000.0, "rho": 2500.0},    # 混凝土
            # ... 可以继续添加
        }

    # 2. 生成模型 (模型矩阵元素为材料编号)
    manager = ModelManager()
    # 底座蛋糕模型
    model = manager.generate_cake_model(depth=(110,300,2),
                                        width=(0,500,2),
                                        strat=(0, (2, 3), 3),
                                        thickness=(1, 1),
                                        mode="linear")
    # 上方混凝土重力坝
    H = 100           # 坝高
    W = 15            # 坝顶宽
    slope_up = 0.2   # 上游边坡
    slope_dn = 0.5   # 下游边坡
    ratio_up = 0.4   # 上游斜面占坝高的比例
    ratio_dn = 0.8   # 下游斜面占坝高的比例

    dam_polygon = create_gravity_dam_mask(H, W, slope_up, slope_dn, ratio_up, ratio_dn)
    # 应用mask
    wedge = add_dam(model.wedge,1,1,dam_polygon,(430,0),110)
    
    manager.visualize_model(wedge, model.top, model.base, 0)
    
    # 3. 定义震源、检波器位置(不含 PML 偏移)
    source_position = (370, 20)  # 在物理区 Z=50, X=150 处放一个源
    # 比如在 Z=60 到 Z=100 间隔 5 点；X=140
    receiver_positions = [(z, 140) for z in range(60, 101, 5)]

    # 4. 创建波场模拟器
    simulator = AcousticWave2D(
        model=model.wedge,  # 使用楔形模型
        materials=materials,
        wavelet_type="ricker",
        dz=2.0,
        dx=2.0,
        dt=0.0005,
        tmax=3.0,         # 时间短一些，快速演示
        fm=20.0,
        pml_n=50,         # 30网格厚度的PML
        fd_order=10,
        time_order=2,
        source_position=source_position,
        receiver_positions=receiver_positions,
    )

    # 可视化震源子波
    simulator.show_wavelet()

    # 5. 运行正演，获取波场快照
    snapshots = simulator.run(save_snapshots=True, snapshot_interval=50)

    # 6. 可选：将快照做成动画
    simulator.visualize_snapshots(snapshots)

    # 7. 保存检波器记录
    simulator.save_receiver_data("./output/receiver_data.npy")
    
    # manager = ModelManager()
    # w, top, base, ref = manager.generate_wedge_model(mode="linear")
    # manager.visualize_model(w, top, base, ref)

    # cake, top, base, ref = manager.generate_cake_model()
    # manager.visualize_model(cake, top, base, ref)
    
    # modes = ["linear", "root", "power", "sigmoid"]
    # fig, axs = plt.subplots(ncols=len(modes), figsize=(15, 5))
    # for ax, mode in zip(axs, modes):
    #     w, top, base, ref = manager.generate_wedge_model(mode=mode)
    #     ax.imshow(w, interpolation="none")
    #     ax.axvline(ref, color="k", ls="--")
    #     ax.plot(top, "b-", lw=4)
    #     ax.plot(base, "r-", lw=4)
    #     ax.set_title(mode)
    # plt.show()
