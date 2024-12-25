from src.acoustic import AcousticWave2D
from src.model import ModelManager

if __name__ == "__main__":
    materials = {
            0: {"v": 340, "rho": 1.225},      # 空气
            1: {"v": 1500.0, "rho": 1000.0},    # 水
            2: {"v": 2500.0, "rho": 2300.0},    # 岩石 1
            3: {"v": 4500.0, "rho": 2800.0},    # 岩石 2
            4: {"v": 6000.0, "rho": 3200.0},    # 岩石 3
            # ... 可以继续添加
        }

    # 2. 生成模型 (模型矩阵元素为材料编号)
    manager = ModelManager()
    # 层状蛋糕模型
    # model = manager.generate_cake_model(depth=(10,400,10),
    #                                     width=(10,500,10),
    #                                     strat=(0, (1, 2, 3, 2, 3), 4),
    #                                     thickness=(1, 1),
    #                                     mode="linear")
    
    # 楔形模型
    model = manager.generate_wedge_model(depth=(20, 400, 10),
                                        width=(200, 500, 10),
                                        strat=(0, (1, 2, 3, 2, 3), 4),
                                        thickness=(0.1, 1),
                                        mode="linear")
    manager.visualize_model(*model)
    
    # 3. 定义震源、检波器位置(不含 PML 偏移)
    source_position = (22, 350)  # 在物理区 Z=50, X=150 处放一个源
    # 比如在 Z=60 到 Z=100 间隔 5 点；X=140
    receiver_positions = [(z, 140) for z in range(60, 101, 5)]

    # 4. 创建波场模拟器
    simulator = AcousticWave2D(
        model=model.wedge,  # 使用楔形模型
        materials=materials,
        wavelet_type="berlage",
        dz=5.0,
        dx=5.0,
        dt=0.001,
        tmax=3.0,         # 时间短一些，快速演示
        fm=20.0,
        pml_n=50,         # 50网格厚度的PML
        fd_order=8,
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

    # 7. 可选：保存检波器记录
    simulator.save_receiver_data("./output/receiver_data.npy")
    
    # 其他可选的模型可视化
    # manager = ModelManager()
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
