import bruges as bg
import matplotlib.pyplot as plt

class ModelManager:
    """
    模型管理器，用于生成和管理地质模型。
    支持 Bruges 提供的楔形模型、分层蛋糕模型，以及自定义模型。
    """

    @staticmethod
    def generate_wedge_model(
        depth=(100, 600, 100),
        width=(200, 1600, 200),
        strat=(0, (1, 2, 1, 2, 1), 3),
        thickness=(0.5, 1),
        mode="linear",
    ):
        """
        生成楔形模型。

        参数：
        -------
        depth : tuple
            楔形深度范围 (start, end, step)。
        width : tuple
            楔形宽度范围 (start, end, step)。
        strat : tuple
            分层信息，包含顶部、底部值和中间层。
        thickness : tuple
            左右相对深度 (left, right)。0-1
        mode : str
            楔形插值模式 ("linear", "root", "power", "sigmoid")。

        返回：
        -------
        模型数组及其关键特征 (wedge, top, base, ref)。
        """
        return bg.models.wedge(depth=depth, width=width, strat=strat, thickness=thickness, mode=mode)

    @staticmethod
    def generate_cake_model(
        depth=(10, 80, 10),
        width=(10, 80, 10),
        strat=(1.48, (2.10, 2.25, 2.35), 2.40),
        thickness=(1, 0.5),
        mode="linear",
    ):
        """
        生成分层蛋糕模型。

        参数：
        -------
        depth : tuple
            深度范围 (start, end, step)。
        width : tuple
            宽度范围 (start, end, step)。
        strat : tuple
            分层信息，包含顶部、底部值和中间层。
        thickness : tuple
            每层厚度 (top, bottom)。
        mode : str
            模式 ("linear")。

        返回：
        -------
        模型数组及其关键特征 (cake, top, base, ref)。
        """
        return bg.models.wedge(depth=depth, width=width, strat=strat, thickness=thickness, mode=mode)

    @staticmethod
    def visualize_model(model_data, top, base, ref):
        """
        可视化模型并标注关键特征。

        参数：
        -------
        model_data : np.ndarray
            模型数据。
        top : np.ndarray
            顶部曲线。
        base : np.ndarray
            底部曲线。
        ref : int
            参考线位置。

        返回：
        -------
        None
        """
        plt.imshow(model_data, interpolation="none")
        plt.axvline(ref, color="k", ls="--", label="Reference Line")
        plt.plot(top, "b-", lw=4, label="Top Layer")
        plt.plot(base, "r-", lw=4, label="Base Layer")
        plt.colorbar(label="Layer Value").ax.invert_yaxis()
        plt.legend()
        plt.show()
