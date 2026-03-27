import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sb3_contrib import MaskablePPO
from smart_fixture_env import SmartFixtureEnv3D
from scipy.interpolate import griddata

# ================= 配置 =================
MODEL_PATH = "models_3d/final_model_3d"
DATA_DIR = "E:\\ansys_data_final"
TARGET_N = 8


def main():
    print("🚀 启动 3D 步进序列可视化 (Step-by-Step)...")
    if not os.path.exists(MODEL_PATH + ".zip"):
        print(f"❌ 模型未找到");
        return

    # 1. 初始化
    env = SmartFixtureEnv3D(data_dir=DATA_DIR, target_n=TARGET_N, constraint_mode="n21")
    model = MaskablePPO.load(MODEL_PATH)

    # 2. 收集每一步的数据
    print("🤖 AI 正在推理并记录过程...")
    history = []

    obs, _ = env.reset()

    # 记录 Step 0 (初始 N-2-1)
    # 注意：env.fixtures 此时已经有3个点
    uz_init = env.last_solution["uz"]
    history.append({
        "step": len(env.fixtures),
        "uz": uz_init,
        "fixtures": list(env.fixtures),
        "new_idx": -1  # 初始点无所谓新旧
    })

    done = False
    while not done:
        action_masks = env.action_masks()
        action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
        obs, _, done, _, _ = env.step(action)

        # 记录当前步
        history.append({
            "step": len(env.fixtures),
            "uz": env.last_solution["uz"],
            "fixtures": list(env.fixtures),
            "new_idx": len(env.fixtures) - 1  # 列表最后一个是新加的
        })
        print(f"   -> Step {history[-1]['step']}: Max Def = {env.last_solution['max_abs_uz_mm']:.4f} mm")

    # ================= 3. 绘图 (核心渲染逻辑) =================
    print("\n🎨 正在渲染序列图...")

    # 准备几何数据 (只计算一次)
    x_raw, y_raw, z_raw = env.nodes[:, 0], env.nodes[:, 1], env.nodes[:, 2]
    range_x, range_y, range_z = np.ptp(x_raw), np.ptp(y_raw), np.ptp(z_raw)

    # 插值网格
    xi = np.linspace(x_raw.min(), x_raw.max(), 150)  # 稍微降低分辨率提高绘图速度
    yi = np.linspace(y_raw.min(), y_raw.max(), 150)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((x_raw, y_raw), z_raw, (Xi, Yi), method='cubic')

    # 创建画布 (假设从 N=3 到 N=8 共6步 -> 2行3列)
    fig = plt.figure(figsize=(18, 10))

    for i, state in enumerate(history):
        if i >= 6: break  # 最多画6张

        ax = fig.add_subplot(2, 3, i + 1, projection='3d')

        # 数据处理
        uz_mm = np.abs(state["uz"]) * 1000.0
        max_def = np.max(uz_mm)
        Ci = griddata((x_raw, y_raw), uz_mm, (Xi, Yi), method='cubic')

        # 绘制曲面 (使用动态色标 vmax=当前最大值，以便看清每一步的分布形状)
        # 如果想看绝对数值下降，可以将 vmax 固定为 history[0]["max"]
        norm_C = Ci / max_def
        ax.plot_surface(Xi, Yi, Zi, facecolors=cm.jet(norm_C),
                        rstride=2, cstride=2, shade=False, linewidth=0, antialiased=True)

        # 绘制夹具
        fx = np.array(state["fixtures"])
        for idx, (fx_x, fx_y, fx_z) in enumerate(fx):
            # 判断是否是这一步新加的点
            is_new = (idx == state["new_idx"]) and (i > 0)
            is_init = (idx < 3)

            color = 'lime' if is_new else ('yellow' if is_init else 'white')
            # 新加点画大一点
            size = 200 if is_new else 80
            zorder = 100 if is_new else 90

            # 画支撑杆
            z_top = fx_z + range_z * 0.15
            ax.plot([fx_x, fx_x], [fx_y, fx_y], [fx_z, z_top], color='black', lw=1)
            ax.scatter(fx_x, fx_y, z_top, c=color, s=size, edgecolors='black', zorder=zorder)

            # 只给新加点标号，避免混乱
            if is_new:
                ax.text(fx_x, fx_y, z_top + range_z * 0.05, "NEW", fontsize=10, fontweight='bold', color='red',
                        zorder=101)

        # 核心：物理比例锁定
        ax.set_box_aspect((range_x, range_y, range_z))

        # 装饰
        ax.set_title(f"Step {i}: N={state['step']}\nMax Def: {max_def:.4f} mm", fontsize=12, fontweight='bold')
        ax.axis('off')  # 隐藏坐标轴，更美观
        ax.view_init(elev=35, azim=-60)

    plt.suptitle(
        f"AI Optimization Sequence (True Scale)\nStart: {history[0]['step']} pts -> End: {history[-1]['step']} pts",
        fontsize=18)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()