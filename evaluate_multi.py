import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sb3_contrib import MaskablePPO
from smart_fixture_env import SmartFixtureEnv3D
from scipy.interpolate import griddata

MODEL_PATH = "models_3d/final_model_3d"
DATA_DIR = "E:\\ansys_data_final"
TARGET_N = 8
TEST_ROUNDS = 4  # 测试 4 次，方便在 1行4列 中展示


def main():
    print(f"🚀 启动多轮一致性测试 (共 {TEST_ROUNDS} 轮)...")
    if not os.path.exists(MODEL_PATH + ".zip"): return

    env = SmartFixtureEnv3D(data_dir=DATA_DIR, target_n=TARGET_N)
    model = MaskablePPO.load(MODEL_PATH)

    results = []

    # 1. 运行多轮
    print("🤖 正在进行多次推理 (Deterministic=True)...")
    for r in range(TEST_ROUNDS):
        obs, _ = env.reset()
        done = False
        while not done:
            action_masks = env.action_masks()
            # 开启 deterministic=True 以检查模型在生产环境下的稳定性
            # 如果想看探索多样性，可改为 False
            action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
            obs, _, done, _, _ = env.step(action)

        uz = env.last_uz
        results.append({
            "round": r + 1,
            "max": np.max(np.abs(uz)) * 1000.0,
            "uz": uz,
            "fixtures": np.array(env.fixtures)
        })
        print(f"   Round {r + 1}: {results[-1]['max']:.4f} mm")

    # 2. 绘图
    print("\n🎨 正在绘制对比图...")
    x_raw, y_raw, z_raw = env.nodes[:, 0], env.nodes[:, 1], env.nodes[:, 2]
    range_x, range_y, range_z = np.ptp(x_raw), np.ptp(y_raw), np.ptp(z_raw)

    xi = np.linspace(x_raw.min(), x_raw.max(), 150)
    yi = np.linspace(y_raw.min(), y_raw.max(), 150)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((x_raw, y_raw), z_raw, (Xi, Yi), method='cubic')

    fig = plt.figure(figsize=(20, 6))

    # 统一色标 (取所有轮次中的最大值，公平对比)
    global_max = max([r["max"] for r in results])

    for i, res in enumerate(results):
        ax = fig.add_subplot(1, TEST_ROUNDS, i + 1, projection='3d')

        uz_mm = np.abs(res["uz"]) * 1000.0
        Ci = griddata((x_raw, y_raw), uz_mm, (Xi, Yi), method='cubic')

        norm_C = Ci / global_max
        ax.plot_surface(Xi, Yi, Zi, facecolors=cm.jet(norm_C),
                        rstride=2, cstride=2, shade=False, linewidth=0, antialiased=True)

        # 画夹具
        for idx, (fx_x, fx_y, fx_z) in enumerate(res["fixtures"]):
            color = 'lime' if idx >= 3 else 'yellow'
            z_top = fx_z + range_z * 0.15
            ax.plot([fx_x, fx_x], [fx_y, fx_y], [fx_z, z_top], color='black', lw=1)
            ax.scatter(fx_x, fx_y, z_top, c=color, s=80, edgecolors='black', zorder=100)

        ax.set_box_aspect((range_x, range_y, range_z))
        ax.set_title(f"Round {res['round']}\nMax: {res['max']:.4f} mm", fontsize=12, fontweight='bold')
        ax.axis('off')
        ax.view_init(elev=35, azim=-60)

    plt.suptitle(f"Consistency Check ({TEST_ROUNDS} Runs) | Deterministic Policy", fontsize=16)
    plt.show()


if __name__ == "__main__":
    main()