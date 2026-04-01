import os

# 解决 OMP 冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sb3_contrib import MaskablePPO
from smart_fixture_env import SmartFixtureEnv3D
from scipy.interpolate import griddata
from scipy.spatial import KDTree

# ================= 配置 =================
# MODEL_PATH = "models_3d/final_model_3d"
MODEL_PATH = "models_3d/best/best_model"
DATA_DIR = "E:\\ansys_data_final"
TARGET_N = 8


def get_smart_manual_baseline(env):
    """
    🟢 [逻辑修复] 智能生成人工基准
    只在 XY 平面上寻找最近的有效点，避免 2D vs 3D 维度报错
    """
    candidates = env.candidates

    # 🟢 [关键修改] 只使用 XY 坐标建立搜索树
    tree = KDTree(candidates[:, :2])

    x_min, x_max = candidates[:, 0].min(), candidates[:, 0].max()
    y_min, y_max = candidates[:, 1].min(), candidates[:, 1].max()
    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2

    # 定义理想的人工布点位置 (标准 N-2-1 + 边缘加强)
    ideal_points = [
        [x_min, y_min], [x_max, y_min], [x_min, y_max], [x_max, y_max],  # 4角
        [x_mid, y_min], [x_mid, y_max],  # 长边中点
        [x_min, y_mid], [x_max, y_mid]  # 短边中点
    ]

    manual_fixtures = []
    # 找最近的有效点
    for pt in ideal_points:
        # tree 是 2D 的，pt 也是 2D 的，现在匹配了
        dist, idx = tree.query(pt)
        # idx 是索引，candidates[idx] 会返回完整的 3D 坐标，这是我们要的
        manual_fixtures.append(tuple(candidates[idx]))

    # 简单的去重并补齐到 TARGET_N
    unique_fixtures = list(set(manual_fixtures))
    # 如果不够8个，就补几个随机的有效点 (按索引均匀抽取)
    if len(unique_fixtures) < TARGET_N:
        needed = TARGET_N - len(unique_fixtures)
        indices = np.linspace(0, len(candidates) - 1, needed, dtype=int)
        for i in indices:
            cand = tuple(candidates[i])
            if cand not in unique_fixtures:
                unique_fixtures.append(cand)
            if len(unique_fixtures) == TARGET_N: break

    return unique_fixtures[:TARGET_N]


def main():
    print(f"🚀 启动终极评估 (Auto-Manual Baseline)...")

    if not os.path.exists(MODEL_PATH + ".zip"):
        print(f"❌ 模型未找到: {MODEL_PATH}")
        return

    # 1. 初始化环境
    env = SmartFixtureEnv3D(data_dir=DATA_DIR, target_n=TARGET_N, constraint_mode="n21")
    model = MaskablePPO.load(MODEL_PATH)

    # 2. AI 推理
    print(f"🤖 AI 正在推理...")
    obs, _ = env.reset()
    done = False
    while not done:
        action_masks = env.action_masks()
        action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
        obs, _, done, _, _ = env.step(action)

    ai_fixtures = np.array(env.fixtures)
    ai_solution = env.last_solution
    ai_uz = ai_solution["uz"]
    ai_max = ai_solution["max_abs_uz_mm"]

    # 3. 计算智能人工基准
    print(f"👷 生成几何人工基准...")
    manual_fixtures_list = get_smart_manual_baseline(env)

    if env.constraint_mode == "n21":
        man_solution = env._solve_mdsm(manual_fixtures_list, return_metrics=True)
        man_uz = man_solution["uz"]
        man_max = man_solution["max_abs_uz_mm"]
        env.last_locator_meta = man_solution["locator_meta"]
        env.last_locator2_points = man_solution["locator2_points"]
        env.last_locator1_point = man_solution["locator1_point"]
    else:
        man_uz = env._solve_mdsm(manual_fixtures_list)
        man_max = np.max(np.abs(man_uz)) * 1000.0  # mm

    # 打印结果
    print(f"\n📊 结果对比:")
    print(f"   🤖 AI Max Def:     {ai_max:.6e} mm")
    print(f"   👷 Manual Max Def: {man_max:.6e} mm")

    if man_max > 1e-9:
        improvement = (man_max - ai_max) / man_max * 100
        print(f"   🎉 AI 相对提升: {improvement:.2f}%")

    # ================= 4. 绘图 (带防除零保护) =================
    print("\n🎨 正在绘制...")
    fig = plt.figure(figsize=(18, 9))

    # 🟢 [防报错] 防止最大变形量为 0 导致除零错误
    vmax = max(ai_max, man_max)
    if vmax < 1e-9:
        vmax = 1e-9
        print("⚠️ 警告: 变形量极小，云图可能显示为单色")

    # 预计算插值网格
    x_raw, y_raw, z_raw = env.nodes[:, 0], env.nodes[:, 1], env.nodes[:, 2]
    range_x, range_y, range_z = np.ptp(x_raw), np.ptp(y_raw), np.ptp(z_raw)

    xi = np.linspace(x_raw.min(), x_raw.max(), 200)
    yi = np.linspace(y_raw.min(), y_raw.max(), 200)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((x_raw, y_raw), z_raw, (Xi, Yi), method='cubic')

    def plot_true_scale(ax, data, fixtures, title):
        val = np.abs(data) * 1000.0
        Ci = griddata((x_raw, y_raw), val, (Xi, Yi), method='cubic')

        # 归一化颜色
        norm_C = Ci / vmax

        # 绘制曲面
        surf = ax.plot_surface(Xi, Yi, Zi, facecolors=cm.jet(norm_C),
                               rstride=2, cstride=2, shade=False, linewidth=0, antialiased=True)

        # 绘制夹具
        fx = np.array(fixtures)
        if len(fx) > 0:
            for i, (fx_x, fx_y, fx_z) in enumerate(fx):
                is_init = (i < 3) and ("AI" in title)
                color = 'yellow' if is_init else 'lime'
                if "Manual" in title: color = 'white'
                label = f"P{i + 1}"

                # 箭头长度自适应
                arrow_len = max(range_z * 0.2, 0.05)
                z_top = fx_z + arrow_len

                ax.plot([fx_x, fx_x], [fx_y, fx_y], [fx_z, z_top], color='black', lw=1.5)
                ax.scatter(fx_x, fx_y, z_top, c=color, s=150, edgecolors='black', zorder=100)

        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_box_aspect((range_x, range_y, range_z))  # 保持物理比例
        ax.view_init(elev=30, azim=-60)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        return surf

    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = plot_true_scale(ax1, ai_uz, ai_fixtures, f"AI Solution\nMax: {ai_max:.4e} mm")

    ax2 = fig.add_subplot(122, projection='3d')
    plot_true_scale(ax2, man_uz, manual_fixtures_list, f"Smart Manual Baseline\nMax: {man_max:.4e} mm")

    m = cm.ScalarMappable(cmap=cm.jet)
    m.set_array(np.array([0, vmax]))
    cbar = fig.colorbar(m, ax=[ax1, ax2], orientation='horizontal', fraction=0.05, pad=0.05)
    cbar.set_label("Deformation (mm) [Red=Max, Blue=Min]", fontsize=12)

    plt.suptitle("Fixture Optimization Result", fontsize=18)
    plt.show()


if __name__ == "__main__":
    main()