# import os
# import numpy as np
# from sb3_contrib import MaskablePPO
# from smart_fixture_env import SmartFixtureEnv3D
#
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#
# LOG_DIR = "logs_3d/"
# MODEL_DIR = "models_3d/"
# os.makedirs(LOG_DIR, exist_ok=True)
# os.makedirs(MODEL_DIR, exist_ok=True)
#
# TOTAL_TIMESTEPS = 5000
#
# def main():
#     print("🚀 启动 3D 曲面训练任务...")
#
#     # 1. 创建环境
#     env = SmartFixtureEnv3D(data_dir="E:\\ansys_data_final", target_n=8)
#
#     # 🟢 [新增] 动作空间安全性检查
#     print(f"🧐 动作空间检查: 共有 {env.n_candidates} 个有效候选点")
#     if env.n_candidates < env.target_n:
#         print(f"❌ 致命错误: 候选点数量 ({env.n_candidates}) 少于目标夹具数 ({env.target_n})！")
#         print("   请在 smart_fixture_env.py 中放宽筛选阈值 (safety_check_radius 或 max_angle_gap)")
#         return
#
#     # 2. 载荷检查
#     max_force = np.max(np.abs(env.F_base))
#     print(f"🔥 环境载荷检查: Max F = {max_force:.4e}")
#     if max_force < 1e-6:
#         print("❌ 载荷异常（力太小），请检查 extract_data.py 的单位缩放！")
#         return
#
#     # 3. 初始化模型
#     model = MaskablePPO(
#         "MlpPolicy",
#         env,
#         verbose=1,
#         learning_rate=3e-4,
#         tensorboard_log=LOG_DIR,
#         device="cpu",
#         gamma=0.99,
#         ent_coef=0.02,
#         n_steps=64,      # 稍微调大一点，适应稍微复杂的3D环境
#         batch_size=32
#     )
#
#     # 4. 训练
#     print("🏋️ 开始训练...")
#     model.learn(total_timesteps=TOTAL_TIMESTEPS)
#
#     # 5. 保存
#     save_path = os.path.join(MODEL_DIR, "final_model_3d")
#     model.save(save_path)
#     print(f"✅ 训练完成，模型已保存至: {save_path}.zip")
#
# if __name__ == "__main__":
#     main()
import os
import numpy as np
from sb3_contrib import MaskablePPO
from smart_fixture_env import SmartFixtureEnv3D

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

LOG_DIR = "logs_3d/"
MODEL_DIR = "models_3d/"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# 🟢 [建议] 增加到 50,000 步，让 AI 有足够时间探索大板布局
TOTAL_TIMESTEPS = 50000

def main():
    print("🚀 启动 3D 曲面训练任务 (Physics V43 Ready)...")

    # 1. 创建环境
    env = SmartFixtureEnv3D(data_dir="E:\\ansys_data_final", target_n=8)

    print(f"🧐 动作空间检查: 共有 {env.n_candidates} 个有效候选点")
    if env.n_candidates < env.target_n:
        print(f"❌ 致命错误: 候选点数量 ({env.n_candidates}) 少于目标夹具数 ({env.target_n})！")
        return

    # 2. 载荷检查 (二次确认)
    max_force = np.max(np.abs(env.F_base))
    print(f"🔥 环境载荷检查: Max Nodal Force = {max_force:.4e} N")
    if max_force < 1e-4: # 稍微放宽一点，因为这是单节点力
        print("❌ 警告：载荷似乎过小，请再次确认 extract_data.py 是否运行成功。")
        # 这里不 return，允许强行尝试，但在日志里留痕

    # 3. 初始化模型
    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        tensorboard_log=LOG_DIR,
        device="cpu",
        gamma=0.99,
        ent_coef=0.01,   # 熵系数，保持探索
        n_steps=128,     # 增加步长，适应更长的探索
        batch_size=64
    )

    # 4. 训练
    print(f"🏋️ 开始训练 ({TOTAL_TIMESTEPS} steps)...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    # 5. 保存
    save_path = os.path.join(MODEL_DIR, "final_model_3d")
    model.save(save_path)
    print(f"✅ 训练完成，模型已保存至: {save_path}.zip")

if __name__ == "__main__":
    main()