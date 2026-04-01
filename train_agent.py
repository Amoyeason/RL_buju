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
from stable_baselines3.common.callbacks import BaseCallback
from smart_fixture_env import SmartFixtureEnv3D

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

LOG_DIR = "logs_3d/"
MODEL_DIR = "models_3d/"
BEST_MODEL_DIR = "models_3d/best/"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_DIR, exist_ok=True)

# 🟢 [建议] 增加到 50,000 步，让 AI 有足够时间探索大板布局
TOTAL_TIMESTEPS = 10000

EVAL_FREQ = 1000       # 每隔多少训练步做一次评估
N_EVAL_EPISODES = 5   # 每次评估跑几个 episode


class BestModelCallback(BaseCallback):
    """每隔 EVAL_FREQ 步用 deterministic 策略跑 N_EVAL_EPISODES 局，
    按平均 max_def_mm 保存历史最优模型。"""

    def __init__(self, eval_env, eval_freq=EVAL_FREQ, n_eval_episodes=N_EVAL_EPISODES,
                 save_path=BEST_MODEL_DIR, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.save_path = save_path
        self.best_mean_def = np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True
        defs = []
        for _ in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(
                    obs,
                    action_masks=self.eval_env.action_masks(),
                    deterministic=True,
                )
                obs, _, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
            defs.append(info.get("max_def_mm", np.inf))
        mean_def = float(np.mean(defs))
        if self.verbose:
            print(f"[EvalCallback] step={self.n_calls}  mean_max_def={mean_def:.4f} mm  "
                  f"(best={self.best_mean_def:.4f} mm)")
        if mean_def < self.best_mean_def:
            self.best_mean_def = mean_def
            path = os.path.join(self.save_path, "best_model")
            self.model.save(path)
            if self.verbose:
                print(f"   ✅ 新最优模型已保存: {path}.zip")
        return True


def main():
    print("🚀 启动 3D 曲面训练任务 (Physics V50 + BestModel Callback)...")

    # 1. 创建环境
    env = SmartFixtureEnv3D(data_dir="E:\\ansys_data_final", target_n=8, constraint_mode="n21")

    # 独立评估环境（与训练环境隔离，保证评估不污染训练状态）
    eval_env = SmartFixtureEnv3D(data_dir="E:\\ansys_data_final", target_n=8, constraint_mode="n21")

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
    best_cb = BestModelCallback(
        eval_env=eval_env,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        save_path=BEST_MODEL_DIR,
        verbose=1,
    )
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=best_cb)

    # 5. 保存
    save_path = os.path.join(MODEL_DIR, "final_model_3d")
    model.save(save_path)
    print(f"✅ 训练完成，final_model 已保存至: {save_path}.zip")
    print(f"✅ best_model 已保存至: {os.path.join(BEST_MODEL_DIR, 'best_model')}.zip")

if __name__ == "__main__":
    main()