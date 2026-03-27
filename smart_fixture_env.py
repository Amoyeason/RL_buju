import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.spatial import KDTree
import os
from constraint_n21 import select_21_locator_points, solve_mdsm_n21


class SmartFixtureEnv3D(gym.Env):
    def __init__(self, data_dir="E:\\ansys_data_final", target_n=8, constraint_mode="n21", locator_scheme="x2_y1", support_radius=0.05, locator_clearance=0.12, boundary_tol=0.032, penalty=1e15):
        super(SmartFixtureEnv3D, self).__init__()

        self.constraint_mode = constraint_mode
        self.locator_scheme = locator_scheme
        self.support_radius = support_radius
        self.locator_clearance = locator_clearance
        self.boundary_tol = boundary_tol
        self.penalty = penalty

        # ================= 1. 加载 3D 物理数据 =================
        if self.constraint_mode == "n21":
            print(f"🤖 初始化 3D 物理引擎 (N-2-1 Constraint Mode, scheme={self.locator_scheme})...")
        else:
            print("🤖 初始化 3D 物理引擎 (Full-Constraint Rigid Mode)...")
        try:
            # 加载刚度矩阵 K
            K_raw = sparse.load_npz(os.path.join(data_dir, "K_pure.npz"))
            self.K_base = K_raw + K_raw.T - sparse.diags(K_raw.diagonal())
            self.K_base = self.K_base.tocsr()

            # 加载载荷 F 和 节点信息
            data = np.load(os.path.join(data_dir, "digital_twin_data.npz"))
            self.F_base = data['F']
            self.nodes = data['nodes']  # (N, 3)
            self.n_nodes = len(self.nodes)

            self.ux_map = data['ux_map']
            self.uy_map = data['uy_map']
            self.uz_map = data['uz_map']

            self.fem_tree_3d = KDTree(self.nodes)
            self.fem_tree_2d = KDTree(self.nodes[:, :2])

        except Exception as e:
            raise RuntimeError(f"❌ 数据加载失败: {e}")

        # ================= 2. 几何与布局参数 =================
        self.fixture_radius = 0.05  # 吸盘半径

        # 候选点生成参数
        self.layout_step = 0.25
        self.layout_margin = 0.08

        # 筛选参数
        self.safety_check_radius = 0.07
        self.max_angle_gap = 150
        self.max_centroid_offset = 0.015

        self.candidates = self._generate_surface_candidates()
        self.n_candidates = len(self.candidates)

        print(f"🔧 3D 动作空间生成完毕: {self.n_candidates} 个有效候选点 (V12筛选已生效)")

        # ================= 3. 空间定义 =================
        if self.n_candidates < target_n:
            print(f"⚠️ 警告: 候选点数量 ({self.n_candidates}) 少于目标夹具数 ({target_n})。")

        self.action_space = spaces.Discrete(self.n_candidates)

        # 🟢 [优化] 观测空间范围虽然是inf，但我们会输入 scaling 后的值
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.n_nodes + self.n_candidates,),
            dtype=np.float32
        )

        self.target_n = target_n
        self.fixtures = []
        self.occupied_indices = set()
        self.last_max_def = 0.0

    def _check_candidate_safety(self, center):
        indices = self.fem_tree_3d.query_ball_point(center, self.safety_check_radius)
        if len(indices) < 4: return False
        neighbors = self.nodes[indices]

        centroid = np.mean(neighbors, axis=0)
        if np.linalg.norm(centroid[:2] - center[:2]) > self.max_centroid_offset: return False

        dx = neighbors[:, 0] - center[0]
        dy = neighbors[:, 1] - center[1]
        angles = np.arctan2(dy, dx)
        angles.sort()
        diffs = np.diff(angles)
        if len(diffs) == 0: return False
        last_gap = (angles[0] + 2 * np.pi) - angles[-1]
        max_gap = max(np.max(diffs), last_gap)
        return np.rad2deg(max_gap) <= self.max_angle_gap

    def _generate_surface_candidates(self):
        x_min, x_max = self.nodes[:, 0].min(), self.nodes[:, 0].max()
        y_min, y_max = self.nodes[:, 1].min(), self.nodes[:, 1].max()

        xs = np.arange(x_min + self.layout_margin, x_max - self.layout_margin, self.layout_step)
        ys = np.arange(y_min + self.layout_margin, y_max - self.layout_margin, self.layout_step)
        grid_x, grid_y = np.meshgrid(xs, ys)
        flat_grid = np.column_stack([grid_x.ravel(), grid_y.ravel()])

        valid_candidates = []
        for gx, gy in flat_grid:
            dist, node_idx = self.fem_tree_2d.query([gx, gy])
            if dist > 0.10: continue
            candidate_node = self.nodes[node_idx]
            if self._check_candidate_safety(candidate_node):
                valid_candidates.append(candidate_node)

        valid_candidates = np.array(valid_candidates)
        if len(valid_candidates) > 0:
            valid_candidates = np.unique(valid_candidates, axis=0)
        return valid_candidates

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.fixtures = []
        self.occupied_indices = set()

        init_indices = self._find_max_area_triangle()
        for idx in init_indices:
            self.fixtures.append(tuple(self.candidates[idx]))
            self.occupied_indices.add(idx)

        # 统一使用带有 return_metrics=True 的调用，收集包含所有全量数据的字典
        self.last_solution = self._solve_mdsm(self.fixtures, return_metrics=True)
        self.last_uz = self.last_solution["uz"]

        # 为了兼容现有训练机制，仍然把最大 Z 向变形作为 self.last_max_def
        self.last_max_def = self.last_solution["max_abs_uz_mm"]

        # 保存这步的定位点元数据，方便渲染或后续分析
        if self.constraint_mode == "n21":
            self.last_locator_meta = self.last_solution.get("locator_meta")
            self.last_locator2_points = self.last_solution.get("locator2_points")
            self.last_locator1_point = self.last_solution.get("locator1_point")

        return self._get_obs(self.last_uz), {}

    def _find_max_area_triangle(self):
        if self.n_candidates < 3: return list(range(self.n_candidates))
        sum_xy = self.candidates[:, 0] + self.candidates[:, 1]
        idx1 = np.argmin(sum_xy)
        p1 = self.candidates[idx1]
        dists = np.linalg.norm(self.candidates - p1, axis=1)
        idx2 = np.argmax(dists)
        p2 = self.candidates[idx2]
        vec = p2 - p1
        cross = np.abs(np.cross(vec[:2], (self.candidates - p1)[:, :2]))
        idx3 = np.argmax(cross)
        if cross[idx3] < 1e-4: idx3 = (idx1 + 1) % self.n_candidates  # 防共线兜底
        return [idx1, idx2, idx3]


    # def step(self, action):
    #     idx = int(action)
    #
    #     # 🔴 [优化] 重复选择的惩罚加倍，迫使它去探索新位置
    #     if idx in self.occupied_indices:
    #         return self._get_obs(self.last_uz), -20.0, False, False, {"max_def_mm": self.last_max_def}
    #
    #     self.fixtures.append(tuple(self.candidates[idx]))
    #     self.occupied_indices.add(idx)
    #
    #     uz_current = self._solve_mdsm(self.fixtures)
    #     current_max_def = np.max(np.abs(uz_current)) * 1000.0  # mm
    #
    #     # ================= 🟢 V46 混合动力奖励函数 =================
    #
    #     # 1. 对数奖励 (Log Reward) - 负责粗略的大幅下降
    #     # 增加 offset 防止 log(0) 爆炸，这里用 0.1 作为基准底噪
    #     log_prev = np.log(self.last_max_def + 0.1)
    #     log_curr = np.log(current_max_def + 0.1)
    #     reward_log = (log_prev - log_curr) * 20.0  # 放大系数，鼓励前期快速收敛
    #
    #     # 2. 线性精密奖励 (Linear Precision Reward) - 负责后期的毫米级微调
    #     # 只有当变形量小于 2.0mm (进入精细区) 时才生效
    #     reward_linear = 0.0
    #     if current_max_def < 2.0:
    #         diff = self.last_max_def - current_max_def
    #         if diff > 0:
    #             # 放大系数 50: 每降低 0.1mm 奖励 5分
    #             # 即使是 0.02mm 的微小下降，也能拿 1.0分，足以驱动 AI 优化
    #             reward_linear = diff * 100
    #
    #     step_reward = reward_log + reward_linear
    #
    #     # 3. 进步小红花
    #     if current_max_def < self.last_max_def:
    #         step_reward += 1.0
    #
    #     self.last_uz = uz_current
    #     self.last_max_def = current_max_def
    #     terminated = len(self.fixtures) >= self.target_n
    #
    #     # 4. 连续型终局大奖 (Continuous Terminal Reward)
    #     if terminated:
    #         # 目标基准线: 0.8mm
    #         # 使用反比例函数，越接近 0 分数越高，没有上限
    #         # max(x, 0.4) 是为了防止除以 0 或者分数无限大，0.4mm 是物理极限底线
    #         final_score = 20.0 / max(current_max_def, 0.4)
    #         # 例如:
    #         # 跑出 1.0mm -> 得 20分
    #         # 跑出 0.8mm -> 得 25分
    #         # 跑出 0.6mm -> 得 33分
    #         step_reward += final_score
    #
    #         # 额外的“里程碑”奖励 (Dopamine hits)
    #         if current_max_def < 0.8: step_reward += 10.0
    #         if current_max_def < 0.6: step_reward += 20.0
    #
    #     # 截断防止梯度爆炸 (范围放宽到 -50 ~ 100)
    #     step_reward = np.clip(step_reward, -50.0, 100.0)
    #
    #     return self._get_obs(uz_current), step_reward, terminated, False, {"max_def_mm": current_max_def}

    def step(self, action):
        idx = int(action)

        # 1. 基础惩罚
        if idx in self.occupied_indices:
            return self._get_obs(self.last_uz), -10.0, False, False, {"max_def_mm": self.last_max_def}

        # 记录当前决策的位置坐标
        current_action_pos = self.candidates[idx]

        # -----------------------------------------------------------
        # 🔥 新增：在物理计算前，先判断 AI 是否“瞄准”了最大变形区域
        # -----------------------------------------------------------
        # 找到上一帧中，变形最大的那个物理节点的坐标
        max_node_idx = np.argmax(np.abs(self.last_uz))  # 找最大变形点的索引
        max_node_pos = self.nodes[max_node_idx]  # 获取该点的 (x,y,z)

        # 计算 AI 下子位置与最大变形点的 2D 平面距离
        dist_to_hotspot = np.linalg.norm(current_action_pos[:2] - max_node_pos[:2])

        # 🎯 热点引导奖励 (Sniper Bonus)
        # 距离越近分越高。如果在 0.2m (20cm) 范围内，给高分
        sniper_reward = 0.0
        if dist_to_hotspot < 0.2:
            sniper_reward = 5.0 * (1.0 - dist_to_hotspot / 0.2)  # 最高 5 分

        # -----------------------------------------------------------
        # 执行物理仿真
        # -----------------------------------------------------------
        self.fixtures.append(tuple(self.candidates[idx]))
        self.occupied_indices.add(idx)

        self.last_solution = self._solve_mdsm(self.fixtures, return_metrics=True)
        uz_current = self.last_solution["uz"]
        current_max_def = self.last_solution["max_abs_uz_mm"]  # mm

        if self.constraint_mode == "n21":
            self.last_locator_meta = self.last_solution.get("locator_meta")
            self.last_locator2_points = self.last_solution.get("locator2_points")
            self.last_locator1_point = self.last_solution.get("locator1_point")

        # ================= 🟢 V48 百分比强力驱动版 =================
        step_reward = 0.0

        # 加入瞄准奖励
        step_reward += sniper_reward

        diff = self.last_max_def - current_max_def

        # 🟢 核心改变：使用“相对提升率”而不是“绝对差值”
        # 只要你能让当前最大变形下降 10%，无论基数是 2mm 还是 0.2mm，奖励都一样！
        # 这彻底解决了后期“没动力”的问题

        improvement_ratio = 0.0
        if self.last_max_def > 1e-5:
            improvement_ratio = diff / self.last_max_def

        if abs(diff) > 0.002:  # Deadband 保持，防止噪声
            if diff > 0:
                # 进步奖励
                # 例：下降 10% (0.1) -> 奖励 10 分
                # 例：下降 1% (0.01) -> 奖励 1 分
                # 这种尺度非常适合 PPO，且 Step 8 和 Step 4 权重一致
                step_reward += improvement_ratio * 100.0

                # 额外叠加线性项，保证大尺度下降依然爽
                step_reward += diff * 50.0
            else:
                # 退步惩罚 (保持较轻，鼓励试错)
                step_reward += improvement_ratio * 50.0  # 注意 ratio 是负的

        # 进步微奖 (削弱，防止刷分)
        if current_max_def < self.last_max_def:
            step_reward += 0.5

        self.last_uz = uz_current
        self.last_max_def = current_max_def
        terminated = len(self.fixtures) >= self.target_n

        # D. 终局奖励
        if terminated:
            # 保持 V47 的温和设计，不喧宾夺主
            final_score = 25.0 / max(current_max_def, 0.4)
            step_reward += final_score

            if current_max_def < 0.8: step_reward += 10.0
            if current_max_def < 0.6: step_reward += 20.0

        step_reward = np.clip(step_reward, -50.0, 100.0)

        return self._get_obs(uz_current), step_reward, terminated, False, {"max_def_mm": current_max_def}
    def action_masks(self):
        mask = np.ones(self.n_candidates, dtype=bool)
        mask[list(self.occupied_indices)] = False
        return mask

    def _get_obs(self, uz_data):
        occupancy = np.zeros(self.n_candidates)
        occupancy[list(self.occupied_indices)] = 1.0

        # 🟢 [关键] 输入归一化 (Scaling)
        # 将变形量 (米) 乘以 1000 变为 (毫米)，让数值在 0.1~10 之间
        # 神经网络对这个范围的数值最敏感
        uz_scaled = uz_data * 1000.0

        return np.concatenate([uz_scaled, occupancy]).astype(np.float32)

    def _solve_mdsm(self, active_fixtures, return_metrics=False):
        if self.constraint_mode == "n21":
            if not active_fixtures:
                # 没有任何支撑点时的兜底
                metrics = {
                    "uz": np.zeros(self.n_nodes),
                    "max_abs_uz_mm": 0.0,
                    "locator_meta": None,
                    "locator2_points": None,
                    "locator1_point": None
                }
                return metrics if return_metrics else metrics["uz"]

            pts2, pts1, meta = select_21_locator_points(
                env=self,
                support_points=active_fixtures,
                scheme=self.locator_scheme,
                clearance=self.locator_clearance,
                boundary_tol=self.boundary_tol
            )
            metrics = solve_mdsm_n21(
                env=self,
                support_points=active_fixtures,
                locator2_points=pts2,
                locator1_point=pts1,
                locator_meta=meta,
                penalty=self.penalty,
                support_radius=self.support_radius,
                locator_clearance=self.locator_clearance
            )
            return metrics if return_metrics else metrics["uz"]
        else:
            # 兼容旧版的 "full" 刚性约束
            K_mod = self.K_base.copy()
            F_mod = self.F_base.copy()
            penalty = 1e15

            if active_fixtures:
                indices_list = self.fem_tree_3d.query_ball_point(active_fixtures, self.fixture_radius)
                for indices in indices_list:
                    for idx in indices:
                        for eq_map in [self.uz_map, self.ux_map, self.uy_map]:
                            eq = eq_map[idx]
                            if eq != -1:
                                K_mod[eq, eq] += penalty
                                F_mod[eq] = 0.0
            try:
                U_vec = spsolve(K_mod, F_mod)
            except:
                U_vec = np.zeros(self.n_nodes)

            uz_result = np.zeros(self.n_nodes)
            valid_mask = self.uz_map != -1
            uz_result[valid_mask] = U_vec[self.uz_map[valid_mask]]

            if return_metrics:
                # 为了格式统一，造一个 metrics 字典
                return {
                    "uz": uz_result,
                    "max_abs_uz_mm": float(np.max(np.abs(uz_result)) * 1000.0)
                }
            return uz_result