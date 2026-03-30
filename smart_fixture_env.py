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

        self.target_n = target_n
        self.max_episode_steps = target_n + 2  # 防止无限步卡死
        self.step_count = 0
        self.fixtures = []
        self.occupied_indices = set()
        self.last_max_def = 0.0
        self.last_solution = None
        self.last_uz = None
        self.last_locator_meta = None
        self.last_locator2_points = None
        self.last_locator1_point = None

        # 扩展观测空间：uz(n_nodes) + occupancy(n_cands) + locator2_xy(4) + locator1_xy(2) + hotspot_xy(2) + max_def(1) + step_ratio(1)
        _extra_dim = 4 + 2 + 2 + 1 + 1  # = 10
        _low  = np.concatenate([
            np.full(self.n_nodes,      -1.0, dtype=np.float32),   # uz after tanh ∈ (-1, 1)
            np.zeros(self.n_candidates,       dtype=np.float32),  # occupancy ∈ {0,1}
            np.zeros(_extra_dim,              dtype=np.float32),  # 辅助信息全部归一化到 [0,1]
        ])
        _high = np.concatenate([
            np.full(self.n_nodes,        1.0, dtype=np.float32),
            np.ones(self.n_candidates,        dtype=np.float32),
            np.ones(_extra_dim,               dtype=np.float32),
        ])
        self.observation_space = spaces.Box(low=_low, high=_high, dtype=np.float32)

        # 缓存几何归一化常量（热路径优化，避免每次 _get_obs 重复计算）
        self.x_min = float(self.nodes[:, 0].min())
        self.x_max = float(self.nodes[:, 0].max())
        self.y_min = float(self.nodes[:, 1].min())
        self.y_max = float(self.nodes[:, 1].max())
        self.x_span = max(self.x_max - self.x_min, 1e-6)
        self.y_span = max(self.y_max - self.y_min, 1e-6)

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
        self.step_count = 0

        init_indices = self._find_max_area_triangle()
        for idx in init_indices:
            self.fixtures.append(tuple(self.candidates[idx]))
            self.occupied_indices.add(idx)

        try:
            self.last_solution = self._solve_mdsm(self.fixtures, return_metrics=True)
        except Exception as e:
            raise RuntimeError(
                f"[Env.reset] 初始三角布局求解失败，请检查物理数据或候选点: {e}"
            ) from e

        self.last_uz = self.last_solution["uz"]
        self.last_max_def = self.last_solution["max_abs_uz_mm"]

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
        self.step_count += 1

        if idx in self.occupied_indices:
            truncated = self.step_count >= self.max_episode_steps
            return self._get_obs(self.last_uz), -10.0, False, truncated, {"max_def_mm": self.last_max_def}

        current_action_pos = self.candidates[idx]

        # Sniper Bonus：判断是否瞄准热点
        max_node_idx = np.argmax(np.abs(self.last_uz))
        max_node_pos = self.nodes[max_node_idx]
        dist_to_hotspot = np.linalg.norm(current_action_pos[:2] - max_node_pos[:2])

        # 执行物理仿真（含异常保护 + rollback）
        self.fixtures.append(tuple(self.candidates[idx]))
        self.occupied_indices.add(idx)

        solve_failed = False
        try:
            self.last_solution = self._solve_mdsm(self.fixtures, return_metrics=True)
        except RuntimeError as e:
            print(f"[Env] 物理求解失败，执行 rollback: {e}")
            self.fixtures.pop()
            self.occupied_indices.discard(idx)
            solve_failed = True

        if solve_failed:
            return (
                self._get_obs(self.last_uz),
                -10.0,
                False,
                True,
                {"max_def_mm": self.last_max_def, "solve_failed": True},
            )

        uz_current = self.last_solution["uz"]
        current_max_def = self.last_solution["max_abs_uz_mm"]

        if self.constraint_mode == "n21":
            self.last_locator_meta = self.last_solution.get("locator_meta")
            self.last_locator2_points = self.last_solution.get("locator2_points")
            self.last_locator1_point = self.last_solution.get("locator1_point")

        terminated = len(self.fixtures) >= self.target_n
        step_reward, reward_info = self._compute_reward(
            prev_max_def=self.last_max_def,
            current_max_def=current_max_def,
            dist_to_hotspot=dist_to_hotspot,
            terminated=terminated,
        )
        self.last_uz = uz_current
        self.last_max_def = current_max_def

        truncated = (not terminated) and (self.step_count >= self.max_episode_steps)
        step_reward = np.clip(step_reward, -10.0, 20.0)

        info = {
            "max_def_mm": float(current_max_def),
            "dist_to_hotspot": float(dist_to_hotspot),
            "step_count": int(self.step_count),
            "n_fixtures": int(len(self.fixtures)),
            "solve_failed": False,
        }
        info.update(reward_info)
        return self._get_obs(uz_current), step_reward, terminated, truncated, info

    def _compute_reward(self, prev_max_def, current_max_def, dist_to_hotspot=0.0, terminated=False):
        """三段式奖励函数（V50 对数改善版）。

        返回:
            total_reward (float): 本步总奖励（clip 前）
            info (dict): 各分项，供 step() 透传到 info 字典供调试
        """
        alpha = 8.0  # 对数改善项系数

        # ── A. 主 dense reward：对数改善 ──────────────────────────────
        # log((d_{t-1} + eps) / (d_t + eps)) > 0 表示改善，< 0 表示退步
        # 对数形式：量纲小、不饱和、前后期权重自然均衡
        eps = 0.1  # mm，防止 log(0)
        reward_dense = alpha * np.log(
            (prev_max_def + eps) / (current_max_def + eps)
        )

        # ── B. 热点引导项 ─────────────────────────────────────────────
        # 与 dense reward 同量级（最高 ~2.5 分）
        reward_hotspot = 0.0
        if dist_to_hotspot < 0.25:
            reward_hotspot = 2.5 * (1.0 - dist_to_hotspot / 0.25)

        # ── C. 终局奖励（显著降权，避免喧宾夺主）────────────────────
        reward_terminal = 0.0
        if terminated:
            reward_terminal += 8.0 / max(current_max_def, 0.8)
            if current_max_def < 1.0:
                reward_terminal += 2.0
            if current_max_def < 0.8:
                reward_terminal += 3.0
            if current_max_def < 0.6:
                reward_terminal += 4.0

        total = reward_dense + reward_hotspot + reward_terminal
        info = {
            "reward_dense": float(reward_dense),
            "reward_hotspot": float(reward_hotspot),
            "reward_terminal": float(reward_terminal),
        }
        return total, info

    def action_masks(self):
        mask = np.ones(self.n_candidates, dtype=bool)
        mask[list(self.occupied_indices)] = False
        return mask

    def _get_obs(self, uz_data):
        occupancy = np.zeros(self.n_candidates)
        occupancy[list(self.occupied_indices)] = 1.0
        # tanh 平滑压缩：输出 ∈ (-1, 1)，与 observation_space 一致
        uz_mm = uz_data * 1000.0
        uz_scaled = np.tanh(uz_mm / 20.0).astype(np.float32)

        # ---- 额外 10 维辅助信息 ----
        # 使用 __init__ 中缓存的几何常量，避免热路径重复计算
        def norm_x(v):
            return (float(v) - self.x_min) / self.x_span  # → [0, 1]

        def norm_y(v):
            return (float(v) - self.y_min) / self.y_span

        # locator2_xy: 2 个定位点的归一化 (x,y)，共 4 维
        if self.last_locator2_points is not None and len(self.last_locator2_points) >= 2:
            p2a = self.last_locator2_points[0]
            p2b = self.last_locator2_points[1]
            loc2_vec = np.array([
                norm_x(p2a[0]), norm_y(p2a[1]),
                norm_x(p2b[0]), norm_y(p2b[1]),
            ], dtype=np.float32)
        else:
            loc2_vec = np.zeros(4, dtype=np.float32)

        # locator1_xy: 1 个定位点的归一化 (x,y)，共 2 维
        if self.last_locator1_point is not None:
            p1 = self.last_locator1_point
            loc1_vec = np.array([norm_x(p1[0]), norm_y(p1[1])], dtype=np.float32)
        else:
            loc1_vec = np.zeros(2, dtype=np.float32)

        # hotspot_xy: 最大变形点归一化坐标，共 2 维
        max_node_idx = int(np.argmax(np.abs(uz_data)))
        hp = self.nodes[max_node_idx]
        hotspot_vec = np.array([norm_x(hp[0]), norm_y(hp[1])], dtype=np.float32)

        # max_def: 当前最大变形 (mm) 归一化到 [0, 1]，clip 上限 20 mm
        max_def_norm = np.array(
            [float(np.tanh(float(np.max(np.abs(uz_data))) * 1000.0 / 20.0))],
            dtype=np.float32,
        )

        # step_ratio: 当前步占最大步数的比例
        step_ratio = np.array(
            [float(len(self.fixtures)) / float(self.target_n)],
            dtype=np.float32,
        )

        return np.concatenate(
            [uz_scaled, occupancy, loc2_vec, loc1_vec, hotspot_vec, max_def_norm, step_ratio]
        ).astype(np.float32)

    def _solve_mdsm(self, active_fixtures, return_metrics=False):
        if self.constraint_mode == "n21":
            if not active_fixtures:
                metrics = {
                    "uz": np.zeros(self.n_nodes),
                    "max_abs_uz_mm": 0.0,
                    "locator_meta": None,
                    "locator2_points": None,
                    "locator1_point": None
                }
                return metrics if return_metrics else metrics["uz"]

            try:
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
            except RuntimeError:
                raise
            except Exception as e:
                raise RuntimeError(f"[n21] 定位点选择或物理求解失败: {e}") from e
            if not isinstance(metrics, dict):
                raise RuntimeError(
                    f"solve_mdsm_n21 返回了非 dict 类型: {type(metrics)}，无法解析求解结果"
                )
            return metrics if return_metrics else metrics["uz"]
        else:
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
            except Exception as e:
                raise RuntimeError(f"full 模式 spsolve 失败: {e}") from e

            uz_result = np.zeros(self.n_nodes)
            valid_mask = self.uz_map != -1
            uz_result[valid_mask] = U_vec[self.uz_map[valid_mask]]

            if return_metrics:
                return {
                    "uz": uz_result,
                    "max_abs_uz_mm": float(np.max(np.abs(uz_result)) * 1000.0)
                }
            return uz_result