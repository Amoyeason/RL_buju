# import os

# # 解决 OpenMP / MKL 冲突
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# import glob
# import shutil
# import time
# import threading
# import numpy as np
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt

# from matplotlib import cm
# from scipy import sparse
# from scipy.sparse.linalg import spsolve
# from scipy.spatial import KDTree
# from scipy.interpolate import griddata

# from ansys.mapdl.core import launch_mapdl
# from sb3_contrib import MaskablePPO
# from smart_fixture_env import SmartFixtureEnv3D


# # ==============================
# # 配置区域
# # ==============================
# DATA_DIR = r"E:\ansys_data_final"
# CAD_PATH = r"E:\ZJU\Learning baogao\0128.IGS"
# MODEL_PATH = r"models_3d/final_model_3d"
# TARGET_N = 8

# # 必须与 extract_data.py 保持一致
# THICKNESS = 0.005
# MESH_SIZE = 0.04
# DENSITY = 1600.0

# WORK_DIR = "ansys_temp_verify_support_plus_21"
# JOB_NAME = "verify_support21"

# # 逻辑参数
# SUPPORT_RADIUS = 0.05
# LOCATOR_CLEARANCE = 0.12
# BOUNDARY_TOL = 0.032
# PENALTY = 1e15

# # 输出文件
# LAYOUT_PNG = "verify_support21_layout.png"
# ANSYS_CONSTRAINT_PNG = "verify_support21_constraints.png"
# ANSYS_RESULT_UZ_PNG = "verify_support21_result_uz.png"
# PYTHON_MDSM_UZ_PNG = "verify_support21_python_mdsm_3d.png"
# COMPARE_PNG = "verify_support21_mdsm_vs_ansys.png"


# # ==============================
# # 基础工具
# # ==============================
# def tail_ansys_log(stop_event, work_dir, job_name):
#     log_path = os.path.join(work_dir, f"{job_name}.out")
#     while not os.path.exists(log_path):
#         if stop_event.is_set():
#             return
#         time.sleep(0.1)

#     with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
#         f.seek(0, 2)
#         while not stop_event.is_set():
#             line = f.readline()
#             if line:
#                 low = line.lower()
#                 if ("error" in low) or ("warning" in low):
#                     print(f"   [ANSYS] {line.strip()}")
#             else:
#                 time.sleep(0.1)


# def ensure_clean_workdir(work_dir):
#     if os.path.exists(work_dir):
#         shutil.rmtree(work_dir)
#     os.makedirs(work_dir, exist_ok=True)


# def format_points(name, pts):
#     arr = np.asarray(pts, dtype=float)
#     print(f"\n{name}:")
#     for i, p in enumerate(arr, start=1):
#         print(f"  {i:02d}: ({p[0]:.6f}, {p[1]:.6f}, {p[2]:.6f})")


# def list_job_pngs():
#     pattern = os.path.join(WORK_DIR, f"{JOB_NAME}*.png")
#     return set(glob.glob(pattern))


# def copy_newest_job_png(before_set, dst_path):
#     after_set = list_job_pngs()
#     new_files = sorted(list(after_set - before_set), key=os.path.getmtime)
#     if not new_files:
#         # 兜底：取最新的
#         all_files = sorted(list(after_set), key=os.path.getmtime)
#         if not all_files:
#             return False
#         src = all_files[-1]
#     else:
#         src = new_files[-1]
#     shutil.copy(src, dst_path)
#     return True


# def fill_nan_grid(arr, Xi, Yi, x_raw, y_raw, raw_val):
#     """
#     对 griddata 产生的 NaN 进行 nearest 兜底填补
#     """
#     if np.isnan(arr).any():
#         arr_nearest = griddata((x_raw, y_raw), raw_val, (Xi, Yi), method="nearest")
#         mask = np.isnan(arr)
#         arr[mask] = arr_nearest[mask]
#     return arr


# # ==============================
# # Step 1: AI 推理，得到 8 个支撑点
# # ==============================
# def infer_ai_support_layout():
#     if not os.path.exists(MODEL_PATH + ".zip"):
#         raise FileNotFoundError(f"找不到模型文件: {MODEL_PATH}.zip")

#     print("🤖 [Step 1] AI 正在推理 8 个 N 类支撑点 ...")
#     env = SmartFixtureEnv3D(data_dir=DATA_DIR, target_n=TARGET_N)
#     model = MaskablePPO.load(MODEL_PATH)

#     obs, _ = env.reset()
#     done = False
#     while not done:
#         action, _ = model.predict(
#             obs,
#             action_masks=env.action_masks(),
#             deterministic=True
#         )
#         obs, _, done, _, _ = env.step(action)

#     support_points = np.asarray(env.fixtures, dtype=float)
#     full_constraint_python_mm = float(env.last_max_def)

#     print(f"   ✅ AI 推理完成，共 {len(support_points)} 个支撑点")
#     print(f"   📘 现有环境（全向刚约束）Python 结果: {full_constraint_python_mm:.6f} mm")
#     return env, support_points, full_constraint_python_mm


# # ==============================
# # Step 2: 单独构造 2-1 定位点
# # ==============================
# def select_21_locator_points(env, support_points):
#     print("\n📐 [Step 2] 自动构造独立的 2-1 定位点 ...")

#     nodes = np.asarray(env.nodes, dtype=float)
#     support_xy = np.asarray(support_points[:, :2], dtype=float)
#     support_tree = KDTree(support_xy)

#     x = nodes[:, 0]
#     y = nodes[:, 1]

#     x_min, x_max = x.min(), x.max()
#     y_min, y_max = y.min(), y.max()
#     x_span = x_max - x_min
#     y_span = y_max - y_min

#     if x_span >= y_span:
#         two_edge_primary = "xmin"
#         two_edge_fallback = "xmax"
#         one_edge_primary = "ymin"
#         one_edge_fallback = "ymax"
#         two_target_fracs = [0.25, 0.75]
#         one_target_frac = 0.50
#     else:
#         two_edge_primary = "ymin"
#         two_edge_fallback = "ymax"
#         one_edge_primary = "xmin"
#         one_edge_fallback = "xmax"
#         two_target_fracs = [0.25, 0.75]
#         one_target_frac = 0.50

#     used_indices = set()

#     def edge_mask(edge_name, tol):
#         if edge_name == "ymin":
#             return np.abs(nodes[:, 1] - y_min) <= tol
#         if edge_name == "ymax":
#             return np.abs(nodes[:, 1] - y_max) <= tol
#         if edge_name == "xmin":
#             return np.abs(nodes[:, 0] - x_min) <= tol
#         if edge_name == "xmax":
#             return np.abs(nodes[:, 0] - x_max) <= tol
#         raise ValueError(f"未知边界名: {edge_name}")

#     def target_coordinate(edge_name, frac):
#         if edge_name in ("ymin", "ymax"):
#             return x_min + frac * x_span
#         return y_min + frac * y_span

#     def edge_parameter(edge_name):
#         if edge_name in ("ymin", "ymax"):
#             return nodes[:, 0]
#         return nodes[:, 1]

#     def pick_node_on_edge(edge_name, frac, tol, clearance):
#         mask = edge_mask(edge_name, tol)
#         if not np.any(mask):
#             return None

#         d_to_support, _ = support_tree.query(nodes[:, :2])
#         mask = mask & (d_to_support >= clearance)

#         if used_indices:
#             used_mask = np.ones(len(nodes), dtype=bool)
#             used_mask[list(used_indices)] = False
#             mask = mask & used_mask

#         candidate_ids = np.where(mask)[0]
#         if len(candidate_ids) == 0:
#             return None

#         param = edge_parameter(edge_name)[candidate_ids]
#         target = target_coordinate(edge_name, frac)
#         best_local = np.argmin(np.abs(param - target))
#         return int(candidate_ids[best_local])

#     def robust_pick(edge_primary, edge_fallback, frac):
#         for tol in [BOUNDARY_TOL, BOUNDARY_TOL * 1.5, BOUNDARY_TOL * 2.0]:
#             for clr in [LOCATOR_CLEARANCE, LOCATOR_CLEARANCE * 0.7, LOCATOR_CLEARANCE * 0.4]:
#                 idx = pick_node_on_edge(edge_primary, frac, tol, clr)
#                 if idx is not None:
#                     return idx
#                 idx = pick_node_on_edge(edge_fallback, frac, tol, clr)
#                 if idx is not None:
#                     return idx
#         raise RuntimeError(
#             f"无法为 edge={edge_primary}/{edge_fallback}, frac={frac:.2f} 选出合适定位点"
#         )

#     locator2_idx = []
#     for frac in two_target_fracs:
#         idx = robust_pick(two_edge_primary, two_edge_fallback, frac)
#         locator2_idx.append(idx)
#         used_indices.add(idx)

#     locator1_idx = robust_pick(one_edge_primary, one_edge_fallback, one_target_frac)
#     used_indices.add(locator1_idx)

#     locator2_points = nodes[locator2_idx]
#     locator1_point = nodes[locator1_idx]

#     meta = {
#         "two_edge_primary": two_edge_primary,
#         "two_edge_fallback": two_edge_fallback,
#         "one_edge_primary": one_edge_primary,
#         "one_edge_fallback": one_edge_fallback,
#         "locator2_indices": locator2_idx,
#         "locator1_index": locator1_idx,
#     }

#     print(f"   ✅ 2点定位边: {two_edge_primary} (fallback={two_edge_fallback})")
#     print(f"   ✅ 1点定位边: {one_edge_primary} (fallback={one_edge_fallback})")
#     return locator2_points, locator1_point, meta


# # ==============================
# # Step 3: Python 端独立验证
# # 支撑点: UZ = 0（在支撑半径邻域内施加）
# # 2定位点: UY = 0
# # 1定位点: UX = 0
# # ==============================
# def solve_python_support_plus_21(env, support_points, locator2_points, locator1_point, penalty=PENALTY):
#     print("\n🧮 [Step 3] Python 端进行 Support + 2-1 独立重分析 ...")

#     ndof = env.K_base.shape[0]
#     penalty_vec = np.zeros(ndof, dtype=np.float64)

#     support_indices_list = env.fem_tree_3d.query_ball_point(np.asarray(support_points), SUPPORT_RADIUS)
#     support_node_ids = set()

#     for idx_list in support_indices_list:
#         for node_idx in idx_list:
#             support_node_ids.add(int(node_idx))
#             eq = int(env.uz_map[node_idx])
#             if eq >= 0:
#                 penalty_vec[eq] = penalty

#     locator2_node_ids = []
#     for pt in np.asarray(locator2_points):
#         _, node_idx = env.fem_tree_3d.query(pt)
#         locator2_node_ids.append(int(node_idx))
#         eq = int(env.uy_map[node_idx])
#         if eq >= 0:
#             penalty_vec[eq] = penalty

#     _, locator1_node_idx = env.fem_tree_3d.query(np.asarray(locator1_point))
#     locator1_node_idx = int(locator1_node_idx)
#     eq = int(env.ux_map[locator1_node_idx])
#     if eq >= 0:
#         penalty_vec[eq] = penalty

#     K_mod = env.K_base + sparse.diags(penalty_vec, format="csr")
#     F_mod = env.F_base.copy()

#     U_vec = spsolve(K_mod, F_mod)

#     ux = np.zeros(env.n_nodes, dtype=np.float64)
#     uy = np.zeros(env.n_nodes, dtype=np.float64)
#     uz = np.zeros(env.n_nodes, dtype=np.float64)

#     valid_ux = env.ux_map >= 0
#     valid_uy = env.uy_map >= 0
#     valid_uz = env.uz_map >= 0

#     ux[valid_ux] = U_vec[env.ux_map[valid_ux]]
#     uy[valid_uy] = U_vec[env.uy_map[valid_uy]]
#     uz[valid_uz] = U_vec[env.uz_map[valid_uz]]

#     usum = np.sqrt(ux**2 + uy**2 + uz**2)

#     metrics = {
#         "ux": ux,
#         "uy": uy,
#         "uz": uz,
#         "usum": usum,
#         "max_abs_uz_mm": float(np.max(np.abs(uz)) * 1000.0),
#         "max_usum_mm": float(np.max(usum) * 1000.0),
#         "support_patch_node_count": len(support_node_ids),
#         "locator2_node_ids_python": locator2_node_ids,
#         "locator1_node_id_python": locator1_node_idx,
#     }

#     print("   ✅ Python Support+2-1 重分析完成")
#     print(f"   📘 Python Max |UZ| = {metrics['max_abs_uz_mm']:.6f} mm")
#     print(f"   📘 Python Max |U|  = {metrics['max_usum_mm']:.6f} mm")
#     return metrics


# # ==============================
# # Step 4: 布局示意图
# # ==============================
# def save_layout_figure(env, support_points, locator2_points, locator1_point, save_path=LAYOUT_PNG):
#     print("\n🎨 [Step 4] 生成布局示意图 ...")

#     nodes = np.asarray(env.nodes, dtype=float)
#     fig, ax = plt.subplots(figsize=(12, 5))

#     ax.scatter(nodes[:, 0], nodes[:, 1], s=3, c="lightgray", alpha=0.45, label="FEM nodes")

#     sp = np.asarray(support_points)
#     ax.scatter(
#         sp[:, 0], sp[:, 1], s=110, c="limegreen",
#         edgecolors="black", linewidths=0.8, marker="o",
#         label="AI supports (N)"
#     )
#     if len(sp) >= 3:
#         ax.scatter(
#             sp[:3, 0], sp[:3, 1], s=150, c="gold",
#             edgecolors="black", linewidths=1.0, marker="*",
#             label="Initial 3 supports"
#         )

#     l2 = np.asarray(locator2_points)
#     ax.scatter(
#         l2[:, 0], l2[:, 1], s=160, c="deepskyblue",
#         edgecolors="black", linewidths=1.0, marker="^",
#         label="2 locators (UY=0)"
#     )

#     l1 = np.asarray(locator1_point).reshape(1, 3)
#     ax.scatter(
#         l1[:, 0], l1[:, 1], s=180, c="magenta",
#         edgecolors="black", linewidths=1.0, marker="s",
#         label="1 locator (UX=0)"
#     )

#     for i, p in enumerate(sp, start=1):
#         ax.text(p[0], p[1], f"S{i}", fontsize=8, ha="left", va="bottom")
#     for i, p in enumerate(l2, start=1):
#         ax.text(p[0], p[1], f"L2-{i}", fontsize=9, ha="left", va="bottom", color="navy")
#     ax.text(l1[0, 0], l1[0, 1], "L1", fontsize=9, ha="left", va="bottom", color="purple")

#     ax.set_xlabel("X (m)")
#     ax.set_ylabel("Y (m)")
#     ax.set_title("Support + 2-1 Verification Layout")
#     ax.axis("equal")
#     ax.grid(True, alpha=0.3)
#     ax.legend(loc="best")
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300, bbox_inches="tight")
#     plt.close(fig)

#     print(f"   ✅ 已保存: {save_path}")


# # ==============================
# # Step 5: ANSYS 环境重建
# # ==============================
# def rebuild_ansys_model(mapdl):
#     print("\n🏭 [Step 5] ANSYS 重建模型 ...")

#     mapdl.clear()
#     mapdl.aux15()
#     mapdl.ioptn("GTOL", 0.05)
#     mapdl.ioptn("IGES", "SMOOTH")
#     mapdl.ioptn("MERG", "YES")
#     mapdl.ioptn("SOLID", "NO")
#     mapdl.igesin(CAD_PATH)
#     mapdl.finish()
#     mapdl.prep7()

#     mapdl.allsel()
#     xmin = mapdl.get_value("KP", 0, "MNLOC", "X")
#     xmax = mapdl.get_value("KP", 0, "MXLOC", "X")
#     if (xmax - xmin) > 10.0:
#         print("   ⚠️ 检测到 mm 单位，执行 x0.001 缩放 ...")
#         mapdl.arscale("ALL", "", "", 0.001, 0.001, 0.001, "", "", 1)

#     # 材料/单元：与 extract_data.py 完全一致
#     mapdl.et(1, "SHELL181")
#     mapdl.sectype(1, "SHELL")
#     mapdl.secdata(THICKNESS)

#     mapdl.mp("EX", 1, 157.7e9)
#     mapdl.mp("EY", 1, 9.05e9)
#     mapdl.mp("EZ", 1, 9.05e9)
#     mapdl.mp("GXY", 1, 4.69e9)
#     mapdl.mp("GXZ", 1, 4.69e9)
#     mapdl.mp("GYZ", 1, 3.24e9)
#     mapdl.mp("PRXY", 1, 0.3)
#     mapdl.mp("DENS", 1, DENSITY)

#     mapdl.esize(MESH_SIZE)
#     mapdl.mshape(1, "2D")
#     mapdl.mshkey(0)
#     mapdl.shpp("OFF", "ALL")
#     mapdl.amesh("ALL")

#     n_node = int(mapdl.mesh.n_node)
#     print(f"   ✅ ANSYS 网格完成: {n_node} nodes")


# # ==============================
# # Step 6: 点坐标映射到 ANSYS 节点
# # ==============================
# def map_points_to_ansys_nodes(mapdl, points):
#     current_nodes = np.asarray(mapdl.mesh.nodes, dtype=float)
#     current_nnum = np.asarray(mapdl.mesh.nnum, dtype=int)
#     tree = KDTree(current_nodes)

#     mapped_nids = []
#     mapped_dists = []
#     for pt in np.asarray(points, dtype=float):
#         dist, idx = tree.query(pt)
#         mapped_nids.append(int(current_nnum[idx]))
#         mapped_dists.append(float(dist))

#     return mapped_nids, mapped_dists

# def map_support_patches_to_ansys_nodes(mapdl, support_points, radius):
#     """
#     在 ANSYS 当前网格上，为每个 support point 找到半径域内的节点集合。
#     返回:
#         patch_node_ids_list: list[list[int]]，每个支撑点对应一个节点号列表
#         patch_node_coords_list: list[np.ndarray]，每个支撑点对应的节点坐标
#         patch_sizes: list[int]，每个 patch 的节点数
#     """
#     current_nodes = np.asarray(mapdl.mesh.nodes, dtype=float)   # (N, 3)
#     current_nnum = np.asarray(mapdl.mesh.nnum, dtype=int)       # (N,)
#     tree = KDTree(current_nodes)

#     support_points = np.asarray(support_points, dtype=float)
#     patch_indices_list = tree.query_ball_point(support_points, radius)

#     patch_node_ids_list = []
#     patch_node_coords_list = []
#     patch_sizes = []

#     for idx_list in patch_indices_list:
#         idx_arr = np.asarray(idx_list, dtype=int)
#         if idx_arr.size == 0:
#             patch_node_ids_list.append([])
#             patch_node_coords_list.append(np.empty((0, 3), dtype=float))
#             patch_sizes.append(0)
#             continue

#         node_ids = current_nnum[idx_arr].astype(int).tolist()
#         node_coords = current_nodes[idx_arr]

#         patch_node_ids_list.append(node_ids)
#         patch_node_coords_list.append(node_coords)
#         patch_sizes.append(len(node_ids))

#     return patch_node_ids_list, patch_node_coords_list, patch_sizes
# # ==============================
# # Step 7: ANSYS 约束 + 求解
# # 严格复刻 verify_in_ansys.py 的出图习惯
# # ==============================
# def solve_ansys_support_plus_21(mapdl, env, support_points, locator2_points, locator1_point):
#     print("\n🔒 [Step 6] 在 ANSYS 中施加 Support + 2-1 约束并求解 ...")

#     # ---- A. N 支撑点：改为 patch 模型（与 Python 端一致） ----
#     support_patch_ids_list, support_patch_coords_list, support_patch_sizes = \
#         map_support_patches_to_ansys_nodes(mapdl, support_points, SUPPORT_RADIUS)

#     if any(sz == 0 for sz in support_patch_sizes):
#         raise RuntimeError(f"存在空支撑 patch，sizes = {support_patch_sizes}")

#     # 同时保留“中心点 -> 最近节点”的映射，仅用于打印参考
#     support_center_ids, support_center_dists = map_points_to_ansys_nodes(mapdl, support_points)

#     # ---- B. 2-1 定位点仍保持单节点模型（与 Python 端一致） ----
#     locator2_ids, locator2_dists = map_points_to_ansys_nodes(mapdl, locator2_points)
#     locator1_ids, locator1_dists = map_points_to_ansys_nodes(
#         mapdl, np.asarray(locator1_point).reshape(1, 3)
#     )

#     print(f"   支撑中心点映射平均距离: {np.mean(support_center_dists):.6e} m")
#     print(f"   支撑 patch 节点数: {support_patch_sizes}")
#     print(f"   支撑 patch 平均节点数: {np.mean(support_patch_sizes):.2f}")
#     print(f"   2定位点映射平均距离: {np.mean(locator2_dists):.6e} m")
#     print(f"   1定位点映射距离:     {locator1_dists[0]:.6e} m")

#     # ---- 求解设置 ----
#     mapdl.run("/SOLU")
#     mapdl.antype("STATIC")
#     mapdl.acel(0, 0, 9.8)
#     mapdl.ddele("ALL", "ALL")

#     # ---- 8个 N 支撑点：对 patch 内所有节点施加 UZ=0 ----
#     all_support_patch_node_ids = set()
#     for patch_ids in support_patch_ids_list:
#         for nid in patch_ids:
#             all_support_patch_node_ids.add(int(nid))
#             mapdl.d(int(nid), "UZ", 0)

#     # ---- 2 个定位点：UY=0（单节点） ----
#     for nid in locator2_ids:
#         mapdl.d(int(nid), "UY", 0)

#     # ---- 1 个定位点：UX=0（单节点） ----
#     mapdl.d(int(locator1_ids[0]), "UX", 0)

#     # --- 约束图 ---
#     before_pngs = list_job_pngs()
#     mapdl.run("/SHOW, PNG")
#     mapdl.run("/GFILE, 800")
#     mapdl.run("/VUP, 1, Z")
#     mapdl.run("/VIEW, 1, 1, -1.73, 1")
#     mapdl.run("/PBC, U, , 1")
#     mapdl.run("EPLOT")
#     mapdl.run("/SHOW, CLOSE")
#     copied0 = copy_newest_job_png(before_pngs, ANSYS_CONSTRAINT_PNG)
#     if copied0:
#         print(f"   📸 已生成约束图: {ANSYS_CONSTRAINT_PNG}")

#     # --- 求解 ---
#     print("   🧮 求解中 (NLGEOM=OFF)...")
#     mapdl.nlgeom("OFF")
#     mapdl.ncnv(2)
#     mapdl.eqslv("SPARSE")
#     mapdl.solve()
#     mapdl.finish()

#     # --- 后处理 ---
#     mapdl.post1()
#     mapdl.set("LAST")

#     # 直接分别取 X/Y/Z，再自己合成总位移，避免 SUM=nan
#     try:
#         ux_arr_ansys = np.asarray(mapdl.post_processing.nodal_displacement("X"), dtype=float)
#         uy_arr_ansys = np.asarray(mapdl.post_processing.nodal_displacement("Y"), dtype=float)
#         uz_arr_ansys = np.asarray(mapdl.post_processing.nodal_displacement("Z"), dtype=float)

#         usum_arr_ansys = np.sqrt(ux_arr_ansys**2 + uy_arr_ansys**2 + uz_arr_ansys**2)

#         ansys_max_abs_uz_mm = float(np.max(np.abs(uz_arr_ansys)) * 1000.0)
#         ansys_max_usum_mm = float(np.max(np.abs(usum_arr_ansys)) * 1000.0)
#     except Exception:
#         ux_arr_ansys = None
#         uy_arr_ansys = None
#         uz_arr_ansys = None
#         usum_arr_ansys = None
#         ansys_max_abs_uz_mm = np.nan
#         ansys_max_usum_mm = np.nan

#     # --- 结果图 ---
#     before_pngs = list_job_pngs()
#     mapdl.run("/SHOW, PNG")
#     mapdl.run("/GFILE, 1200")
#     mapdl.run("/RGB,INDEX,100,100,100,0")
#     mapdl.run("/RGB,INDEX,0,0,0,15")
#     mapdl.run("/DSCALE, 1, 10")
#     mapdl.run("/VUP, 1, Z")
#     mapdl.run("/VIEW, 1, 1, -1.73, 1")
#     mapdl.run("PLNSOL, U, Z")
#     mapdl.run("/SHOW, CLOSE")
#     copied1 = copy_newest_job_png(before_pngs, ANSYS_RESULT_UZ_PNG)
#     if copied1:
#         print(f"   ✅ 已生成结果图: {ANSYS_RESULT_UZ_PNG}")

#     # 映射回 env.nodes 顺序，便于后续数值逐点对齐
#     current_nodes = np.asarray(mapdl.mesh.nodes, dtype=float)
#     tree_ansys = KDTree(current_nodes)
#     _, idx_map = tree_ansys.query(np.asarray(env.nodes, dtype=float))

#     uz_env_order = uz_arr_ansys[idx_map] if uz_arr_ansys is not None else None
#     usum_env_order = usum_arr_ansys[idx_map] if usum_arr_ansys is not None else None

#     return {
#         "support_center_ids": support_center_ids,
#         "support_patch_ids_list": support_patch_ids_list,
#         "support_patch_sizes": support_patch_sizes,
#         "locator2_ids": locator2_ids,
#         "locator1_id": int(locator1_ids[0]),
#         "ansys_max_abs_uz_mm": ansys_max_abs_uz_mm,
#         "ansys_max_usum_mm": ansys_max_usum_mm,
#         "support_center_mapping_dists": support_center_dists,
#         "locator2_mapping_dists": locator2_dists,
#         "locator1_mapping_dist": locator1_dists[0],
#         "uz": uz_env_order,
#         "usum": usum_env_order,
#     }
# # ==============================
# # Step 8A: Python MDSM 的 3D 真比例云图
# # 参考 evaluate.py 的 plot_surface 逻辑
# # ==============================
# def save_python_mdsm_true_scale_3d(env, py_metrics, support_points, locator2_points, locator1_point,
#                                    save_path=PYTHON_MDSM_UZ_PNG):
#     print("\n🎨 [Step 7] 生成 Python MDSM 3D 真比例变形云图 ...")

#     x_raw = env.nodes[:, 0]
#     y_raw = env.nodes[:, 1]
#     z_raw = env.nodes[:, 2]
#     range_x = np.ptp(x_raw)
#     range_y = np.ptp(y_raw)
#     range_z = np.ptp(z_raw)

#     val = np.abs(py_metrics["uz"]) * 1000.0  # mm
#     vmax = np.max(val)
#     if vmax < 1e-9:
#         vmax = 1e-9

#     xi = np.linspace(x_raw.min(), x_raw.max(), 200)
#     yi = np.linspace(y_raw.min(), y_raw.max(), 200)
#     Xi, Yi = np.meshgrid(xi, yi)

#     Zi = griddata((x_raw, y_raw), z_raw, (Xi, Yi), method="cubic")
#     Zi = fill_nan_grid(Zi, Xi, Yi, x_raw, y_raw, z_raw)

#     Ci = griddata((x_raw, y_raw), val, (Xi, Yi), method="cubic")
#     Ci = fill_nan_grid(Ci, Xi, Yi, x_raw, y_raw, val)

#     norm_C = np.clip(Ci / vmax, 0.0, 1.0)

#     fig = plt.figure(figsize=(10, 7))
#     ax = fig.add_subplot(111, projection="3d")

#     ax.plot_surface(
#         Xi, Yi, Zi,
#         facecolors=cm.jet(norm_C),
#         rstride=2, cstride=2,
#         shade=False,
#         linewidth=0,
#         antialiased=True
#     )

#     # 画支撑点和定位点
#     def draw_vertical_marker(px, py, pz, color, marker, size=150):
#         arrow_len = max(range_z * 0.2, 0.05)
#         z_top = pz + arrow_len
#         ax.plot([px, px], [py, py], [pz, z_top], color="black", lw=1.5)
#         ax.scatter(px, py, z_top, c=color, s=size, edgecolors="black", marker=marker, zorder=100)

#     for i, (px, py, pz) in enumerate(np.asarray(support_points)):
#         color = "yellow" if i < 3 else "lime"
#         draw_vertical_marker(px, py, pz, color, "o", 150)

#     for (px, py, pz) in np.asarray(locator2_points):
#         draw_vertical_marker(px, py, pz, "cyan", "^", 170)

#     px, py, pz = np.asarray(locator1_point)
#     draw_vertical_marker(px, py, pz, "magenta", "s", 180)

#     ax.set_title(f"Python MDSM Predicted UZ\nMax: {np.max(val):.4e} mm", fontsize=15, fontweight="bold")
#     ax.set_xlabel("X (m)")
#     ax.set_ylabel("Y (m)")
#     ax.set_zlabel("Z (m)")
#     ax.set_box_aspect((range_x, range_y, range_z))
#     ax.view_init(elev=30, azim=-60)
#     ax.xaxis.pane.fill = False
#     ax.yaxis.pane.fill = False
#     ax.zaxis.pane.fill = False

#     m = cm.ScalarMappable(cmap=cm.jet)
#     m.set_array(np.array([0, vmax]))
#     cbar = fig.colorbar(m, ax=ax, orientation="horizontal", fraction=0.05, pad=0.08)
#     cbar.set_label("Deformation (mm) [Red=Max, Blue=Min]", fontsize=11)

#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300, bbox_inches="tight")
#     plt.close(fig)

#     print(f"   ✅ 已保存: {save_path}")


# # ==============================
# # Step 8B: MDSM 与 ANSYS 结果对比图
# # 左：Python MDSM 3D 真比例云图
# # 右：ANSYS 真实结果图
# # ==============================
# def save_mdsm_vs_ansys_comparison(mdsm_png=PYTHON_MDSM_UZ_PNG,
#                                   ansys_png=ANSYS_RESULT_UZ_PNG,
#                                   save_path=COMPARE_PNG):
#     print("\n🖼️ [Step 8] 生成 MDSM vs ANSYS 对比图 ...")

#     if not os.path.exists(mdsm_png):
#         raise FileNotFoundError(f"找不到 Python MDSM 云图: {mdsm_png}")
#     if not os.path.exists(ansys_png):
#         raise FileNotFoundError(f"找不到 ANSYS 真实结果图: {ansys_png}")

#     mdsm_img = plt.imread(mdsm_png)
#     ansys_img = plt.imread(ansys_png)

#     fig, axes = plt.subplots(1, 2, figsize=(18, 7))
#     axes[0].imshow(mdsm_img)
#     axes[0].set_title("Python MDSM Predicted UZ", fontsize=14)
#     axes[0].axis("off")

#     axes[1].imshow(ansys_img)
#     axes[1].set_title("ANSYS True UZ (Axonometric View)", fontsize=14)
#     axes[1].axis("off")

#     plt.suptitle("Support + 2-1: MDSM Prediction vs ANSYS Result", fontsize=16)
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300, bbox_inches="tight")
#     plt.close(fig)

#     print(f"   ✅ 已保存: {save_path}")


# # ==============================
# # 主流程
# # ==============================
# def main():
#     print("🚀 启动验证：AI 8点支撑（N） + 独立 2-1 定位")
#     print("=" * 72)

#     # Step 1
#     env, support_points, full_constraint_python_mm = infer_ai_support_layout()

#     # Step 2
#     locator2_points, locator1_point, meta = select_21_locator_points(env, support_points)

#     format_points("AI 支撑点（N）", support_points)
#     format_points("2 点定位器", locator2_points)
#     format_points("1 点定位器", np.asarray(locator1_point).reshape(1, 3))

#     # Step 3
#     py_metrics = solve_python_support_plus_21(
#         env=env,
#         support_points=support_points,
#         locator2_points=locator2_points,
#         locator1_point=locator1_point,
#         penalty=PENALTY,
#     )

#     # Step 4
#     save_layout_figure(
#         env=env,
#         support_points=support_points,
#         locator2_points=locator2_points,
#         locator1_point=locator1_point,
#         save_path=LAYOUT_PNG
#     )

#     # Step 5 ~ 6
#     ensure_clean_workdir(WORK_DIR)

#     stop_log = threading.Event()
#     mapdl = launch_mapdl(
#         run_location=WORK_DIR,
#         jobname=JOB_NAME,
#         nproc=1,
#         additional_switches="-smp",
#         override=True
#     )

#     t = threading.Thread(target=tail_ansys_log, args=(stop_log, WORK_DIR, JOB_NAME))
#     t.daemon = True
#     t.start()

#     try:
#         rebuild_ansys_model(mapdl)

#         ansys_metrics = solve_ansys_support_plus_21(
#             mapdl=mapdl,
#             env=env,
#             support_points=support_points,
#             locator2_points=locator2_points,
#             locator1_point=locator1_point
#         )

#         # Step 7
#         save_python_mdsm_true_scale_3d(
#             env=env,
#             py_metrics=py_metrics,
#             support_points=support_points,
#             locator2_points=locator2_points,
#             locator1_point=locator1_point,
#             save_path=PYTHON_MDSM_UZ_PNG
#         )

#         # Step 8
#         save_mdsm_vs_ansys_comparison(
#             mdsm_png=PYTHON_MDSM_UZ_PNG,
#             ansys_png=ANSYS_RESULT_UZ_PNG,
#             save_path=COMPARE_PNG
#         )

#         # 汇总输出
#         print("\n" + "=" * 72)
#         print("📊 验证结果汇总")
#         print("=" * 72)
#         print(f"现有环境（全向刚约束） Python Max |UZ| : {full_constraint_python_mm:.6f} mm")
#         print(f"独立脚本（Support+2-1） Python Max |UZ|: {py_metrics['max_abs_uz_mm']:.6f} mm")
#         print(f"独立脚本（Support+2-1） Python Max |U| : {py_metrics['max_usum_mm']:.6f} mm")
#         print(f"ANSYS 真实验证          Max |UZ| : {ansys_metrics['ansys_max_abs_uz_mm']:.6f} mm")
#         print(f"ANSYS 真实验证          Max |U|  : {ansys_metrics['ansys_max_usum_mm']:.6f} mm")
#         print("-" * 72)

#         if np.isfinite(ansys_metrics["ansys_max_abs_uz_mm"]) and py_metrics["max_abs_uz_mm"] > 0:
#             ratio = ansys_metrics["ansys_max_abs_uz_mm"] / py_metrics["max_abs_uz_mm"]
#             print(f"Python/ANSYS 在 Max |UZ| 上的比值: {ratio:.6f}")

#         print("\n🔧 约束定义：")
#         print("  - 8 个 AI 点全部作为 N 类支撑点，仅施加 UZ=0")
#         print("  - 2 个独立定位点施加 UY=0")
#         print("  - 1 个独立定位点施加 UX=0")
#         print("  - 几何非线性: OFF（与现有线性验证口径一致）")

#         print("\n📁 输出图像：")
#         print(f"  - {LAYOUT_PNG}")
#         print(f"  - {ANSYS_CONSTRAINT_PNG}")
#         print(f"  - {ANSYS_RESULT_UZ_PNG}")
#         print(f"  - {PYTHON_MDSM_UZ_PNG}")
#         print(f"  - {COMPARE_PNG}")

#     except Exception as e:
#         print(f"\n❌ 验证过程出错: {e}")
#         raise
#     finally:
#         stop_log.set()
#         try:
#             mapdl.exit()
#         except Exception:
#             pass


# if __name__ == "__main__":
#     main()
import os

# 解决 OpenMP / MKL 冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import glob
import shutil
import time
import threading
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib import cm
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.spatial import KDTree
from scipy.interpolate import griddata

from ansys.mapdl.core import launch_mapdl
from sb3_contrib import MaskablePPO
from smart_fixture_env import SmartFixtureEnv3D


# ==============================
# 配置区域
# ==============================
DATA_DIR = r"E:\ansys_data_final"
CAD_PATH = r"E:\ZJU\Learning baogao\0128.IGS"
MODEL_PATH = r"models_3d/final_model_3d"
TARGET_N = 8

# 必须与 extract_data.py 保持一致
THICKNESS = 0.005
MESH_SIZE = 0.04
DENSITY = 1600.0

WORK_DIR = "ansys_temp_verify_support_plus_21"
JOB_NAME = "verify_support21"

# 逻辑参数
SUPPORT_RADIUS = 0.05
LOCATOR_CLEARANCE = 0.12
BOUNDARY_TOL = 0.032
PENALTY = 1e15


# ==============================
# 基础工具
# ==============================
def scheme_file(prefix: str, scheme: str, ext: str = "png") -> str:
    return f"{prefix}_{scheme}.{ext}"


def tail_ansys_log(stop_event, work_dir, job_name):
    log_path = os.path.join(work_dir, f"{job_name}.out")
    while not os.path.exists(log_path):
        if stop_event.is_set():
            return
        time.sleep(0.1)

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        f.seek(0, 2)
        while not stop_event.is_set():
            line = f.readline()
            if line:
                low = line.lower()
                if ("error" in low) or ("warning" in low):
                    print(f"   [ANSYS] {line.strip()}")
            else:
                time.sleep(0.1)


def ensure_clean_workdir(work_dir):
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir, exist_ok=True)


def format_points(name, pts):
    arr = np.asarray(pts, dtype=float)
    print(f"\n{name}:")
    for i, p in enumerate(arr, start=1):
        print(f"  {i:02d}: ({p[0]:.6f}, {p[1]:.6f}, {p[2]:.6f})")


def list_job_pngs():
    pattern = os.path.join(WORK_DIR, f"{JOB_NAME}*.png")
    return set(glob.glob(pattern))


def copy_newest_job_png(before_set, dst_path):
    after_set = list_job_pngs()
    new_files = sorted(list(after_set - before_set), key=os.path.getmtime)
    if not new_files:
        all_files = sorted(list(after_set), key=os.path.getmtime)
        if not all_files:
            return False
        src = all_files[-1]
    else:
        src = new_files[-1]
    shutil.copy(src, dst_path)
    return True


def fill_nan_grid(arr, Xi, Yi, x_raw, y_raw, raw_val):
    """
    对 griddata 产生的 NaN 进行 nearest 兜底填补
    """
    if np.isnan(arr).any():
        arr_nearest = griddata((x_raw, y_raw), raw_val, (Xi, Yi), method="nearest")
        mask = np.isnan(arr)
        arr[mask] = arr_nearest[mask]
    return arr


# ==============================
# Step 1: AI 推理，得到 8 个支撑点
# ==============================
def infer_ai_support_layout():
    if not os.path.exists(MODEL_PATH + ".zip"):
        raise FileNotFoundError(f"找不到模型文件: {MODEL_PATH}.zip")

    print("🤖 [Step 1] AI 正在推理 8 个 N 类支撑点 ...")
    env = SmartFixtureEnv3D(data_dir=DATA_DIR, target_n=TARGET_N)
    model = MaskablePPO.load(MODEL_PATH)

    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(
            obs,
            action_masks=env.action_masks(),
            deterministic=True
        )
        obs, _, done, _, _ = env.step(action)

    support_points = np.asarray(env.fixtures, dtype=float)
    full_constraint_python_mm = float(env.last_max_def)

    print(f"   ✅ AI 推理完成，共 {len(support_points)} 个支撑点")
    print(f"   📘 现有环境（全向刚约束）Python 结果: {full_constraint_python_mm:.6f} mm")
    return env, support_points, full_constraint_python_mm


# ==============================
# Step 2: 单独构造 2-1 定位点
# scheme:
#   - x2_y1: X方向作为2，Y方向作为1
#   - y2_x1: Y方向作为2，X方向作为1
# ==============================
def select_21_locator_points(env, support_points, scheme="x2_y1"):
    print(f"\n📐 [Step 2] 自动构造独立的 2-1 定位点 (scheme={scheme}) ...")

    nodes = np.asarray(env.nodes, dtype=float)
    support_xy = np.asarray(support_points[:, :2], dtype=float)
    support_tree = KDTree(support_xy)

    x = nodes[:, 0]
    y = nodes[:, 1]

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    x_span = x_max - x_min
    y_span = y_max - y_min

    if scheme == "x2_y1":
        # 两个点放在 y = const 边上，沿 X 方向分布
        two_edge_primary = "ymin"
        two_edge_fallback = "ymax"
        one_edge_primary = "xmin"
        one_edge_fallback = "xmax"
        two_target_fracs = [0.25, 0.75]
        one_target_frac = 0.50
        two_dof = "UY"
        one_dof = "UX"

    elif scheme == "y2_x1":
        # 两个点放在 x = const 边上，沿 Y 方向分布
        two_edge_primary = "xmin"
        two_edge_fallback = "xmax"
        one_edge_primary = "ymin"
        one_edge_fallback = "ymax"
        two_target_fracs = [0.25, 0.75]
        one_target_frac = 0.50
        two_dof = "UX"
        one_dof = "UY"

    else:
        raise ValueError(f"未知 scheme: {scheme}")

    used_indices = set()

    def edge_mask(edge_name, tol):
        if edge_name == "ymin":
            return np.abs(nodes[:, 1] - y_min) <= tol
        if edge_name == "ymax":
            return np.abs(nodes[:, 1] - y_max) <= tol
        if edge_name == "xmin":
            return np.abs(nodes[:, 0] - x_min) <= tol
        if edge_name == "xmax":
            return np.abs(nodes[:, 0] - x_max) <= tol
        raise ValueError(f"未知边界名: {edge_name}")

    def target_coordinate(edge_name, frac):
        if edge_name in ("ymin", "ymax"):
            return x_min + frac * x_span
        return y_min + frac * y_span

    def edge_parameter(edge_name):
        if edge_name in ("ymin", "ymax"):
            return nodes[:, 0]
        return nodes[:, 1]

    def pick_node_on_edge(edge_name, frac, tol, clearance):
        mask = edge_mask(edge_name, tol)
        if not np.any(mask):
            return None

        d_to_support, _ = support_tree.query(nodes[:, :2])
        mask = mask & (d_to_support >= clearance)

        if used_indices:
            used_mask = np.ones(len(nodes), dtype=bool)
            used_mask[list(used_indices)] = False
            mask = mask & used_mask

        candidate_ids = np.where(mask)[0]
        if len(candidate_ids) == 0:
            return None

        param = edge_parameter(edge_name)[candidate_ids]
        target = target_coordinate(edge_name, frac)
        best_local = np.argmin(np.abs(param - target))
        return int(candidate_ids[best_local])

    def robust_pick(edge_primary, edge_fallback, frac):
        for tol in [BOUNDARY_TOL, BOUNDARY_TOL * 1.5, BOUNDARY_TOL * 2.0]:
            for clr in [LOCATOR_CLEARANCE, LOCATOR_CLEARANCE * 0.7, LOCATOR_CLEARANCE * 0.4]:
                idx = pick_node_on_edge(edge_primary, frac, tol, clr)
                if idx is not None:
                    return idx
                idx = pick_node_on_edge(edge_fallback, frac, tol, clr)
                if idx is not None:
                    return idx
        raise RuntimeError(
            f"无法为 edge={edge_primary}/{edge_fallback}, frac={frac:.2f} 选出合适定位点"
        )

    locator2_idx = []
    for frac in two_target_fracs:
        idx = robust_pick(two_edge_primary, two_edge_fallback, frac)
        locator2_idx.append(idx)
        used_indices.add(idx)

    locator1_idx = robust_pick(one_edge_primary, one_edge_fallback, one_target_frac)
    used_indices.add(locator1_idx)

    locator2_points = nodes[locator2_idx]
    locator1_point = nodes[locator1_idx]

    meta = {
        "scheme": scheme,
        "two_edge_primary": two_edge_primary,
        "two_edge_fallback": two_edge_fallback,
        "one_edge_primary": one_edge_primary,
        "one_edge_fallback": one_edge_fallback,
        "locator2_indices": locator2_idx,
        "locator1_index": locator1_idx,
        "two_dof": two_dof,
        "one_dof": one_dof,
    }

    print(f"   ✅ 2点定位边: {two_edge_primary} (fallback={two_edge_fallback}), DOF={two_dof}")
    print(f"   ✅ 1点定位边: {one_edge_primary} (fallback={one_edge_fallback}), DOF={one_dof}")
    return locator2_points, locator1_point, meta


# ==============================
# Step 3: Python 端独立验证
# 支撑点: UZ = 0（在支撑半径邻域内施加）
# 2定位点 / 1定位点: 按 locator_meta 指定自由度
# ==============================
def solve_python_support_plus_21(env, support_points, locator2_points, locator1_point, locator_meta, penalty=PENALTY):
    print("\n🧮 [Step 3] Python 端进行 Support + 2-1 独立重分析 ...")

    ndof = env.K_base.shape[0]
    penalty_vec = np.zeros(ndof, dtype=np.float64)

    # A. N 类支撑点：UZ patch
    support_indices_list = env.fem_tree_3d.query_ball_point(np.asarray(support_points), SUPPORT_RADIUS)
    support_node_ids = set()
    for idx_list in support_indices_list:
        for node_idx in idx_list:
            support_node_ids.add(int(node_idx))
            eq = int(env.uz_map[node_idx])
            if eq >= 0:
                penalty_vec[eq] = penalty

    # B. 2 点定位
    locator2_node_ids = []
    for pt in np.asarray(locator2_points):
        _, node_idx = env.fem_tree_3d.query(pt)
        locator2_node_ids.append(int(node_idx))

        if locator_meta["two_dof"] == "UX":
            eq = int(env.ux_map[node_idx])
        elif locator_meta["two_dof"] == "UY":
            eq = int(env.uy_map[node_idx])
        else:
            raise ValueError(f"未知 two_dof: {locator_meta['two_dof']}")

        if eq >= 0:
            penalty_vec[eq] = penalty

    # C. 1 点定位
    _, locator1_node_idx = env.fem_tree_3d.query(np.asarray(locator1_point))
    locator1_node_idx = int(locator1_node_idx)

    if locator_meta["one_dof"] == "UX":
        eq = int(env.ux_map[locator1_node_idx])
    elif locator_meta["one_dof"] == "UY":
        eq = int(env.uy_map[locator1_node_idx])
    else:
        raise ValueError(f"未知 one_dof: {locator_meta['one_dof']}")

    if eq >= 0:
        penalty_vec[eq] = penalty

    K_mod = env.K_base + sparse.diags(penalty_vec, format="csr")
    F_mod = env.F_base.copy()

    U_vec = spsolve(K_mod, F_mod)

    ux = np.zeros(env.n_nodes, dtype=np.float64)
    uy = np.zeros(env.n_nodes, dtype=np.float64)
    uz = np.zeros(env.n_nodes, dtype=np.float64)

    valid_ux = env.ux_map >= 0
    valid_uy = env.uy_map >= 0
    valid_uz = env.uz_map >= 0

    ux[valid_ux] = U_vec[env.ux_map[valid_ux]]
    uy[valid_uy] = U_vec[env.uy_map[valid_uy]]
    uz[valid_uz] = U_vec[env.uz_map[valid_uz]]

    usum = np.sqrt(ux**2 + uy**2 + uz**2)

    metrics = {
        "ux": ux,
        "uy": uy,
        "uz": uz,
        "usum": usum,
        "max_abs_ux_mm": float(np.max(np.abs(ux)) * 1000.0),
        "max_abs_uy_mm": float(np.max(np.abs(uy)) * 1000.0),
        "max_abs_uz_mm": float(np.max(np.abs(uz)) * 1000.0),
        "max_usum_mm": float(np.max(usum) * 1000.0),
        "support_patch_node_count": len(support_node_ids),
        "locator2_node_ids_python": locator2_node_ids,
        "locator1_node_id_python": locator1_node_idx,
    }

    two_dof = locator_meta["two_dof"]   # 2点被约束的自由度
    one_dof = locator_meta["one_dof"]   # 1点被约束的自由度
    def _dof_tag(dof_name):
        if two_dof == dof_name:
            return "  ← 被2点约束"
        if one_dof == dof_name:
            return "  ← 被1点约束"
        return "  ← 自由分量"
    print("   ✅ Python Support+2-1 重分析完成")
    print(f"   📘 约束分配: 2点={two_dof}=0  |  1点={one_dof}=0  |  支撑=UZ=0")
    print(f"   📘 Python Max |UX| = {metrics['max_abs_ux_mm']:.6f} mm{_dof_tag('UX')}")
    print(f"   📘 Python Max |UY| = {metrics['max_abs_uy_mm']:.6f} mm{_dof_tag('UY')}")
    print(f"   📘 Python Max |UZ| = {metrics['max_abs_uz_mm']:.6f} mm  ← 被支撑约束")
    print(f"   📘 Python Max |U|  = {metrics['max_usum_mm']:.6f} mm")
    return metrics


# ==============================
# Step 4: 布局示意图
# ==============================
def save_layout_figure(env, support_points, locator2_points, locator1_point, locator_meta,
                       save_path):
    print("\n🎨 [Step 4] 生成布局示意图 ...")

    nodes = np.asarray(env.nodes, dtype=float)
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.scatter(nodes[:, 0], nodes[:, 1], s=3, c="lightgray", alpha=0.45, label="FEM nodes")

    sp = np.asarray(support_points)
    ax.scatter(
        sp[:, 0], sp[:, 1], s=110, c="limegreen",
        edgecolors="black", linewidths=0.8, marker="o",
        label="AI supports (N)"
    )
    if len(sp) >= 3:
        ax.scatter(
            sp[:3, 0], sp[:3, 1], s=150, c="gold",
            edgecolors="black", linewidths=1.0, marker="*",
            label="Initial 3 supports"
        )

    l2 = np.asarray(locator2_points)
    ax.scatter(
        l2[:, 0], l2[:, 1], s=160, c="deepskyblue",
        edgecolors="black", linewidths=1.0, marker="^",
        label=f"2 locators ({locator_meta['two_dof']}=0)"
    )

    l1 = np.asarray(locator1_point).reshape(1, 3)
    ax.scatter(
        l1[:, 0], l1[:, 1], s=180, c="magenta",
        edgecolors="black", linewidths=1.0, marker="s",
        label=f"1 locator ({locator_meta['one_dof']}=0)"
    )

    for i, p in enumerate(sp, start=1):
        ax.text(p[0], p[1], f"S{i}", fontsize=8, ha="left", va="bottom")
    for i, p in enumerate(l2, start=1):
        ax.text(p[0], p[1], f"L2-{i}", fontsize=9, ha="left", va="bottom", color="navy")
    ax.text(l1[0, 0], l1[0, 1], "L1", fontsize=9, ha="left", va="bottom", color="purple")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"Support + 2-1 Verification Layout ({locator_meta['scheme']})")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"   ✅ 已保存: {save_path}")


# ==============================
# Step 5: ANSYS 环境重建
# ==============================
def rebuild_ansys_model(mapdl):
    print("\n🏭 [Step 5] ANSYS 重建模型 ...")

    mapdl.clear()
    mapdl.aux15()
    mapdl.ioptn("GTOL", 0.05)
    mapdl.ioptn("IGES", "SMOOTH")
    mapdl.ioptn("MERG", "YES")
    mapdl.ioptn("SOLID", "NO")
    mapdl.igesin(CAD_PATH)
    mapdl.finish()
    mapdl.prep7()

    mapdl.allsel()
    xmin = mapdl.get_value("KP", 0, "MNLOC", "X")
    xmax = mapdl.get_value("KP", 0, "MXLOC", "X")
    if (xmax - xmin) > 10.0:
        print("   ⚠️ 检测到 mm 单位，执行 x0.001 缩放 ...")
        mapdl.arscale("ALL", "", "", 0.001, 0.001, 0.001, "", "", 1)

    # 材料/单元：与 extract_data.py 完全一致
    mapdl.et(1, "SHELL181")
    mapdl.sectype(1, "SHELL")
    mapdl.secdata(THICKNESS)

    mapdl.mp("EX", 1, 157.7e9)
    mapdl.mp("EY", 1, 9.05e9)
    mapdl.mp("EZ", 1, 9.05e9)
    mapdl.mp("GXY", 1, 4.69e9)
    mapdl.mp("GXZ", 1, 4.69e9)
    mapdl.mp("GYZ", 1, 3.24e9)
    mapdl.mp("PRXY", 1, 0.3)
    mapdl.mp("DENS", 1, DENSITY)

    mapdl.esize(MESH_SIZE)
    mapdl.mshape(1, "2D")
    mapdl.mshkey(0)
    mapdl.shpp("OFF", "ALL")
    mapdl.amesh("ALL")

    n_node = int(mapdl.mesh.n_node)
    print(f"   ✅ ANSYS 网格完成: {n_node} nodes")


# ==============================
# Step 6: 点坐标映射到 ANSYS 节点
# ==============================
def map_points_to_ansys_nodes(mapdl, points):
    current_nodes = np.asarray(mapdl.mesh.nodes, dtype=float)
    current_nnum = np.asarray(mapdl.mesh.nnum, dtype=int)
    tree = KDTree(current_nodes)

    mapped_nids = []
    mapped_dists = []
    for pt in np.asarray(points, dtype=float):
        dist, idx = tree.query(pt)
        mapped_nids.append(int(current_nnum[idx]))
        mapped_dists.append(float(dist))

    return mapped_nids, mapped_dists


def map_support_patches_to_ansys_nodes(mapdl, support_points, radius):
    """
    在 ANSYS 当前网格上，为每个 support point 找到半径域内的节点集合。
    """
    current_nodes = np.asarray(mapdl.mesh.nodes, dtype=float)
    current_nnum = np.asarray(mapdl.mesh.nnum, dtype=int)
    tree = KDTree(current_nodes)

    support_points = np.asarray(support_points, dtype=float)
    patch_indices_list = tree.query_ball_point(support_points, radius)

    patch_node_ids_list = []
    patch_node_coords_list = []
    patch_sizes = []

    for idx_list in patch_indices_list:
        idx_arr = np.asarray(idx_list, dtype=int)
        if idx_arr.size == 0:
            patch_node_ids_list.append([])
            patch_node_coords_list.append(np.empty((0, 3), dtype=float))
            patch_sizes.append(0)
            continue

        node_ids = current_nnum[idx_arr].astype(int).tolist()
        node_coords = current_nodes[idx_arr]

        patch_node_ids_list.append(node_ids)
        patch_node_coords_list.append(node_coords)
        patch_sizes.append(len(node_ids))

    return patch_node_ids_list, patch_node_coords_list, patch_sizes


# ==============================
# Step 7: ANSYS 约束 + 求解
# 严格复刻 verify_in_ansys.py 的出图习惯
# ==============================
def solve_ansys_support_plus_21(mapdl, env, support_points, locator2_points, locator1_point,
                                locator_meta, constraint_png, result_png):
    print("\n🔒 [Step 6] 在 ANSYS 中施加 Support + 2-1 约束并求解 ...")

    # ---- A. N 支撑点：patch 模型 ----
    support_patch_ids_list, _, support_patch_sizes = \
        map_support_patches_to_ansys_nodes(mapdl, support_points, SUPPORT_RADIUS)

    if any(sz == 0 for sz in support_patch_sizes):
        raise RuntimeError(f"存在空支撑 patch，sizes = {support_patch_sizes}")

    support_center_ids, support_center_dists = map_points_to_ansys_nodes(mapdl, support_points)

    # ---- B. 2-1 定位点：单节点 ----
    locator2_ids, locator2_dists = map_points_to_ansys_nodes(mapdl, locator2_points)
    locator1_ids, locator1_dists = map_points_to_ansys_nodes(
        mapdl, np.asarray(locator1_point).reshape(1, 3)
    )

    print(f"   支撑中心点映射平均距离: {np.mean(support_center_dists):.6e} m")
    print(f"   支撑 patch 节点数: {support_patch_sizes}")
    print(f"   支撑 patch 平均节点数: {np.mean(support_patch_sizes):.2f}")
    print(f"   2定位点映射平均距离: {np.mean(locator2_dists):.6e} m")
    print(f"   1定位点映射距离:     {locator1_dists[0]:.6e} m")

    # ---- 求解设置 ----
    mapdl.run("/SOLU")
    mapdl.antype("STATIC")
    mapdl.acel(0, 0, 9.8)
    mapdl.ddele("ALL", "ALL")

    # ---- N 支撑点：patch 内全部 UZ=0 ----
    for patch_ids in support_patch_ids_list:
        for nid in patch_ids:
            mapdl.d(int(nid), "UZ", 0)

    # ---- 2 个定位点：按 scheme 约束 ----
    for nid in locator2_ids:
        mapdl.d(int(nid), locator_meta["two_dof"], 0)

    # ---- 1 个定位点：按 scheme 约束 ----
    mapdl.d(int(locator1_ids[0]), locator_meta["one_dof"], 0)

    # --- 约束图 ---
    before_pngs = list_job_pngs()
    mapdl.run("/SHOW, PNG")
    mapdl.run("/GFILE, 800")
    mapdl.run("/VUP, 1, Z")
    mapdl.run("/VIEW, 1, 1, -1.73, 1")
    mapdl.run("/PBC, U, , 1")
    mapdl.run("EPLOT")
    mapdl.run("/SHOW, CLOSE")
    copied0 = copy_newest_job_png(before_pngs, constraint_png)
    if copied0:
        print(f"   📸 已生成约束图: {constraint_png}")

    # --- 求解 ---
    print("   🧮 求解中 (NLGEOM=OFF)...")
    mapdl.nlgeom("OFF")
    mapdl.ncnv(2)
    mapdl.eqslv("SPARSE")
    mapdl.solve()
    mapdl.finish()

    # --- 后处理 ---
    mapdl.post1()
    mapdl.set("LAST")

    try:
        ux_arr_ansys = np.asarray(mapdl.post_processing.nodal_displacement("X"), dtype=float)
        uy_arr_ansys = np.asarray(mapdl.post_processing.nodal_displacement("Y"), dtype=float)
        uz_arr_ansys = np.asarray(mapdl.post_processing.nodal_displacement("Z"), dtype=float)

        usum_arr_ansys = np.sqrt(ux_arr_ansys**2 + uy_arr_ansys**2 + uz_arr_ansys**2)

        ansys_max_abs_uz_mm = float(np.max(np.abs(uz_arr_ansys)) * 1000.0)
        ansys_max_abs_ux_mm = float(np.max(np.abs(ux_arr_ansys)) * 1000.0)
        ansys_max_abs_uy_mm = float(np.max(np.abs(uy_arr_ansys)) * 1000.0)
        ansys_max_usum_mm = float(np.max(np.abs(usum_arr_ansys)) * 1000.0)
    except Exception:
        ux_arr_ansys = None
        uy_arr_ansys = None
        uz_arr_ansys = None
        usum_arr_ansys = None
        ansys_max_abs_uz_mm = np.nan
        ansys_max_abs_ux_mm = np.nan
        ansys_max_abs_uy_mm = np.nan
        ansys_max_usum_mm = np.nan

    # --- 结果图 ---
    before_pngs = list_job_pngs()
    mapdl.run("/SHOW, PNG")
    mapdl.run("/GFILE, 1200")
    mapdl.run("/RGB,INDEX,100,100,100,0")
    mapdl.run("/RGB,INDEX,0,0,0,15")
    mapdl.run("/DSCALE, 1, 10")
    mapdl.run("/VUP, 1, Z")
    mapdl.run("/VIEW, 1, 1, -1.73, 1")
    mapdl.run("PLNSOL, U, Z")
    mapdl.run("/SHOW, CLOSE")
    copied1 = copy_newest_job_png(before_pngs, result_png)
    if copied1:
        print(f"   ✅ 已生成结果图: {result_png}")

    current_nodes = np.asarray(mapdl.mesh.nodes, dtype=float)
    tree_ansys = KDTree(current_nodes)
    _, idx_map = tree_ansys.query(np.asarray(env.nodes, dtype=float))

    uz_env_order = uz_arr_ansys[idx_map] if uz_arr_ansys is not None else None
    usum_env_order = usum_arr_ansys[idx_map] if usum_arr_ansys is not None else None

    return {
        "support_center_ids": support_center_ids,
        "support_patch_ids_list": support_patch_ids_list,
        "support_patch_sizes": support_patch_sizes,
        "locator2_ids": locator2_ids,
        "locator1_id": int(locator1_ids[0]),
        "ansys_max_abs_uz_mm": ansys_max_abs_uz_mm,
        "ansys_max_abs_ux_mm": ansys_max_abs_ux_mm,
        "ansys_max_abs_uy_mm": ansys_max_abs_uy_mm,
        "ansys_max_usum_mm": ansys_max_usum_mm,
        "support_center_mapping_dists": support_center_dists,
        "locator2_mapping_dists": locator2_dists,
        "locator1_mapping_dist": locator1_dists[0],
        "uz": uz_env_order,
        "usum": usum_env_order,
    }


# ==============================
# Step 8A: Python MDSM 的 3D 真比例云图
# ==============================
def save_python_mdsm_true_scale_3d(env, py_metrics, support_points, locator2_points, locator1_point,
                                   locator_meta, save_path):
    print("\n🎨 [Step 7] 生成 Python MDSM 3D 真比例变形云图 ...")

    x_raw = env.nodes[:, 0]
    y_raw = env.nodes[:, 1]
    z_raw = env.nodes[:, 2]
    range_x = np.ptp(x_raw)
    range_y = np.ptp(y_raw)
    range_z = np.ptp(z_raw)

    val = np.abs(py_metrics["uz"]) * 1000.0  # mm，取绝对值
    vmax = np.max(val)
    if vmax < 1e-9:
        vmax = 1e-9

    xi = np.linspace(x_raw.min(), x_raw.max(), 200)
    yi = np.linspace(y_raw.min(), y_raw.max(), 200)
    Xi, Yi = np.meshgrid(xi, yi)

    Zi = griddata((x_raw, y_raw), z_raw, (Xi, Yi), method="cubic")
    Zi = fill_nan_grid(Zi, Xi, Yi, x_raw, y_raw, z_raw)

    Ci = griddata((x_raw, y_raw), val, (Xi, Yi), method="cubic")
    Ci = fill_nan_grid(Ci, Xi, Yi, x_raw, y_raw, val)

    # 映射 [0, vmax] 到 [0, 1] 用于着色
    norm_C = np.clip(Ci / vmax, 0.0, 1.0)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    CMAP = cm.jet   # 蓝(小)→红(大)，与 evaluate.py 保持一致
    ax.plot_surface(
        Xi, Yi, Zi,
        facecolors=CMAP(norm_C),
        rstride=2, cstride=2,
        shade=False,
        linewidth=0,
        antialiased=True
    )

    def draw_vertical_marker(px, py, pz, color, marker, size=150):
        arrow_len = max(range_z * 0.2, 0.05)
        z_top = pz + arrow_len
        ax.plot([px, px], [py, py], [pz, z_top], color="black", lw=1.5)
        ax.scatter(px, py, z_top, c=color, s=size, edgecolors="black", marker=marker, zorder=100)

    for i, (px, py, pz) in enumerate(np.asarray(support_points)):
        color = "yellow" if i < 3 else "lime"
        draw_vertical_marker(px, py, pz, color, "o", 150)

    for (px, py, pz) in np.asarray(locator2_points):
        draw_vertical_marker(px, py, pz, "cyan", "^", 170)

    px, py, pz = np.asarray(locator1_point)
    draw_vertical_marker(px, py, pz, "magenta", "s", 180)

    ax.set_title(
        f"Python MDSM Predicted UZ [{locator_meta['scheme']}]\n"
        f"Max: {np.max(val):.4e} mm",
        fontsize=15, fontweight="bold"
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_box_aspect((range_x, range_y, range_z))
    ax.view_init(elev=30, azim=-60)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # 色条：[0, vmax]，与 evaluate.py 保持一致
    m = cm.ScalarMappable(cmap=CMAP)
    m.set_array(np.array([0, vmax]))
    cbar = fig.colorbar(m, ax=ax, orientation="horizontal", fraction=0.05, pad=0.08)
    cbar.set_label("Deformation (mm) [Red=Max, Blue=Min]", fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"   ✅ 已保存: {save_path}")


# ==============================
# Step 8B: MDSM 与 ANSYS 结果对比图
# ==============================
def save_mdsm_vs_ansys_comparison(mdsm_png, ansys_png, save_path, scheme):
    print("\n🖼️ [Step 8] 生成 MDSM vs ANSYS 对比图 ...")

    if not os.path.exists(mdsm_png):
        raise FileNotFoundError(f"找不到 Python MDSM 云图: {mdsm_png}")
    if not os.path.exists(ansys_png):
        raise FileNotFoundError(f"找不到 ANSYS 真实结果图: {ansys_png}")

    mdsm_img = plt.imread(mdsm_png)
    ansys_img = plt.imread(ansys_png)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    axes[0].imshow(mdsm_img)
    axes[0].set_title("Python MDSM Predicted UZ", fontsize=14)
    axes[0].axis("off")

    axes[1].imshow(ansys_img)
    axes[1].set_title("ANSYS True UZ (Axonometric View)", fontsize=14)
    axes[1].axis("off")

    plt.suptitle(f"Support + 2-1 ({scheme}): MDSM Prediction vs ANSYS Result", fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"   ✅ 已保存: {save_path}")


# ==============================
# 单个方案执行
# ==============================
def run_one_scheme(env, mapdl, support_points, scheme_name):
    print("\n" + "=" * 72)
    print(f"🚀 开始运行定位方案: {scheme_name}")
    print("=" * 72)

    layout_png = scheme_file("verify_support21_layout", scheme_name)
    constraint_png = scheme_file("verify_support21_constraints", scheme_name)
    result_png = scheme_file("verify_support21_result_uz", scheme_name)
    python_png = scheme_file("verify_support21_python_mdsm_3d", scheme_name)
    compare_png = scheme_file("verify_support21_mdsm_vs_ansys", scheme_name)

    locator2_points, locator1_point, locator_meta = select_21_locator_points(
        env, support_points, scheme=scheme_name
    )

    format_points(f"{scheme_name} - 2 点定位器", locator2_points)
    format_points(f"{scheme_name} - 1 点定位器", np.asarray(locator1_point).reshape(1, 3))

    py_metrics = solve_python_support_plus_21(
        env=env,
        support_points=support_points,
        locator2_points=locator2_points,
        locator1_point=locator1_point,
        locator_meta=locator_meta,
        penalty=PENALTY,
    )

    save_layout_figure(
        env=env,
        support_points=support_points,
        locator2_points=locator2_points,
        locator1_point=locator1_point,
        locator_meta=locator_meta,
        save_path=layout_png
    )

    rebuild_ansys_model(mapdl)

    ansys_metrics = solve_ansys_support_plus_21(
        mapdl=mapdl,
        env=env,
        support_points=support_points,
        locator2_points=locator2_points,
        locator1_point=locator1_point,
        locator_meta=locator_meta,
        constraint_png=constraint_png,
        result_png=result_png
    )

    save_python_mdsm_true_scale_3d(
        env=env,
        py_metrics=py_metrics,
        support_points=support_points,
        locator2_points=locator2_points,
        locator1_point=locator1_point,
        locator_meta=locator_meta,
        save_path=python_png
    )

    save_mdsm_vs_ansys_comparison(
        mdsm_png=python_png,
        ansys_png=result_png,
        save_path=compare_png,
        scheme=scheme_name
    )

    result = {
        "scheme": scheme_name,
        "locator_meta": locator_meta,
        "python_max_abs_uz_mm": py_metrics["max_abs_uz_mm"],
        "python_max_abs_ux_mm": py_metrics["max_abs_ux_mm"],
        "python_max_abs_uy_mm": py_metrics["max_abs_uy_mm"],
        "python_max_usum_mm": py_metrics["max_usum_mm"],
        "ansys_max_abs_uz_mm": ansys_metrics["ansys_max_abs_uz_mm"],
        "ansys_max_abs_ux_mm": ansys_metrics["ansys_max_abs_ux_mm"],
        "ansys_max_abs_uy_mm": ansys_metrics["ansys_max_abs_uy_mm"],
        "ansys_max_usum_mm": ansys_metrics["ansys_max_usum_mm"],
        "support_patch_sizes": ansys_metrics["support_patch_sizes"],
        "files": {
            "layout_png": layout_png,
            "constraint_png": constraint_png,
            "result_png": result_png,
            "python_png": python_png,
            "compare_png": compare_png,
        }
    }
    return result


# ==============================
# 主流程
# ==============================
def main():
    print("🚀 启动验证：AI 8点支撑（N） + 双方案 2-1 定位敏感性分析")
    print("=" * 72)

    # Step 1: 固定同一组 AI 支撑点
    env, support_points, full_constraint_python_mm = infer_ai_support_layout()
    format_points("AI 支撑点（N）", support_points)

    ensure_clean_workdir(WORK_DIR)

    stop_log = threading.Event()
    mapdl = launch_mapdl(
        run_location=WORK_DIR,
        jobname=JOB_NAME,
        nproc=1,
        additional_switches="-smp",
        override=True
    )

    t = threading.Thread(target=tail_ansys_log, args=(stop_log, WORK_DIR, JOB_NAME))
    t.daemon = True
    t.start()

    results = []

    try:
        schemes = ["x2_y1", "y2_x1"]

        for scheme_name in schemes:
            result = run_one_scheme(env, mapdl, support_points, scheme_name)
            results.append(result)

        # ──────────────────────────────────────────────────────────────────
        # 汇总对比表
        # ──────────────────────────────────────────────────────────────────
        print("\n" + "=" * 72)
        print("📊 两组 2-1 定位方案对比")
        print("=" * 72)
        print(f"基准（全向刚约束 Python） Max |UZ| : {full_constraint_python_mm:.6f} mm")
        print("  ↑ 该值仅含 UZ，用于与支撑+2-1 方案的 UZ 结果对比")
        print("-" * 72)

        for r in results:
            meta = r["locator_meta"]
            two_dof = meta["two_dof"]
            one_dof = meta["one_dof"]
            print(f"方案: {r['scheme']}")
            print(f"  约束配置: 2点({meta['two_edge_primary']})={two_dof}=0"
                  f"  |  1点({meta['one_edge_primary']})={one_dof}=0"
                  f"  |  支撑patch=UZ=0")
            print(f"  支撑 patch 节点数: {r['support_patch_sizes']}")
            print()
            # ── Python 端 ──
            print(f"  [Python]  Max |UX| = {r['python_max_abs_ux_mm']:.6f} mm"
                  + ("  (被2点约束↓)" if two_dof == "UX" else
                     ("  (被1点约束↓)" if one_dof == "UX" else "  (平面内自由)")))
            print(f"  [Python]  Max |UY| = {r['python_max_abs_uy_mm']:.6f} mm"
                  + ("  (被2点约束↓)" if two_dof == "UY" else
                     ("  (被1点约束↓)" if one_dof == "UY" else "  (平面内自由)")))
            print(f"  [Python]  Max |UZ| = {r['python_max_abs_uz_mm']:.6f} mm  (被支撑约束↓)")
            print(f"  [Python]  Max |U|  = {r['python_max_usum_mm']:.6f} mm")
            # ── ANSYS 端 ──
            print(f"  [ANSYS ]  Max |UX| = {r['ansys_max_abs_ux_mm']:.6f} mm"
                  + ("  (被2点约束↓)" if two_dof == "UX" else
                     ("  (被1点约束↓)" if one_dof == "UX" else "  (平面内自由)")))
            print(f"  [ANSYS ]  Max |UY| = {r['ansys_max_abs_uy_mm']:.6f} mm"
                  + ("  (被2点约束↓)" if two_dof == "UY" else
                     ("  (被1点约束↓)" if one_dof == "UY" else "  (平面内自由)")))
            print(f"  [ANSYS ]  Max |UZ| = {r['ansys_max_abs_uz_mm']:.6f} mm  (被支撑约束↓)")
            print(f"  [ANSYS ]  Max |U|  = {r['ansys_max_usum_mm']:.6f} mm")
            print()
            print(f"  输出图像:")
            for fval in r["files"].values():
                print(f"    - {fval}")
            print("-" * 72)

        if len(results) == 2:
            r0, r1 = results
            duz_a  = r1["ansys_max_abs_uz_mm"]  - r0["ansys_max_abs_uz_mm"]
            dux_a  = r1["ansys_max_abs_ux_mm"]  - r0["ansys_max_abs_ux_mm"]
            duy_a  = r1["ansys_max_abs_uy_mm"]  - r0["ansys_max_abs_uy_mm"]
            du_a   = r1["ansys_max_usum_mm"]     - r0["ansys_max_usum_mm"]
            duz_p  = r1["python_max_abs_uz_mm"] - r0["python_max_abs_uz_mm"]
            dux_p  = r1["python_max_abs_ux_mm"] - r0["python_max_abs_ux_mm"]
            duy_p  = r1["python_max_abs_uy_mm"] - r0["python_max_abs_uy_mm"]
            du_p   = r1["python_max_usum_mm"]   - r0["python_max_usum_mm"]

            print(f"📈 两方案差异  (y2_x1 − x2_y1):")
            print(f"  {'指标':<22}  {'ANSYS':>12}  {'Python':>12}  说明")
            print(f"  {'-'*22}  {'-'*12}  {'-'*12}  {'-'*24}")
            print(f"  {'Δ Max |UX| (mm)':<22}  {dux_a:>+12.6f}  {dux_p:>+12.6f}"
                  f"  {'← y2_x1 中 UX 被2点压制' if abs(dux_a) > abs(duy_a) else '← x2_y1 中 UX 被1点压制'}")
            print(f"  {'Δ Max |UY| (mm)':<22}  {duy_a:>+12.6f}  {duy_p:>+12.6f}"
                  f"  {'← y2_x1 中 UY 被1点压制' if abs(duy_a) < abs(dux_a) else '← x2_y1 中 UY 被2点压制'}")
            print(f"  {'Δ Max |UZ| (mm)':<22}  {duz_a:>+12.6f}  {duz_p:>+12.6f}"
                  f"  ← 支撑配置相同，UZ 应几乎不变")
            print(f"  {'Δ Max |U|  (mm)':<22}  {du_a:>+12.6f}  {du_p:>+12.6f}"
                  f"  ← U 微变量由平面内自由分量贡献")
            print("-" * 72)
            print("💡 物理解释:")
            print(f"   x2_y1: 2点约束 UY（沿Y方向压紧）→ UY 被抑制，UX 相对自由")
            print(f"   y2_x1: 2点约束 UX（沿X方向压紧）→ UX 被抑制，UY 相对自由")
            print(f"   UZ 由支撑点决定（两方案支撑相同）→ Δ|UZ| ≈ 0 符合预期")
            print(f"   |U| 的微小差异即来源于 UX/UY 自由分量在两方案间的交换")

    except Exception as e:
        print(f"\n❌ 验证过程出错: {e}")
        raise
    finally:
        stop_log.set()
        try:
            mapdl.exit()
        except Exception:
            pass


if __name__ == "__main__":
    main()