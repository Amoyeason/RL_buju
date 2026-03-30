import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.spatial import KDTree


def select_21_locator_points(env, support_points, scheme="x2_y1", clearance=0.12, boundary_tol=0.032):
    """
    根据当前的支撑点，以及选定的 scheme（"x2_y1" 或 "y2_x1"），自动分配 2-1 定位点。
    返回:
        locator2_points (list): 2点定位的坐标列表 [(x,y,z), (x,y,z)]
        locator1_point (tuple): 1点定位的坐标 (x,y,z)
        locator_meta (dict): 包含分配详情的字典，供后续约束使用
    """
    nodes = np.asarray(env.nodes, dtype=float)
    support_xy = np.asarray(support_points)[:, :2]
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
        for tol in [boundary_tol, boundary_tol * 1.5, boundary_tol * 2.0]:
            for clr in [clearance, clearance * 0.7, clearance * 0.4]:
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

    locator2_points = [tuple(p) for p in nodes[locator2_idx]]
    locator1_point = tuple(nodes[locator1_idx])

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

    # print(f"   ✅ 2点定位边: {two_edge_primary} (fallback={two_edge_fallback}), DOF={two_dof}")
    # print(f"   ✅ 1点定位边: {one_edge_primary} (fallback={one_edge_fallback}), DOF={one_dof}")
    return locator2_points, locator1_point, meta

def map_support_patches_to_ansys_nodes(mapdl, support_points, support_radius=0.05):
    """
    ANSYS 映射支持函数，映射 N 个支撑点到网格节点。
    """
    current_nodes = np.asarray(mapdl.mesh.nodes, dtype=float)
    if len(current_nodes) == 0:
        return [], [], []

    tree = KDTree(current_nodes)
    node_numbers = mapdl.mesh.nnum

    patch_ids_list = []
    patch_coords_list = []
    patch_sizes = []

    for sp in support_points:
        indices = tree.query_ball_point(sp, support_radius)
        if len(indices) == 0:
            patch_ids_list.append([])
            patch_coords_list.append([])
            patch_sizes.append(0)
            continue
        mapped_ids = [node_numbers[idx] for idx in indices]
        mapped_coords = [current_nodes[idx] for idx in indices]
        patch_ids_list.append(mapped_ids)
        patch_coords_list.append(mapped_coords)
        patch_sizes.append(len(mapped_ids))

    return patch_ids_list, patch_coords_list, patch_sizes


def map_locator_to_ansys_node(mapdl, loc_point, clearance=0.12):
    """
    ANSYS 映射定位点函数，映射单个侧边定位点。
    """
    current_nodes = np.asarray(mapdl.mesh.nodes, dtype=float)
    if len(current_nodes) == 0:
        return None, None

    dists = np.linalg.norm(current_nodes[:, :2] - np.asarray(loc_point)[:2], axis=1)
    mask_z = np.abs(current_nodes[:, 2] - loc_point[2]) <= clearance
    valid_idx = np.where(mask_z)[0]
    if len(valid_idx) == 0:
        best_idx = np.argmin(dists)
    else:
        best_idx = valid_idx[np.argmin(dists[valid_idx])]

    return mapdl.mesh.nnum[best_idx], current_nodes[best_idx]


def find_nearest_node_for_locator(env, point_3d, clearance=0.12):
    """ 为定位点寻找最近节点，施加侧面定位约束 """
    nodes = env.nodes
    dists = np.linalg.norm(nodes[:, :2] - np.asarray(point_3d)[:2], axis=1)
    mask_z = np.abs(nodes[:, 2] - point_3d[2]) <= clearance
    valid_idx = np.where(mask_z)[0]
    if len(valid_idx) == 0:
        return np.argmin(dists)
    return valid_idx[np.argmin(dists[valid_idx])]

def solve_mdsm_n21(env, support_points, locator2_points, locator1_point, locator_meta, penalty=1e15, support_radius=0.05, locator_clearance=0.12):
    """
    通用 N-2-1 约束 Python 求解器。
    该函数可以在主强化学习环境和其他 Python 验证脚本中共享。
    """
    ndof = env.K_base.shape[0]
    penalty_vec = np.zeros(ndof, dtype=np.float64)
    # A. N 类支撑点：UZ patch
    support_indices_list = env.fem_tree_3d.query_ball_point(np.asarray(support_points), support_radius)
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
        # _, node_idx = env.fem_tree_3d.query(pt)
        node_idx = find_nearest_node_for_locator(env, pt, clearance=locator_clearance)
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
    # _, locator1_node_idx = env.fem_tree_3d.query(np.asarray(locator1_point))
    locator1_node_idx = find_nearest_node_for_locator(env, locator1_point, clearance=locator_clearance)
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

    try:
        U_vec = spsolve(K_mod, F_mod)
    except Exception as e:
        raise RuntimeError(f"solve_mdsm_n21 spsolve 失败: {e}") from e

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
        "locator_meta": locator_meta,
        "locator2_points": locator2_points,
        "locator1_point": locator1_point
    }

    return metrics

