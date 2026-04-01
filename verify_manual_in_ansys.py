import os

# 解决 OpenMP 冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import shutil
import time
import threading
from ansys.mapdl.core import launch_mapdl
from smart_fixture_env import SmartFixtureEnv3D
from scipy.spatial import KDTree
from constraint_n21 import map_support_patches_to_ansys_nodes, map_locator_to_ansys_node

# ================= 配置区域 =================
DATA_DIR = "E:\\ansys_data_final"
CAD_PATH = "E:\\ZJU\\Learning baogao\\0128.IGS"
TARGET_N = 8

# 必须与 extract_data 一致
THICKNESS = 0.005
MESH_SIZE = 0.04
SMART_SIZE = None

WORK_DIR = "ansys_temp_manual_verify"
JOB_NAME = "manual_verify_job"


# ================= 辅助函数 =================
def tail_ansys_log(stop_event):
    log_path = os.path.join(WORK_DIR, f"{JOB_NAME}.out")
    while not os.path.exists(log_path):
        if stop_event.is_set(): return
        time.sleep(0.1)

    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        f.seek(0, 2)
        while not stop_event.is_set():
            line = f.readline()
            if line:
                if "error" in line.lower() or "warning" in line.lower():
                    print(f"   [ANSYS] {line.strip()}")
            else:
                time.sleep(0.1)


def get_smart_manual_baseline(env):
    """
    完全复刻 evaluate.py 中的人工基准生成逻辑
    """
    candidates = env.candidates
    tree = KDTree(candidates[:, :2])  # 2D 搜索

    x_min, x_max = candidates[:, 0].min(), candidates[:, 0].max()
    y_min, y_max = candidates[:, 1].min(), candidates[:, 1].max()
    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2

    # 理想布点：4角 + 4边中点 (N-2-1)
    ideal_points = [
        [x_min, y_min], [x_max, y_min], [x_min, y_max], [x_max, y_max],
        [x_mid, y_min], [x_mid, y_max],
        [x_min, y_mid], [x_max, y_mid]
    ]

    manual_fixtures = []
    for pt in ideal_points:
        dist, idx = tree.query(pt)
        manual_fixtures.append(tuple(candidates[idx]))

    # 去重补齐
    unique_fixtures = list(set(manual_fixtures))
    if len(unique_fixtures) < TARGET_N:
        needed = TARGET_N - len(unique_fixtures)
        indices = np.linspace(0, len(candidates) - 1, needed, dtype=int)
        for i in indices:
            cand = tuple(candidates[i])
            if cand not in unique_fixtures:
                unique_fixtures.append(cand)
            if len(unique_fixtures) == TARGET_N: break

    return unique_fixtures[:TARGET_N]


# ================= 主程序 =================
def main():
    print("🚀 启动人工基准物理验证 (Manual Baseline Verify)...")

    # 1. 初始化 Python 环境并计算预测值
    print("\n🤖 [Step 1] Python 环境计算...")
    env = SmartFixtureEnv3D(data_dir=DATA_DIR, target_n=TARGET_N, constraint_mode="n21")

    # 获取人工基准点
    manual_fixtures = get_smart_manual_baseline(env)
    print(f"   📍 已生成 {len(manual_fixtures)} 个人工基准点")

    # Python 求解
    if env.constraint_mode == "n21":
        py_solution = env._solve_mdsm(manual_fixtures, return_metrics=True)
        py_max_def = py_solution["max_abs_uz_mm"]
        env.last_locator_meta = py_solution["locator_meta"]
        env.last_locator2_points = py_solution["locator2_points"]
        env.last_locator1_point = py_solution["locator1_point"]
    else:
        uz_py = env._solve_mdsm(manual_fixtures)
        py_max_def = np.max(np.abs(uz_py)) * 1000.0
    print(f"   📘 Python 预测变形: {py_max_def:.6f} mm")

    # 2. 启动 ANSYS 进行真实物理验证
    if os.path.exists(WORK_DIR): shutil.rmtree(WORK_DIR)
    os.makedirs(WORK_DIR, exist_ok=True)

    stop_log = threading.Event()
    mapdl = launch_mapdl(run_location=WORK_DIR, jobname=JOB_NAME, nproc=1, additional_switches="-smp", override=True)
    t = threading.Thread(target=tail_ansys_log, args=(stop_log,))
    t.daemon = True;
    t.start()

    try:
        # --- A. 重建物理环境 ---
        print("\n🏭 [Step 2] ANSYS 重建环境 (SMOOTH + NoPCA)...")
        mapdl.clear();
        mapdl.aux15()
        mapdl.ioptn("GTOL", 0.05);
        mapdl.ioptn("IGES", "SMOOTH")
        mapdl.ioptn("MERG", "YES");
        mapdl.ioptn("SOLID", "NO")
        mapdl.igesin(CAD_PATH);
        mapdl.finish();
        mapdl.prep7()

        # 缩放 (与 extract_data 保持一致)
        mapdl.allsel()
        xmin = mapdl.get_value("KP", 0, "MNLOC", "X");
        xmax = mapdl.get_value("KP", 0, "MXLOC", "X")
        if (xmax - xmin) > 10.0:
            print("   ⚠️ 执行单位缩放 (x0.001)...")
            mapdl.arscale("ALL", "", "", 0.001, 0.001, 0.001, "", "", 1)

        # 材料属性
        mapdl.et(1, "SHELL181");
        mapdl.sectype(1, "SHELL");
        mapdl.secdata(THICKNESS)
        mapdl.mp("EX", 1, 157.7e9);
        mapdl.mp("EY", 1, 9.05e9);
        mapdl.mp("EZ", 1, 9.05e9)
        mapdl.mp("GXY", 1, 4.69e9);
        mapdl.mp("GXZ", 1, 4.69e9);
        mapdl.mp("GYZ", 1, 3.24e9)
        mapdl.mp("PRXY", 1, 0.3);
        mapdl.mp("DENS", 1, 1600)

        # 网格划分 (必须与 extract_data 一致)
        print(f"   🕸️ 划分网格 (Size={MESH_SIZE})...")
        mapdl.esize(MESH_SIZE);
        mapdl.mshape(1, "2D");
        mapdl.mshkey(0)
        mapdl.shpp("OFF", "ALL")
        mapdl.amesh("ALL")

        num_nodes = int(mapdl.mesh.n_node)
        print(f"   ✅ 网格就绪: {num_nodes} nodes (Extract时为 {env.n_nodes})")

        # --- B. 映射施加约束 (直接映射，无旋转) ---
        print("   🔍 映射人工基准点...")
        # 直接使用 ANSYS 原始节点 (因为 extract_data 存的就是原始坐标)
        print(f"   🔒 施加约束 ({len(manual_fixtures)} 点)...")
        mapdl.run("/SOLU");
        mapdl.antype("STATIC");
        mapdl.acel(0, 0, 9.8)
        mapdl.ddele("ALL", "ALL")

        if env.constraint_mode == "n21":
            print(f"   🔒 施加 N-2-1 物理约束 ({env.locator_scheme})...")
            # 1. 施加 N 点面域支撑 (UZ=0)
            patch_ids_list, _, sizes = map_support_patches_to_ansys_nodes(mapdl, manual_fixtures, env.support_radius)
            total_sup_nodes = sum(sizes)
            for ids in patch_ids_list:
                for nid in ids:
                    mapdl.d(nid, "UZ", 0)

            # 2. 施加 2点定位
            if hasattr(env, "last_locator2_points") and env.last_locator2_points is not None:
                two_dof = env.last_locator_meta["two_dof"]
                for p2 in env.last_locator2_points:
                    nid2, _ = map_locator_to_ansys_node(mapdl, p2, env.locator_clearance)
                    if nid2 is not None:
                        mapdl.d(nid2, two_dof, 0)

            # 3. 施加 1点定位
            if hasattr(env, "last_locator1_point") and env.last_locator1_point is not None:
                one_dof = env.last_locator_meta["one_dof"]
                nid1, _ = map_locator_to_ansys_node(mapdl, env.last_locator1_point, env.locator_clearance)
                if nid1 is not None:
                    mapdl.d(nid1, one_dof, 0)
        else:
            print("   🔒 施加完全刚性约束...")
            current_nodes = mapdl.mesh.nodes
            current_nnum = mapdl.mesh.nnum
            tree = KDTree(current_nodes)
            target_ids = []
            for coord in manual_fixtures:
                dist, idx = tree.query(coord)
                target_ids.append(current_nnum[idx])
            for nid in target_ids: mapdl.d(nid, "ALL", 0)

        # 保存约束图以便检查位置是否正确
        mapdl.run("/SHOW, PNG");
        mapdl.run("/GFILE, 800")
        mapdl.run("/VUP, 1, Z");
        mapdl.run("/VIEW, 1, 1, -1.73, 1")
        mapdl.run("/PBC, U, , 1");
        mapdl.run("EPLOT");
        mapdl.run("/SHOW, CLOSE")
        if os.path.exists(os.path.join(WORK_DIR, "manual_verify_job000.png")):
            shutil.copy(os.path.join(WORK_DIR, "manual_verify_job000.png"), "debug_manual_constraints.png")

        # --- C. 求解 ---
        print("   🧮 求解中 (NLGEOM=OFF)...")
        mapdl.nlgeom("OFF");
        mapdl.ncnv(2);
        mapdl.eqslv("SPARSE")
        mapdl.solve()
        mapdl.finish()

        # --- D. 结果对比 ---
        mapdl.post1();
        mapdl.set("LAST")

        # 获取最大变形
        try:
            disp_array = mapdl.post_processing.nodal_displacement("Z")
            ansys_max_mm = np.max(np.abs(disp_array)) * 1000.0
        except:
            mapdl.nsort("U", "Z")
            ansys_max_mm = mapdl.get_value("SORT", 0, "MAX") * 1000.0

        print("\n" + "=" * 40)
        print(f"📊 人工基准验证结果 (Manual Baseline Verification)")
        print("=" * 40)
        print(f"   📘 Python 预测: {py_max_def:.6f} mm")
        print(f"   🟧 ANSYS 真实:  {ansys_max_mm:.6f} mm")

        if ansys_max_mm > 1e-9:
            ratio = ansys_max_mm / py_max_def
            print(f"   ⚖️ 吻合度: {ratio:.4f}")
            if 0.8 < ratio < 1.2:
                print("   ✅ 完美！物理引擎完全可信。")
            else:
                print("   ⚠️ 存在偏差。")
                print("   (如果偏差很大，可能是 Python 里的刚度矩阵 K 单位还是有问题，或者 ANSYS 求解器非线性效应太强)")

        # 绘图
        mapdl.run("/SHOW, PNG");
        mapdl.run("/GFILE, 1200")
        mapdl.run("/RGB,INDEX,100,100,100,0");
        mapdl.run("/RGB,INDEX,0,0,0,15")
        mapdl.run("/DSCALE, 1, 10")
        mapdl.run("PLNSOL, U, Z");
        mapdl.run("/SHOW, CLOSE")
        if os.path.exists(os.path.join(WORK_DIR, "manual_verify_job001.png")):
            shutil.copy(os.path.join(WORK_DIR, "manual_verify_job001.png"), "manual_ansys_result.png")
            print("   ✅ 云图已生成: manual_ansys_result.png")

    except Exception as e:
        print(f"❌ 运行出错: {e}")
    finally:
        stop_log.set()
        if mapdl: mapdl.exit()


if __name__ == "__main__":
    main()