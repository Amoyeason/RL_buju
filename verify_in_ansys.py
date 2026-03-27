import os

# 解决 OpenMP 冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import shutil
import time
import threading
from ansys.mapdl.core import launch_mapdl
from sb3_contrib import MaskablePPO
from smart_fixture_env import SmartFixtureEnv3D
from scipy.spatial import KDTree
from constraint_n21 import map_support_patches_to_ansys_nodes, map_locator_to_ansys_node

# ================= 配置区域 =================
DATA_DIR = "E:\\ansys_data_final"
CAD_PATH = "E:\\ZJU\\Learning baogao\\0128.IGS"
MODEL_PATH = "models_3d/final_model_3d"
TARGET_N = 8

# 必须与 extract_data 一致
THICKNESS = 0.005
MESH_SIZE = 0.04
SMART_SIZE = None

WORK_DIR = "ansys_temp_verify"
JOB_NAME = "verify_job"


# ================= 日志监控 =================
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


# ================= 主程序 =================
def main():
    print("🚀 启动 AI 最终验证 (严格复刻 Manual 逻辑)...")

    if not os.path.exists(MODEL_PATH + ".zip"):
        print("❌ 错误：找不到模型文件！")
        return

    # 1. AI 规划
    print("\n🤖 [Step 1] AI 正在推理布局...")
    env = SmartFixtureEnv3D(data_dir=DATA_DIR, target_n=TARGET_N, constraint_mode="n21")
    model = MaskablePPO.load(MODEL_PATH)

    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
        obs, _, done, _, _ = env.step(action)

    ai_python_def = env.last_max_def
    ai_coords = np.array(env.fixtures)
    print(f"   📘 AI (Python) 预测变形: {ai_python_def:.6f} mm")
    print(f"   📍 AI 选择了 {len(ai_coords)} 个支撑点")

    # 2. 启动 ANSYS
    if os.path.exists(WORK_DIR): shutil.rmtree(WORK_DIR)
    os.makedirs(WORK_DIR, exist_ok=True)

    stop_log = threading.Event()
    mapdl = launch_mapdl(run_location=WORK_DIR, jobname=JOB_NAME, nproc=1, additional_switches="-smp", override=True)
    t = threading.Thread(target=tail_ansys_log, args=(stop_log,))
    t.daemon = True;
    t.start()

    try:
        # --- A. 重建 ---
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

        # 缩放
        mapdl.allsel()
        xmin = mapdl.get_value("KP", 0, "MNLOC", "X");
        xmax = mapdl.get_value("KP", 0, "MXLOC", "X")
        if (xmax - xmin) > 10.0:
            print("   ⚠️ 执行单位缩放 (x0.001)...")
            mapdl.arscale("ALL", "", "", 0.001, 0.001, 0.001, "", "", 1)

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

        print(f"   🕸️ 划分网格 (Size={MESH_SIZE})...")
        mapdl.esize(MESH_SIZE);
        mapdl.mshape(1, "2D");
        mapdl.mshkey(0)
        mapdl.shpp("OFF", "ALL")
        mapdl.amesh("ALL")

        num_nodes = int(mapdl.mesh.n_node)
        print(f"   ✅ 网格就绪: {num_nodes} nodes")

        # --- B. 映射 ---
        print("   🔍 映射 AI 布局点...")
        mapdl.run("/SOLU");
        mapdl.antype("STATIC");
        mapdl.acel(0, 0, 9.8)
        mapdl.ddele("ALL", "ALL")

        if env.constraint_mode == "n21":
            print(f"   🔒 施加 N-2-1 物理约束 ({env.locator_scheme})...")
            # 1. 施加 N 点面域支撑 (UZ=0)
            patch_ids_list, _, sizes = map_support_patches_to_ansys_nodes(mapdl, ai_coords, env.support_radius)
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
            for coord in ai_coords:
                dist, idx = tree.query(coord)
                target_ids.append(current_nnum[idx])
            for nid in target_ids: mapdl.d(nid, "ALL", 0)

        # 🟢 [关键复刻] 绘制并保存第一张图：约束位置图 (ISO视角)
        # 这就是您提到的“少输出的一张图”
        mapdl.run("/SHOW, PNG");
        mapdl.run("/GFILE, 800")
        mapdl.run("/VUP, 1, Z");
        mapdl.run("/VIEW, 1, 1, -1.73, 1")  # 手动设定 ISO 视角
        mapdl.run("/PBC, U, , 1");
        mapdl.run("EPLOT");
        mapdl.run("/SHOW, CLOSE")

        # 自动保存为 final_ai_constraints.png
        if os.path.exists(os.path.join(WORK_DIR, "verify_job000.png")):
            shutil.copy(os.path.join(WORK_DIR, "verify_job000.png"), "final_ai_constraints.png")
            print("   📸 已生成约束图: final_ai_constraints.png")

        # --- C. 求解 ---
        print("   🧮 求解中 (NLGEOM=OFF)...")
        # 🟢 [关键复刻] 保持与 Manual 一致的线性求解
        mapdl.nlgeom("OFF")
        mapdl.ncnv(2)
        mapdl.eqslv("SPARSE")
        mapdl.solve()
        mapdl.finish()

        # --- D. 结果 ---
        mapdl.post1();
        mapdl.set("LAST")
        try:
            disp_array = mapdl.post_processing.nodal_displacement("Z")
            ansys_max_mm = np.max(np.abs(disp_array)) * 1000.0
        except:
            mapdl.nsort("U", "Z")
            ansys_max_mm = mapdl.get_value("SORT", 0, "MAX") * 1000.0

        print("\n" + "=" * 40)
        print(f"📊 AI 最终验证结果 (Strict Manual Match)")
        print("=" * 40)
        print(f"   📘 Python (AI): {ai_python_def:.6f} mm")
        print(f"   🟧 ANSYS (Real): {ansys_max_mm:.6f} mm")

        # 硬编码人工基准值 (0.8054 mm)
        manual_baseline = 0.8054
        improvement = (manual_baseline - ansys_max_mm) / manual_baseline * 100
        print("-" * 20)
        print(f"   🆚 对比人工基准 (0.8054 mm):")
        print(f"   📈 提升幅度: {improvement:.2f}%")

        # ================= 绘图 =================
        print("   🎨 正在生成结果云图...")
        mapdl.run("/SHOW, PNG")
        mapdl.run("/GFILE, 1200")
        mapdl.run("/RGB,INDEX,100,100,100,0")
        mapdl.run("/RGB,INDEX,0,0,0,15")

        # 🟢 [关键复刻] 放大倍数与 Manual 一致
        mapdl.run("/DSCALE, 1, 10")

        # 🟢 [关键复刻] 不重置视角，直接继承上面的 ISO 视角
        # 这确保了图2和图1的视角是完全对齐的
        mapdl.run("PLNSOL, U, Z")

        mapdl.run("/SHOW, CLOSE")

        if os.path.exists(os.path.join(WORK_DIR, "verify_job001.png")):
            shutil.copy(os.path.join(WORK_DIR, "verify_job001.png"), "final_ai_result.png")
            print("   ✅ 已生成结果图: final_ai_result.png")

    except Exception as e:
        print(f"❌ 运行出错: {e}")
    finally:
        stop_log.set()
        if mapdl: mapdl.exit()


if __name__ == "__main__":
    main()