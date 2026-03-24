import numpy as np
from ansys.mapdl.core import launch_mapdl
import os
import glob
from scipy import sparse
from scipy.sparse import kron, eye, coo_matrix

# ================= 配置区域 =================
WORK_DIR = "E:\\ansys_data_final"
if not os.path.exists(WORK_DIR): os.makedirs(WORK_DIR)

# 请确保此处指向正确的 5米长 CAD 文件
CAD_PATH = "E:\\ZJU\\Learning baogao\\0128.IGS"

# 物理参数
THICKNESS = 0.005  # 5mm
MESH_SIZE = 0.04  # 40mm
DENSITY = 1600  # T800 Density


def clean_directory():
    """清理旧文件"""
    extensions = ["*.full", "*.npz", "*.lock", "*.err", "*.log", "file.*", "*.mapping", "*.txt"]
    for ext in extensions:
        files = glob.glob(os.path.join(WORK_DIR, ext))
        for f in files:
            try:
                os.remove(f)
            except:
                pass


def parse_mapping_file(mapping_path):
    print(f"📖 解析映射文件: {os.path.basename(mapping_path)} ...")
    eq_map = {}
    with open(mapping_path, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3: continue
            try:
                if not parts[0].isdigit(): continue
                eq_id = int(parts[0]) - 1
                node_id = int(parts[1])
                dof_label = parts[2].strip().upper()
                eq_map[eq_id] = (node_id, dof_label)
            except:
                continue
    return eq_map


def main():
    clean_directory()
    print(f"🚀 [Step 1] 启动数据提取 (V42 无PCA纯净版)...")

    mapdl = launch_mapdl(run_location=WORK_DIR, nproc=1, additional_switches="-smp", override=True,
                         cleanup_on_exit=True)

    try:
        # ================= 1. 导入与缩放 =================
        print(f"🏗️ 导入几何: {CAD_PATH}")
        mapdl.clear();
        mapdl.aux15()
        mapdl.ioptn("IGES", "SMOOTH")
        mapdl.igesin(CAD_PATH)
        mapdl.finish();
        mapdl.prep7()

        print("📏 尺寸检查...")
        mapdl.allsel()
        # 获取包围盒
        xmin = mapdl.get_value("KP", 0, "MNLOC", "X");
        xmax = mapdl.get_value("KP", 0, "MXLOC", "X")
        ymin = mapdl.get_value("KP", 0, "MNLOC", "Y");
        ymax = mapdl.get_value("KP", 0, "MXLOC", "Y")
        zmin = mapdl.get_value("KP", 0, "MNLOC", "Z");
        zmax = mapdl.get_value("KP", 0, "MXLOC", "Z")

        dx, dy, dz = xmax - xmin, ymax - ymin, zmax - zmin
        max_dim = max(dx, dy, dz)
        print(f"   -> 包围盒: {dx:.2f} x {dy:.2f} x {dz:.2f}")

        if max_dim > 10.0:
            print("   ⚠️ 检测到 mm 单位，执行 x0.001 缩放...")
            mapdl.arscale("ALL", "", "", 0.001, 0.001, 0.001, "", "", 1)
            # 更新本地记录的尺寸
            dx *= 0.001;
            dy *= 0.001

        # 物理审计
        est_mass = (dx * dy * THICKNESS) * DENSITY
        print(f"   ⚖️ 估算质量: ~{est_mass:.2f} kg")

        # ================= 2. 物理定义 =================
        print("⚙️ 定义材料 (T800)...")
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
        mapdl.mp("DENS", 1, DENSITY)

        # ================= 3. 网格与求解 =================
        print(f"🕸️ 划分网格 (Size={MESH_SIZE})...")
        mapdl.esize(MESH_SIZE)
        mapdl.mshape(1, "2D");
        mapdl.mshkey(0)
        mapdl.shpp("OFF")
        mapdl.amesh("ALL")
        total_nodes = int(mapdl.mesh.n_node)
        print(f"   ✅ 节点数: {total_nodes}")

        print("🔧 组装矩阵...")
        mapdl.run("/SOLU");
        mapdl.antype("STATIC");
        mapdl.acel(0, 0, 9.8)
        mapdl.run("WRFULL, 1")
        mapdl.solve();
        mapdl.finish()

        # ================= 4. 提取与校验 =================
        print("📥 读取 .full 文件...")
        mm = mapdl.math
        k_mat = mm.stiff(fname="file.full")
        f_vec = mm.rhs(fname="file.full")

        K_raw = k_mat.asarray()
        F_raw = f_vec.asarray()

        # 🟢 修正：使用 Sum (L1) 而不是 Norm (L2) 来检查总力
        # Z方向力大约是 F_raw 中每隔6个数取一个的和，或者直接看绝对值总和估算
        f_sum_abs = np.sum(np.abs(F_raw))
        print(f"   🔎 载荷向量绝对值之和: {f_sum_abs:.2f} N")

        # 粗略判断：总和应该大于总重力 (因为分量叠加)
        expected_weight = est_mass * 9.8
        if f_sum_abs < expected_weight * 0.5:
            print(f"   ❌ 警告: 提取的力 ({f_sum_abs:.1f}) 远小于重力 ({expected_weight:.1f})！")
        else:
            print(f"   ✅ 物理校验通过 (力的大小合理)")

        # ================= 5. 映射与重排 =================
        print("🔄 处理映射与重排序...")
        mapdl.run("/AUX2")
        mapdl.file("file", "full")
        mapdl.run("HBMAT, export_job, txt, , ASCII, STIFF, YES, YES")

        map_files = glob.glob(os.path.join(WORK_DIR, "*.mapping"))
        if not map_files: map_files = glob.glob(os.path.join(WORK_DIR, "export_job.txt"))
        eq_map = parse_mapping_file(map_files[0])

        final_nnum = mapdl.mesh.nnum
        node_id_to_logical = {nid: i for i, nid in enumerate(final_nnum)}
        dof_order = {'UX': 0, 'UY': 1, 'UZ': 2, 'ROTX': 3, 'ROTY': 4, 'ROTZ': 5}

        rows, cols, vals = [], [], []
        target_dim = total_nodes * 6

        for eq_id, (nid, dof) in eq_map.items():
            if nid in node_id_to_logical and dof in dof_order:
                l_idx = node_id_to_logical[nid]
                dof_idx = dof_order[dof]
                col_idx = l_idx * 6 + dof_idx
                rows.append(eq_id)
                cols.append(col_idx)
                vals.append(1.0)

        P = coo_matrix((vals, (rows, cols)), shape=(K_raw.shape[0], target_dim)).tocsr()

        K_ordered = P.T @ K_raw @ P
        F_ordered = P.T @ F_raw

        # ================= 6. 保存 (无PCA) =================
        print("💾 保存数据 (保持原始 CAD 坐标系)...")

        # 🟢 关键：不再旋转节点，直接使用原始坐标
        # 这样 verify.py 里就不需要猜 PCA 参数了
        raw_nodes = mapdl.mesh.nodes

        ux_map = np.arange(0, total_nodes * 6, 6, dtype=np.int32)
        uy_map = np.arange(1, total_nodes * 6, 6, dtype=np.int32)
        uz_map = np.arange(2, total_nodes * 6, 6, dtype=np.int32)

        sparse.save_npz(os.path.join(WORK_DIR, "K_pure.npz"), K_ordered)
        np.savez(
            os.path.join(WORK_DIR, "digital_twin_data.npz"),
            F=F_ordered,
            nodes=raw_nodes,  # 直接存原始坐标
            ux_map=ux_map, uy_map=uy_map, uz_map=uz_map,
            n_ids=final_nnum
        )
        print("✅ 提取完成！")

    except Exception as e:
        print(f"❌ 失败: {e}")
    finally:
        if mapdl: mapdl.exit()


if __name__ == "__main__":
    main()