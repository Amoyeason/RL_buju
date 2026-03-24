import numpy as np
import pyvista as pv
import os

# ================= 配置 =================
DATA_DIR = "E:\\ansys_data_final"
FILE_PATH = os.path.join(DATA_DIR, "digital_twin_data.npz")


def main():
    print(f"📂 读取数据: {FILE_PATH} ...")
    if not os.path.exists(FILE_PATH):
        print("❌ 文件不存在！")
        return

    data = np.load(FILE_PATH)
    nodes = data['nodes']

    print(f"✅ 加载成功！节点数: {len(nodes)}")
    print("🎨 启动 PyVista 专业查看器...")

    # 1. 创建点云
    cloud = pv.PolyData(nodes)

    # 2. 启动绘图器
    pl = pv.Plotter()

    # 添加点云，按 Z 轴高度着色
    # point_size=5 保证点够大，render_points_as_spheres 让显示更圆润
    pl.add_mesh(cloud,
                scalars=nodes[:, 2],
                cmap="jet",
                point_size=5,
                render_points_as_spheres=True,
                show_scalar_bar=True,
                scalar_bar_args={'title': 'Z Height (m)'})

    # 添加坐标轴
    pl.add_axes()
    pl.show_grid()
    pl.add_text("3D Geometry Check\nUse Mouse to Rotate/Zoom", font_size=12)

    # 3. 设置相机位置 (俯视但带角度)
    pl.camera_position = 'iso'

    print("👉 窗口已弹出。像在 CAD 软件里一样操作鼠标即可。")
    pl.show()


if __name__ == "__main__":
    main()