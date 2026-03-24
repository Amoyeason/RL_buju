import numpy as np
from scipy import sparse
import os

DATA_DIR = "E:\\ansys_data_final"
FILE_PATH = os.path.join(DATA_DIR, "digital_twin_data.npz")
K_PATH = os.path.join(DATA_DIR, "K_pure.npz")


def inspect():
    print("🏥 启动数据体检程序...")

    # 1. 检查节点映射数据
    if not os.path.exists(FILE_PATH):
        print("❌ 找不到 digital_twin_data.npz")
        return

    data = np.load(FILE_PATH)
    print(f"\n📂 加载 digital_twin_data.npz:")
    print(f"   - Nodes shape: {data['nodes'].shape}")

    # 检查映射表
    if 'uz_map' in data:
        uz = data['uz_map']
        valid_count = np.sum(uz != -1)
        total_count = len(uz)
        print(f"   - UZ Map (Z轴映射): 总计 {total_count} 个节点")
        print(f"   - ❌ 有效映射数: {valid_count} (应该是几万个，如果是0则彻底损坏)")
        print(f"   - 样本: {uz[:10]} ...")
    else:
        print("   - ❌ 缺少 'uz_map' 字段！")

    # 2. 检查刚度矩阵
    if not os.path.exists(K_PATH):
        print("❌ 找不到 K_pure.npz")
        return

    try:
        K = sparse.load_npz(K_PATH)
        print(f"\n📂 加载 K_pure.npz:")
        print(f"   - K Matrix shape: {K.shape}")
        print(f"   - Non-zeros: {K.nnz}")
        if K.shape[0] == 0:
            print("   - ❌ 刚度矩阵为空 (0x0)！")
    except Exception as e:
        print(f"   - ❌ K矩阵读取失败: {e}")


if __name__ == "__main__":
    inspect()