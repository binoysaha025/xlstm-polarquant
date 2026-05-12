import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── FINAL RESULTS ─────────────────────────────────────────────────────
vanilla_val = [
    1.1226,0.8366,0.6501,0.6190,0.5982,0.4991,0.4593,0.3835,0.3422,0.3155,
    0.3136,0.3120,0.3075,0.2975,0.2986,0.3004,0.3047,0.2600,0.3054,0.2354,
    0.2725,0.2507,0.3060,0.2383,0.2737,0.2214,0.2416,0.2146,0.2502,0.1921,
    0.2451,0.2268,0.2033,0.2254,0.2036,0.2002,0.1759,0.1835,0.1647,0.1742,
    0.1790,0.1825,0.1928,0.1772,0.1925,0.1684,0.1852,0.1648,0.1747,0.1670,
    0.1704,0.1700,0.1640,0.1674,0.1650,0.1651,0.1652,0.1648,0.1647,0.1647,
    0.1646,0.1643,0.1645,0.1644,0.1643,0.1642,0.1641,0.1641,0.1641,0.1640,
    0.1639,0.1639,0.1638,0.1638,0.1637,0.1636,0.1635,0.1636,0.1634,0.1634,
    0.1633,0.1633,0.1632,0.1632,0.1631,0.1630,0.1630,0.1629,0.1629,0.1627,
    0.1627,0.1626,0.1626,0.1625,0.1624,0.1624,0.1624,0.1623,0.1622,0.1622
]

pq_val = [
    1.0533,0.5418,0.3880,0.3619,0.2934,0.2454,0.2343,0.2157,0.2039,0.2311,
    0.2280,0.1992,0.2341,0.2108,0.1868,0.2012,0.1967,0.2161,0.1798,0.1750,
    0.1973,0.1695,0.2018,0.1932,0.1742,0.1935,0.1903,0.2151,0.1551,0.1788,
    0.1780,0.1810,0.1738,0.1720,0.1793,0.2058,0.1757,0.1891,0.1754,0.1765,
    0.1782,0.1777,0.1680,0.1720,0.1687,0.1692,0.1688,0.1682,0.1652,0.1660,
    0.1643,0.1651,0.1657,0.1645,0.1650,0.1643,0.1651,0.1646,0.1652,0.1647,
    0.1633,0.1639,0.1631,0.1635,0.1642,0.1625,0.1636,0.1631,0.1628,0.1628,
    0.1645,0.1637,0.1647,0.1637,0.1632,0.1630,0.1626,0.1627,0.1634,0.1641,
    0.1634,0.1648,0.1627,0.1643,0.1634,0.1638,0.1641,0.1632,0.1635,0.1624,
    0.1633,0.1623,0.1633,0.1627,0.1639,0.1637,0.1640,0.1631,0.1630,0.1642
]

transformer_val = [
    0.6273,0.2282,0.3323,0.2832,0.2806,0.2471,0.2297,0.2115,0.1932,0.1811,
    0.1700,0.1667,0.1454,0.1513,0.1493,0.1256,0.1353,0.1416,0.1316,0.1162,
    0.1333,0.1567,0.1348,0.1424,0.1443,0.1262,0.1501,0.1497,0.1265,0.1312,
    0.1364,0.1371,0.1275,0.1289,0.1243,0.1257,0.1261,0.1157,0.1188,0.1181,
    0.1192,0.1122,0.1155,0.1148,0.1140,0.1109,0.1147,0.1071,0.1099,0.1101,
    0.1064,0.1087,0.1109,0.1059,0.1047,0.1077,0.1029,0.1042,0.1020,0.1038,
    0.0993,0.1026,0.1046,0.1019,0.1010,0.0990,0.1014,0.0988,0.0963,0.0963,
    0.0996,0.0970,0.1005,0.0959,0.0963,0.0986,0.0930,0.0971,0.0933,0.0971,
    0.0959,0.0983,0.0930,0.0943,0.0941,0.0929,0.0959,0.0935,0.0930,0.0959,
    0.0915,0.0939,0.0940,0.0908,0.0940,0.0944,0.0923,0.0916,0.0996,0.0876
]


# ── MEMORY BENCHMARK ──────────────────────────────────────────────────
def memory_benchmark():
    hidden_sizes = [16, 32, 64, 128, 256, 512]
    vanilla_mem, pq_mem = [], []
    for h in hidden_sizes:
        vanilla_mem.append(h * h * 4 / 1024)   # float32, KB
        pq_mem.append(h * h * 1 / 1024)        # int8, KB
    return hidden_sizes, vanilla_mem, pq_mem


# ── THEORETICAL SCALING ───────────────────────────────────────────────
def theoretical_scaling():
    seq_lengths = np.array([32, 64, 128, 256, 512, 1024])
    transformer = seq_lengths ** 2 / (32 ** 2)   # normalized to 1 at seq=32
    xlstm = seq_lengths / 32                      # linear, normalized
    return seq_lengths, transformer, xlstm


# ── PLOT ──────────────────────────────────────────────────────────────
def plot_all():
    fig = plt.figure(figsize=(20, 5))
    gs = gridspec.GridSpec(1, 3, figure=fig)
    fig.suptitle('xLSTM + PolarQuant: Accuracy, Memory & Scaling', fontsize=14)

    # chart 1: loss curves
    ax1 = fig.add_subplot(gs[0])
    epochs = range(1, 101)
    ax1.plot(epochs, transformer_val, label='Transformer', color='crimson', linewidth=1.5)
    ax1.plot(epochs, vanilla_val, label='vanilla xLSTM', color='steelblue', linewidth=1.5)
    ax1.plot(epochs, pq_val, label='xLSTM + PolarQuant (4x compressed)',
             color='darkorange', linewidth=1.5)
    ax1.set_title('Validation Loss (AAPL 2022-2024)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.set_ylim(0, 0.5)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # final values annotation
    ax1.annotate(f'0.0876', xy=(100, 0.0876), fontsize=7, color='crimson')
    ax1.annotate(f'0.1622', xy=(100, 0.1622), fontsize=7, color='steelblue')
    ax1.annotate(f'0.1642', xy=(100, 0.1642), fontsize=7, color='darkorange')

    # chart 2: memory
    ax2 = fig.add_subplot(gs[1])
    hidden_sizes, vanilla_mem, pq_mem = memory_benchmark()
    ax2.plot(hidden_sizes, vanilla_mem, label='vanilla xLSTM (float32)',
             color='steelblue', marker='o')
    ax2.plot(hidden_sizes, pq_mem, label='xLSTM + PolarQuant (int8)',
             color='darkorange', marker='o')
    ax2.set_title('Matrix Memory C Size vs Hidden Size')
    ax2.set_xlabel('Hidden Size')
    ax2.set_ylabel('Memory (KB)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 4x annotation
    ax2.annotate('4x compression', xy=(256, 256), fontsize=9,
                 color='green', fontweight='bold')

    # chart 3: theoretical scaling
    ax3 = fig.add_subplot(gs[2])
    seq_lengths, transformer_ops, xlstm_ops = theoretical_scaling()
    ax3.plot(seq_lengths, transformer_ops, label='Transformer O(n²)',
             color='crimson', marker='o')
    ax3.plot(seq_lengths, xlstm_ops, label='xLSTM O(n)',
             color='steelblue', marker='o')
    ax3.set_title('Theoretical Inference Complexity\n(normalized, per Beck et al. 2024)')
    ax3.set_xlabel('Sequence Length')
    ax3.set_ylabel('Relative Compute Cost')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('benchmark/results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("saved to benchmark/results.png")
    print("\n=== FINAL RESULTS ===")
    print(f"Transformer        best val: {min(transformer_val):.4f}")
    print(f"vanilla xLSTM      best val: {min(vanilla_val):.4f}")
    print(f"xLSTM+PolarQuant   best val: {min(pq_val):.4f}  (4x memory compression)")


if __name__ == '__main__':
    plot_all()