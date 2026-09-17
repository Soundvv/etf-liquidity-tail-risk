"""
第一步: 数据预处理 + 描述性统计
=========================================================
输出:
    tables/
        01_data_quality.csv          数据质量检查 (缺失值/异常值)
        02_descriptive_statistics.csv 描述性统计 + 多项检验
        03_subperiod_stats.csv       分子时段统计 (4个市场阶段)
    figures/
        01_volatility_series_<ETF>.png  各 ETF 的 L_t / R_t 波动序列图
        02_qq_plots.png                 5 只 ETF 的 R_t Q-Q 图
        03_density_plots.png            R_t 与正态分布的密度对比
        04_correlation_heatmap.png      ETF 间相关性热图

输入: ready_*.csv (由 fetch_data.py 生成)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import het_arch

warnings.filterwarnings("ignore")

# ============================================================
# 全局配置
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TABLE_DIR = os.path.join(BASE_DIR, "tables")
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# 中文字体配置 (macOS)
mpl.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "PingFang HK", "STHeiti", "Arial Unicode MS"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["font.size"] = 10
mpl.rcParams["figure.dpi"] = 100
mpl.rcParams["savefig.dpi"] = 150
mpl.rcParams["savefig.bbox"] = "tight"

# 主样本: 5 只宽基 ETF (与上一届口径一致, 便于对比)
MAIN_ETFS = [
    ("510050", "上证50ETF"),
    ("510180", "上证180ETF"),
    ("510300", "沪深300ETF"),
    ("510500", "中证500ETF"),
    ("159915", "创业板ETF"),
]
# 案例 ETF
CASE_ETF = ("513050", "中概互联ETF")

# 关键事件 (用于图上标注)
EVENTS = [
    ("2015-06-12", "股灾起点"),
    ("2016-01-04", "熔断"),
    ("2018-03-22", "中美贸易战"),
    ("2020-01-23", "新冠疫情"),
    ("2022-03-14", "中概股危机"),
]

# 分子时段定义
SUBPERIODS = {
    "牛市2014-2015":     ("2014-07-01", "2015-06-12"),
    "股灾2015-2016":     ("2015-06-12", "2016-02-29"),
    "震荡2017-2019":     ("2017-01-01", "2019-12-31"),
    "疫情2020":          ("2020-01-23", "2020-12-31"),
    "中概危机2021-2022": ("2021-02-01", "2022-10-31"),
    "复苏2023-2025":     ("2023-01-01", "2025-12-31"),
}

# ============================================================
# 辅助函数
# ============================================================
def load_etf(code: str, name: str) -> pd.DataFrame:
    path = os.path.join(BASE_DIR, f"ready_{code}_{name}.csv")
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def safe_test(func, *args, **kwargs):
    """避免单个检验失败导致全表崩溃"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        return None


# ============================================================
# 1. 数据质量检查
# ============================================================
def data_quality_check(etfs):
    rows = []
    for code, name in etfs:
        df = load_etf(code, name)
        n_total = len(df)
        n_missing_L = df["L_t"].isna().sum()
        n_missing_R = df["R_t"].isna().sum()

        # 异常值: |z| > 5 (5 倍标准差以外)
        L_z = np.abs((df["L_t"] - df["L_t"].mean()) / df["L_t"].std(ddof=1))
        R_z = np.abs((df["R_t"] - df["R_t"].mean()) / df["R_t"].std(ddof=1))
        n_outlier_L = int((L_z > 5).sum())
        n_outlier_R = int((R_z > 5).sum())

        # 0 价格/成交量
        n_zero_vol = int((df["volume"] == 0).sum()) if "volume" in df else 0
        n_zero_range = int(((df["high"] - df["low"]) == 0).sum()) if "high" in df else 0

        rows.append({
            "ETF代码": code, "ETF名称": name,
            "总样本": n_total,
            "区间": f"{df['date'].min().date()} ~ {df['date'].max().date()}",
            "L_t缺失": n_missing_L, "R_t缺失": n_missing_R,
            "L_t异常(|z|>5)": n_outlier_L, "R_t异常(|z|>5)": n_outlier_R,
            "零成交量天数": n_zero_vol, "零价差天数": n_zero_range,
        })
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(TABLE_DIR, "01_data_quality.csv"), index=False, encoding="utf-8-sig")
    return out


# ============================================================
# 2. 描述性统计 + 统计检验
# ============================================================
def descriptive_table(etfs):
    rows = []
    for code, name in etfs:
        df = load_etf(code, name).dropna(subset=["L_t", "R_t"])
        for var_name, series in [("L_t", df["L_t"]), ("R_t", df["R_t"])]:
            x = series.values
            # 基本统计
            mean = np.mean(x)
            std = np.std(x, ddof=1)
            skew = stats.skew(x)
            kurt = stats.kurtosis(x, fisher=False)  # 不减3, 与文献一致

            # Jarque-Bera 正态性
            jb_stat, jb_p = stats.jarque_bera(x)

            # ADF 平稳性 (返回 statistic, p, ...)
            adf = safe_test(adfuller, x, autolag="AIC")
            adf_stat = adf[0] if adf is not None else np.nan
            adf_p = adf[1] if adf is not None else np.nan

            # ARCH-LM 检验 (滞后 5)
            arch = safe_test(het_arch, x, nlags=5)
            arch_stat = arch[0] if arch is not None else np.nan
            arch_p = arch[1] if arch is not None else np.nan

            # Ljung-Box (滞后 10)
            lb = safe_test(stats.diagnostic_tests if False else None)
            # 用 statsmodels.stats.diagnostic.acorr_ljungbox 更稳
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb_res = safe_test(acorr_ljungbox, x, lags=[10], return_df=True)
            if lb_res is not None:
                lb_stat = float(lb_res["lb_stat"].iloc[0])
                lb_p = float(lb_res["lb_pvalue"].iloc[0])
            else:
                lb_stat = np.nan; lb_p = np.nan

            rows.append({
                "ETF": f"{code} {name}",
                "变量": var_name,
                "样本数": len(x),
                "均值": round(mean, 6),
                "标准差": round(std, 6),
                "最小值": round(np.min(x), 6),
                "最大值": round(np.max(x), 6),
                "偏度": round(skew, 4),
                "峰度": round(kurt, 4),
                "JB统计量": round(jb_stat, 2),
                "JB p值": f"{jb_p:.4g}",
                "ADF统计量": round(adf_stat, 4),
                "ADF p值": f"{adf_p:.4g}",
                "ARCH-LM(5)": round(arch_stat, 2),
                "ARCH p值": f"{arch_p:.4g}",
                "LB(10)": round(lb_stat, 2),
                "LB p值": f"{lb_p:.4g}",
            })
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(TABLE_DIR, "02_descriptive_statistics.csv"), index=False, encoding="utf-8-sig")
    return out


# ============================================================
# 3. 分子时段统计
# ============================================================
def subperiod_stats(etfs):
    rows = []
    for code, name in etfs:
        df = load_etf(code, name).dropna(subset=["L_t", "R_t"])
        for period_name, (start, end) in SUBPERIODS.items():
            sub = df[(df["date"] >= start) & (df["date"] <= end)]
            if len(sub) < 30:
                continue
            rows.append({
                "ETF": f"{code} {name}",
                "时段": period_name,
                "天数": len(sub),
                "L_t均值": round(sub["L_t"].mean(), 4),
                "L_t标差": round(sub["L_t"].std(ddof=1), 4),
                "R_t均值": round(sub["R_t"].mean(), 6),
                "R_t标差": round(sub["R_t"].std(ddof=1), 6),
                "R_t偏度": round(sub["R_t"].skew(), 4),
                "R_t峰度": round(sub["R_t"].kurtosis() + 3, 4),  # pandas返回超额峰度
                "L-R Pearson相关": round(sub["L_t"].corr(sub["R_t"]), 4),
                "L-R Kendall相关": round(sub["L_t"].corr(sub["R_t"], method="kendall"), 4),
            })
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(TABLE_DIR, "03_subperiod_stats.csv"), index=False, encoding="utf-8-sig")
    return out


# ============================================================
# 4. 可视化
# ============================================================
def plot_volatility_series(etfs):
    """每只 ETF: 双 y 轴, 上图 L_t, 下图 R_t"""
    for code, name in etfs:
        df = load_etf(code, name).dropna(subset=["L_t", "R_t"])
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

        axes[0].plot(df["date"], df["L_t"], color="#1f77b4", linewidth=0.6)
        axes[0].set_ylabel("流动性风险 $L_t$", fontsize=11)
        axes[0].set_title(f"{name} ({code}) 流动性风险与市场风险波动序列", fontsize=12, fontweight="bold")
        axes[0].grid(alpha=0.3)

        axes[1].plot(df["date"], df["R_t"], color="#d62728", linewidth=0.5)
        axes[1].set_ylabel("市场风险 $R_t$", fontsize=11)
        axes[1].set_xlabel("日期", fontsize=11)
        axes[1].grid(alpha=0.3)
        axes[1].axhline(0, color="black", linewidth=0.5, linestyle="--")

        # 标注关键事件
        for date_str, label in EVENTS:
            d = pd.to_datetime(date_str)
            if df["date"].min() <= d <= df["date"].max():
                for ax in axes:
                    ax.axvline(d, color="gray", linewidth=0.6, linestyle=":", alpha=0.6)
                axes[0].annotate(label, xy=(d, axes[0].get_ylim()[1]),
                                 xytext=(0, -8), textcoords="offset points",
                                 fontsize=7, color="gray", ha="center", rotation=90, va="top")

        plt.tight_layout()
        path = os.path.join(FIG_DIR, f"01_volatility_series_{code}_{name}.png")
        plt.savefig(path)
        plt.close()


def plot_qq_panel(etfs):
    """5 只 ETF 的 R_t Q-Q 图 (2x3 网格)"""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for i, (code, name) in enumerate(etfs):
        ax = axes[i]
        df = load_etf(code, name).dropna(subset=["R_t"])
        stats.probplot(df["R_t"], dist="norm", plot=ax)
        ax.set_title(f"{name} 收益率 Q-Q 图", fontsize=11)
        ax.set_xlabel("理论分位数")
        ax.set_ylabel("样本分位数")
        ax.get_lines()[0].set_marker("o")
        ax.get_lines()[0].set_markerfacecolor("#1f77b4")
        ax.get_lines()[0].set_markeredgecolor("none")
        ax.get_lines()[0].set_markersize(3)
        ax.get_lines()[1].set_color("#d62728")
        ax.grid(alpha=0.3)
    # 删除多余子图
    for j in range(len(etfs), len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle("市场风险 $R_t$ 的 Q-Q 图 —— 显著偏离正态分布", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "02_qq_plots.png"))
    plt.close()


def plot_density_panel(etfs):
    """R_t 与正态分布密度对比"""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for i, (code, name) in enumerate(etfs):
        ax = axes[i]
        df = load_etf(code, name).dropna(subset=["R_t"])
        x = df["R_t"].values
        # 经验密度 (核密度)
        from scipy.stats import gaussian_kde, norm
        xs = np.linspace(x.min(), x.max(), 300)
        ax.hist(x, bins=80, density=True, alpha=0.4, color="#1f77b4", edgecolor="none", label="实证分布")
        kde = gaussian_kde(x)
        ax.plot(xs, kde(xs), color="#1f77b4", linewidth=1.5)
        # 同方差正态
        ax.plot(xs, norm.pdf(xs, loc=x.mean(), scale=x.std(ddof=1)),
                color="#d62728", linewidth=1.5, linestyle="--", label="正态拟合")
        ax.set_title(f"{name}  峰度={stats.kurtosis(x, fisher=False):.2f}", fontsize=11)
        ax.set_xlabel("$R_t$")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    for j in range(len(etfs), len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle("市场风险 $R_t$ 的实证密度 vs 正态拟合 —— 尖峰厚尾特征", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "03_density_plots.png"))
    plt.close()


def plot_correlation_heatmap(etfs):
    """两张热图: ETF 之间的 L_t 相关 & R_t 相关"""
    L_panel = pd.DataFrame()
    R_panel = pd.DataFrame()
    for code, name in etfs:
        df = load_etf(code, name).dropna(subset=["L_t", "R_t"]).set_index("date")
        L_panel[name] = df["L_t"]
        R_panel[name] = df["R_t"]
    L_corr = L_panel.corr()
    R_corr = R_panel.corr()

    n = len(etfs)
    fig, axes = plt.subplots(1, 2, figsize=(max(13, 2 * n + 2), max(5, n + 1)))
    for ax, (corr, title) in zip(axes, [(L_corr, f"{n}只ETF的流动性风险 $L_t$ 相关系数"),
                                          (R_corr, f"{n}只ETF的市场风险 $R_t$ 相关系数")]):
        im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=30, ha="right")
        ax.set_yticklabels(corr.columns)
        for i in range(len(corr)):
            for j in range(len(corr)):
                color = "white" if abs(corr.values[i, j]) > 0.5 else "black"
                ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                        fontsize=9, color=color)
        ax.set_title(title, fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "04_correlation_heatmap.png"))
    plt.close()


def plot_l_r_scatter(etfs):
    """单只 ETF 的 L_t vs R_t 散点 + 边缘分布"""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for i, (code, name) in enumerate(etfs):
        ax = axes[i]
        df = load_etf(code, name).dropna(subset=["L_t", "R_t"])
        ax.scatter(df["L_t"], df["R_t"], s=4, alpha=0.3, color="#1f77b4")
        rho = df["L_t"].corr(df["R_t"])
        tau = df["L_t"].corr(df["R_t"], method="kendall")
        ax.set_xlabel("流动性风险 $L_t$")
        ax.set_ylabel("市场风险 $R_t$")
        ax.set_title(f"{name}  ρ={rho:.3f}, τ={tau:.3f}", fontsize=11)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.grid(alpha=0.3)
    for j in range(len(etfs), len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle("$L_t$ 与 $R_t$ 联合散点图", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "05_lt_rt_scatter.png"))
    plt.close()


# ============================================================
# 主流程
# ============================================================
def main():
    all_etfs = MAIN_ETFS + [CASE_ETF]

    print("=" * 60)
    print("第一步: 数据预处理 + 描述性统计")
    print("=" * 60)

    print("\n[1/4] 数据质量检查...")
    dq = data_quality_check(all_etfs)
    print(dq.to_string(index=False))

    print("\n[2/4] 描述性统计 + 统计检验...")
    ds = descriptive_table(all_etfs)
    print(ds.to_string(index=False))

    print("\n[3/4] 分子时段统计 (全部 6 只 ETF)...")
    sp = subperiod_stats(all_etfs)
    print(sp.head(20).to_string(index=False))

    print("\n[4/4] 生成图表 (全部 6 只 ETF)...")
    plot_volatility_series(all_etfs)
    print("  -> 波动序列图 (6 张) 已生成")
    plot_qq_panel(all_etfs)
    print("  -> Q-Q 图已生成 (6 子图)")
    plot_density_panel(all_etfs)
    print("  -> 密度对比图已生成 (6 子图)")
    plot_correlation_heatmap(all_etfs)
    print("  -> 相关性热图已生成 (6x6)")
    plot_l_r_scatter(all_etfs)
    print("  -> L_t-R_t 散点图已生成 (6 子图)")

    print("\n" + "=" * 60)
    print("完成! 输出位置:")
    print(f"  表格: {TABLE_DIR}")
    print(f"  图片: {FIG_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
