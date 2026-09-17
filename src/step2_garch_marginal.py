"""
第二步: 边缘分布建模 (GJR-GARCH-Skewed-t)
=========================================================
对每只 ETF 的两个风险序列建模:
    流动性风险:   一阶差分 ΔL_t = L_t - L_{t-1}  (解决非平稳问题)
    市场风险:    R_t = ETF/指数收益率           (已平稳)

模型规格 (主模型):
    ΔL_t = μ + ε_t,    ε_t = σ_t · z_t,    z_t ~ Skewed-t(η, λ)
    σ²_t = ω + α·ε²_{t-1} + γ·ε²_{t-1}·I(ε_{t-1}<0) + β·σ²_{t-1}      (GJR-GARCH)

模型选择: 对每个序列拟合 4 个嵌套模型, 用 AIC/BIC 比较
    1) GARCH(1,1)-Normal
    2) GJR-GARCH(1,1)-Normal
    3) GARCH(1,1)-StudentT
    4) GJR-GARCH(1,1)-SkewedT   ← 主模型

输出:
    tables/
        04_garch_model_selection.csv     模型选择 (AIC/BIC)
        05_garch_main_params.csv         主模型参数估计
        06_garch_diagnostics.csv         残差诊断 + PIT 均匀性检验
    figures/
        06_conditional_volatility_*.png  6 张条件波动率图
        07_std_resid_qq_*.png            标准化残差 Q-Q 图
        08_pit_uniform_check.png         PIT 输出的均匀性直方图
    pit_outputs/
        pit_<ETF代码>.csv                u_t (来自 ΔL_t), v_t (来自 R_t) -> 喂给第三步 Copula
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
from arch import arch_model
from arch.univariate import SkewStudent, StudentsT, Normal
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

# ============================================================
# 全局配置
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TABLE_DIR = os.path.join(BASE_DIR, "tables")
FIG_DIR = os.path.join(BASE_DIR, "figures")
PIT_DIR = os.path.join(BASE_DIR, "pit_outputs")
os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(PIT_DIR, exist_ok=True)

mpl.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "PingFang HK", "STHeiti", "Arial Unicode MS"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["font.size"] = 10
mpl.rcParams["figure.dpi"] = 100
mpl.rcParams["savefig.dpi"] = 150
mpl.rcParams["savefig.bbox"] = "tight"

ETFS = [
    ("510050", "上证50ETF"),
    ("510180", "上证180ETF"),
    ("510300", "沪深300ETF"),
    ("510500", "中证500ETF"),
    ("159915", "创业板ETF"),
    ("513050", "中概互联ETF"),
]

# 4 个候选模型
MODEL_SPECS = [
    ("GARCH-N",    dict(vol="GARCH", p=1, o=0, q=1, dist="normal")),
    ("GJR-N",      dict(vol="GARCH", p=1, o=1, q=1, dist="normal")),
    ("GARCH-t",    dict(vol="GARCH", p=1, o=0, q=1, dist="t")),
    ("GJR-SkT",    dict(vol="GARCH", p=1, o=1, q=1, dist="skewt")),  # ← 主模型
]


# ============================================================
# 辅助
# ============================================================
def load_data(code: str, name: str) -> pd.DataFrame:
    """读取并预处理: 计算 ΔL_t, 单位放大到 % 便于 GARCH 数值优化"""
    path = os.path.join(BASE_DIR, f"ready_{code}_{name}.csv")
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["dL_t"] = df["L_t"].diff()        # 流动性冲击
    df["R_pct"] = df["R_t"] * 100         # 收益率换算为百分比 (arch 推荐)
    df["dL_pct"] = df["dL_t"]             # ΔL_t 量纲已经合适, 不再缩放
    df = df.dropna(subset=["dL_t", "R_t"]).reset_index(drop=True)
    return df


def fit_one_model(series: np.ndarray, spec: dict, mean: str = "Constant", lags: int = 0):
    """
    mean: "Constant" 或 "AR"
    lags: AR 阶数 (仅当 mean='AR' 时生效)
    """
    if mean == "AR" and lags > 0:
        am = arch_model(series, mean="AR", lags=lags, **spec)
    else:
        am = arch_model(series, mean="Constant", **spec)
    return am.fit(disp="off", show_warning=False)


def pit_transform(std_resid: np.ndarray, dist_name: str, params: dict) -> np.ndarray:
    """
    将标准化残差经过对应分布的 CDF, 输出 [0,1] 上的 u 序列
    """
    if dist_name == "skewt":
        # Hansen Skewed-t: 参数 eta (自由度), lambda (偏斜)
        eta = params.get("eta", params.get("nu", None))
        lam = params.get("lambda", None)
        dist = SkewStudent()
        return dist.cdf(std_resid, np.array([eta, lam]))
    elif dist_name == "t":
        nu = params.get("nu", None)
        dist = StudentsT()
        return dist.cdf(std_resid, np.array([nu]))
    elif dist_name == "normal":
        return stats.norm.cdf(std_resid)
    else:
        raise ValueError(f"Unknown dist {dist_name}")


def diagnostics(std_resid: np.ndarray, lags: int = 10):
    lb = acorr_ljungbox(std_resid, lags=[lags], return_df=True)
    lb2 = acorr_ljungbox(std_resid ** 2, lags=[lags], return_df=True)
    arch_t = het_arch(std_resid, nlags=5)
    return {
        "LB(10)_stat":  float(lb["lb_stat"].iloc[0]),
        "LB(10)_p":     float(lb["lb_pvalue"].iloc[0]),
        "LB²(10)_stat": float(lb2["lb_stat"].iloc[0]),
        "LB²(10)_p":    float(lb2["lb_pvalue"].iloc[0]),
        "ARCH-LM(5)_stat": float(arch_t[0]),
        "ARCH-LM(5)_p":    float(arch_t[1]),
    }


# ============================================================
# 主流程: 模型选择 + 主模型拟合
# ============================================================
def main():
    selection_rows = []     # 模型选择 (AIC/BIC)
    main_param_rows = []    # 主模型参数
    diag_rows = []          # 残差诊断
    pit_uniform_check = {}  # 用于绘图

    for code, name in ETFS:
        print(f"\n{'=' * 60}")
        print(f"处理 {code} {name}")
        print(f"{'=' * 60}")
        df = load_data(code, name)
        date_idx = df["date"]

        for var_label, raw_series in [("ΔL_t", df["dL_t"].values),
                                       ("R_t",  df["R_pct"].values)]:
            print(f"\n--- 序列: {var_label} (n={len(raw_series)}) ---")

            # ADF 复检
            adf_p = adfuller(raw_series, autolag="AIC")[1]
            print(f"  ADF p = {adf_p:.4g}  ({'平稳' if adf_p < 0.05 else '不平稳'})")

            # 均值方程: ΔL_t 用 AR(1) 抑制残差自相关; R_t 用常数均值
            if var_label == "ΔL_t":
                mean_spec, mean_lags = "AR", 1
            else:
                mean_spec, mean_lags = "Constant", 0

            # 4 个模型选择
            best_main = None
            for mname, spec in MODEL_SPECS:
                try:
                    res = fit_one_model(raw_series, spec, mean=mean_spec, lags=mean_lags)
                    selection_rows.append({
                        "ETF": f"{code} {name}",
                        "变量": var_label,
                        "均值方程": "AR(1)" if mean_spec == "AR" else "Const",
                        "模型": mname,
                        "对数似然": round(res.loglikelihood, 2),
                        "AIC": round(res.aic, 2),
                        "BIC": round(res.bic, 2),
                        "参数数": res.num_params,
                    })
                    if mname == "GJR-SkT":
                        best_main = res
                except Exception as e:
                    print(f"  [警告] 模型 {mname} 拟合失败: {e}")

            if best_main is None:
                print(f"  [跳过] 主模型 GJR-SkT 拟合失败, 用 GARCH-t 替代")
                fallback_res = fit_one_model(raw_series, dict(vol="GARCH", p=1, o=0, q=1, dist="t"),
                                              mean=mean_spec, lags=mean_lags)
                best_main = fallback_res
                best_dist = "t"
            else:
                best_dist = "skewt"

            # 主模型参数
            params = best_main.params
            ar_coef = np.nan
            if mean_spec == "AR":
                # arch 命名通常是 "y[1]" 或 "Const" 或 "ar.L1" 之类, 找一下
                for k in params.index:
                    if k.lower().startswith(("y[", "ar.", "phi")) or k == "ar[1]" or k == "y[1]":
                        ar_coef = round(params[k], 4)
                        break
            param_dict = {
                "ETF": f"{code} {name}",
                "变量": var_label,
                "样本": len(raw_series),
                "μ (mu)":     round(params.get("mu", params.get("Const", np.nan)), 6)
                              if ("mu" in params.index or "Const" in params.index) else np.nan,
                "φ (AR1)":    ar_coef,
                "ω (omega)":  round(params.get("omega", np.nan), 6),
                "α (alpha[1])": round(params.get("alpha[1]", np.nan), 4),
                "γ (gamma[1])": round(params.get("gamma[1]", np.nan), 4)
                                if "gamma[1]" in params.index else np.nan,
                "β (beta[1])":  round(params.get("beta[1]", np.nan), 4),
                "η (自由度)": round(params.get("eta", params.get("nu", np.nan)), 3),
                "λ (偏斜)":   round(params.get("lambda", np.nan), 3)
                              if "lambda" in params.index else np.nan,
                "对数似然": round(best_main.loglikelihood, 2),
                "AIC": round(best_main.aic, 2),
                "BIC": round(best_main.bic, 2),
            }
            # α+β+γ/2 (持续性)
            a = params.get("alpha[1]", 0.0)
            b = params.get("beta[1]", 0.0)
            g = params.get("gamma[1]", 0.0) if "gamma[1]" in params.index else 0.0
            param_dict["α+β+γ/2"] = round(a + b + g / 2.0, 4)
            main_param_rows.append(param_dict)

            # 残差诊断 (兼容 arch 返回的 ndarray / pandas.Series)
            resid = np.asarray(best_main.resid)
            cond_vol = np.asarray(best_main.conditional_volatility)
            std_resid = resid / cond_vol
            std_resid = std_resid[~np.isnan(std_resid)]
            diag = diagnostics(std_resid)

            # PIT 变换
            pit_params = {
                "eta": params.get("eta", params.get("nu", None)),
                "nu":  params.get("nu", None),
                "lambda": params.get("lambda", None),
            }
            u = pit_transform(std_resid, best_dist, pit_params)
            # 修剪到 (0,1) 开区间, 避免 Copula 数值问题
            u = np.clip(u, 1e-6, 1 - 1e-6)

            # K-S 检验: PIT 输出是否服从均匀分布
            ks_stat, ks_p = stats.kstest(u, "uniform")

            diag_rows.append({
                "ETF": f"{code} {name}",
                "变量": var_label,
                **{k: round(v, 4) for k, v in diag.items()},
                "K-S(PIT)_stat": round(ks_stat, 4),
                "K-S(PIT)_p": round(ks_p, 4),
                "PIT均匀?": "是" if ks_p > 0.05 else "否",
            })

            print(f"  主模型 (GJR-SkT) AIC={best_main.aic:.2f}, BIC={best_main.bic:.2f}")
            print(f"  α={a:.4f}, β={b:.4f}, γ={g:.4f}, 持续性 α+β+γ/2={a+b+g/2:.4f}")
            print(f"  残差 Ljung-Box(10) p = {diag['LB(10)_p']:.4f}, ARCH-LM(5) p = {diag['ARCH-LM(5)_p']:.4f}")
            print(f"  PIT K-S p = {ks_p:.4f}  ({'通过' if ks_p > 0.05 else '不通过'} 均匀性检验)")

            # 存 PIT 序列
            key = f"{code}_{name}_{var_label}"
            pit_uniform_check[key] = u

            # 临时存储 (后续合并)
            if var_label == "ΔL_t":
                u_L, vol_L, resid_L = u, np.asarray(best_main.conditional_volatility), std_resid
            else:
                u_R, vol_R, resid_R = u, np.asarray(best_main.conditional_volatility), std_resid

        # 合并保存当只 ETF 的 PIT 输出
        # 对齐到 min(len(u_L), len(u_R))
        n_common = min(len(u_L), len(u_R))
        date_aligned = df["date"].iloc[-n_common:].reset_index(drop=True)
        pit_df = pd.DataFrame({
            "date": date_aligned,
            "u_L": u_L[-n_common:],
            "u_R": u_R[-n_common:],
            "std_resid_L": resid_L[-n_common:],
            "std_resid_R": resid_R[-n_common:],
            "cond_vol_L": vol_L[-n_common:],
            "cond_vol_R": vol_R[-n_common:],
        })
        pit_df.to_csv(os.path.join(PIT_DIR, f"pit_{code}.csv"), index=False, encoding="utf-8-sig")

        # 出条件波动率图
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        axes[0].plot(date_aligned, vol_L[-n_common:], color="#1f77b4", linewidth=0.7)
        axes[0].set_ylabel("$\\sigma_t$ 流动性冲击")
        axes[0].set_title(f"{name} ({code}) 条件波动率 $\\sigma_t$ (GJR-GARCH-SkT 估计)", fontweight="bold")
        axes[0].grid(alpha=0.3)
        axes[1].plot(date_aligned, vol_R[-n_common:], color="#d62728", linewidth=0.7)
        axes[1].set_ylabel("$\\sigma_t$ 市场风险")
        axes[1].set_xlabel("日期")
        axes[1].grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"06_conditional_volatility_{code}_{name}.png"))
        plt.close()

    # ============================================================
    # 保存表格
    # ============================================================
    pd.DataFrame(selection_rows).to_csv(
        os.path.join(TABLE_DIR, "04_garch_model_selection.csv"),
        index=False, encoding="utf-8-sig")
    pd.DataFrame(main_param_rows).to_csv(
        os.path.join(TABLE_DIR, "05_garch_main_params.csv"),
        index=False, encoding="utf-8-sig")
    pd.DataFrame(diag_rows).to_csv(
        os.path.join(TABLE_DIR, "06_garch_diagnostics.csv"),
        index=False, encoding="utf-8-sig")

    # ============================================================
    # PIT 均匀性可视化
    # ============================================================
    n_panels = len(pit_uniform_check)
    ncols = 4
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 2.6))
    axes = axes.flatten()
    for i, (key, u) in enumerate(pit_uniform_check.items()):
        ax = axes[i]
        ax.hist(u, bins=25, density=True, color="#1f77b4", alpha=0.7, edgecolor="white")
        ax.axhline(1.0, color="#d62728", linestyle="--", linewidth=1)
        title = key.replace("_", " ")
        ax.set_title(title, fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, max(2.0, ax.get_ylim()[1]))
        ax.tick_params(labelsize=8)
    for j in range(n_panels, len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle("PIT 输出的均匀性检查 (红线 = U[0,1] 理想密度)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "08_pit_uniform_check.png"))
    plt.close()

    # 主模型 Q-Q (标准化残差 vs 学生 t)
    fig, axes = plt.subplots(3, 4, figsize=(14, 9))
    axes = axes.flatten()
    keys = list(pit_uniform_check.keys())
    for i, key in enumerate(keys):
        ax = axes[i]
        # PIT 输出 -> 转 normal scale 看 Q-Q 是否大致正态
        u = pit_uniform_check[key]
        z = stats.norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))
        stats.probplot(z, dist="norm", plot=ax)
        ax.set_title(key.replace("_", " "), fontsize=9)
        # 覆盖 scipy.probplot 自动加的英文 title 和坐标轴
        ax.tick_params(labelsize=7)
        ax.get_lines()[0].set_marker(".")
        ax.get_lines()[0].set_markersize(2)
        ax.get_lines()[0].set_markerfacecolor("#1f77b4")
        ax.get_lines()[0].set_markeredgecolor("none")
        ax.get_lines()[1].set_color("#d62728")
        ax.get_lines()[1].set_linewidth(1)
        ax.set_xlabel("理论分位数", fontsize=8)
        ax.set_ylabel(r"$\Phi^{-1}(\mathrm{PIT})$", fontsize=8)
    for j in range(len(keys), len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle(r"PIT 经 $\Phi^{-1}$ 变换后的正态 Q-Q 图 (应近似一条直线)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "07_pit_qq_check.png"))
    plt.close()

    # ============================================================
    # 终端预览
    # ============================================================
    print("\n" + "=" * 70)
    print("第二步完成! 结果摘要:")
    print("=" * 70)
    sel_df = pd.DataFrame(selection_rows)
    print("\n[模型选择] 按 (ETF, 变量) 看, 哪个模型 BIC 最小:")
    best_by_bic = sel_df.loc[sel_df.groupby(["ETF", "变量"])["BIC"].idxmin()]
    print(best_by_bic[["ETF", "变量", "模型", "AIC", "BIC", "对数似然"]].to_string(index=False))

    print("\n[主模型参数 (GJR-SkT)] :")
    print(pd.DataFrame(main_param_rows).to_string(index=False))

    print("\n[残差诊断 + PIT 均匀性] :")
    print(pd.DataFrame(diag_rows).to_string(index=False))

    print("\n输出:")
    print(f"  表格: {TABLE_DIR}/04_garch_model_selection.csv")
    print(f"        {TABLE_DIR}/05_garch_main_params.csv")
    print(f"        {TABLE_DIR}/06_garch_diagnostics.csv")
    print(f"  PIT:  {PIT_DIR}/pit_<ETF代码>.csv  (这就是第三步 Copula 的输入)")
    print(f"  图:   {FIG_DIR}/06_*..08_*.png")


if __name__ == "__main__":
    main()
