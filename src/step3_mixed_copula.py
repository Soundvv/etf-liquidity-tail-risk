"""
第三步: 混合 Copula 建模 (核心创新点)
=========================================================
对每只 ETF 的 (u_L, u_R) 配对数据 (由第二步 PIT 变换得到), 拟合:
    1) 5 个单一 Copula 基准模型 (Gumbel/Clayton/Frank/Gaussian/StudentT)
    2) 静态 3 分量混合 Copula (Gumbel + Clayton + Frank)
    3) 6 个子时段的静态混合 Copula (揭示状态依赖性)
    4) 250 日滚动窗口的时变混合 Copula (画出 λ_L(t) 动态曲线)

输出:
    tables/
        07_single_copula_compare.csv     5 个单 Copula 的 AIC/BIC 对比
        08_mixed_copula_static.csv       静态混合 Copula 全部参数
        09_subperiod_mixed_copula.csv    6 个子时段的拟合结果
        10_tail_dependence_compare.csv   尾部相依系数对比表 (docx vs 我们)
    figures/
        09_copula_weights_bar.png        各 ETF 静态混合 Copula 权重柱状图
        10_tail_dep_compare.png          尾部相依系数对比 (docx vs 混合)
        11_subperiod_lambda.png          子时段 λ_L 热图
        12_rolling_lambda_<ETF>.png      6 张时变 λ_L(t), λ_U(t) 曲线图
        13_copula_density_contour.png    经验密度 vs 拟合混合 Copula 等高线
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
import time as _time

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

SUBPERIODS = {
    "牛市2014-2015":     ("2014-07-01", "2015-06-12"),
    "股灾2015-2016":     ("2015-06-12", "2016-02-29"),
    "震荡2017-2019":     ("2017-01-01", "2019-12-31"),
    "疫情2020":          ("2020-01-23", "2020-12-31"),
    "中概危机2021-2022": ("2021-02-01", "2022-10-31"),
    "复苏2023-2025":     ("2023-01-01", "2025-12-31"),
}

EVENTS = [
    ("2015-06-12", "股灾"),
    ("2016-01-04", "熔断"),
    ("2018-03-22", "贸易战"),
    ("2020-01-23", "疫情"),
    ("2022-03-14", "中概危机"),
    ("2024-09-24", "924行情"),
]

EPS = 1e-8


# ============================================================
# Copula PDF / 对数似然
# ============================================================
def gumbel_pdf(u, v, theta):
    """
    Gumbel Copula PDF, θ >= 1
    参考: McNeil, Frey & Embrechts (2015), Eq. 7.46
        c(u,v) = C(u,v) / (uv) · (lu·lv)^(θ-1) · A^(1/θ - 2) · [A^(1/θ) + θ - 1]
    在 θ=1 时 PDF=1 (独立)
    """
    if theta < 1:
        return np.full_like(u, EPS)
    u = np.clip(u, EPS, 1 - EPS)
    v = np.clip(v, EPS, 1 - EPS)
    lu = -np.log(u)
    lv = -np.log(v)
    A = lu**theta + lv**theta
    A_pow = A ** (1.0 / theta)
    C = np.exp(-A_pow)
    # 关键: 指数是 1/θ - 2, 不是 2/θ - 2
    pdf = C / (u * v) * (lu * lv) ** (theta - 1) * A ** (1.0 / theta - 2) * (A_pow + theta - 1)
    return np.clip(pdf, EPS, None)


def clayton_pdf(u, v, theta):
    """Clayton Copula PDF, θ > 0"""
    if theta <= 0:
        return np.full_like(u, EPS)
    u = np.clip(u, EPS, 1 - EPS)
    v = np.clip(v, EPS, 1 - EPS)
    pdf = (1 + theta) * (u * v) ** (-1 - theta) * (u ** (-theta) + v ** (-theta) - 1) ** (-2 - 1.0 / theta)
    return np.clip(pdf, EPS, None)


def frank_pdf(u, v, theta):
    """
    Frank Copula PDF, θ != 0
    参考: Nelsen (2006), Example 4.17
        c(u,v) = θ(1-e^(-θ)) e^(-θ(u+v)) / [1 - e^(-θ) - (1-e^(-θu))(1-e^(-θv))]²
    """
    if abs(theta) < 1e-4:
        # 趋于独立
        return np.ones_like(u)
    u = np.clip(u, EPS, 1 - EPS)
    v = np.clip(v, EPS, 1 - EPS)
    eu = np.exp(-theta * u)
    ev = np.exp(-theta * v)
    e1 = np.exp(-theta)
    # 修正: 正号
    num = theta * (1 - e1) * np.exp(-theta * (u + v))
    den = ((1 - e1) - (1 - eu) * (1 - ev)) ** 2
    pdf = num / den
    return np.clip(pdf, EPS, None)


def gaussian_pdf(u, v, rho):
    """Gaussian Copula PDF, ρ ∈ (-1, 1)"""
    if abs(rho) >= 0.9999:
        return np.full_like(u, EPS)
    x = stats.norm.ppf(np.clip(u, EPS, 1 - EPS))
    y = stats.norm.ppf(np.clip(v, EPS, 1 - EPS))
    num = -(rho ** 2 * (x ** 2 + y ** 2) - 2 * rho * x * y) / (2 * (1 - rho ** 2))
    pdf = 1.0 / np.sqrt(1 - rho ** 2) * np.exp(num)
    return np.clip(pdf, EPS, None)


def studentt_pdf(u, v, rho, nu):
    """Student-t Copula PDF, ρ ∈ (-1,1), ν > 2"""
    if abs(rho) >= 0.9999 or nu <= 2:
        return np.full_like(u, EPS)
    x = stats.t.ppf(np.clip(u, EPS, 1 - EPS), df=nu)
    y = stats.t.ppf(np.clip(v, EPS, 1 - EPS), df=nu)
    from scipy.special import gammaln
    # log-density 形式更稳定
    log_num = (gammaln((nu + 2) / 2) + gammaln(nu / 2)
               - 2 * gammaln((nu + 1) / 2) - 0.5 * np.log(1 - rho ** 2))
    log_term1 = -((nu + 2) / 2) * np.log(1 + (x ** 2 - 2 * rho * x * y + y ** 2) / (nu * (1 - rho ** 2)))
    log_term2 = ((nu + 1) / 2) * np.log(1 + x ** 2 / nu) + ((nu + 1) / 2) * np.log(1 + y ** 2 / nu)
    log_pdf = log_num + log_term1 + log_term2
    return np.clip(np.exp(log_pdf), EPS, None)


# ============================================================
# 单一 Copula MLE
# ============================================================
def fit_gumbel(u, v):
    def neg_ll(params):
        theta = 1.0 + np.exp(params[0])  # θ >= 1
        return -np.sum(np.log(gumbel_pdf(u, v, theta)))
    res = minimize(neg_ll, x0=[0.0], method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 500})
    theta = 1.0 + np.exp(res.x[0])
    ll = -res.fun
    return dict(theta=theta, loglik=ll, k=1)


def fit_clayton(u, v):
    def neg_ll(params):
        theta = np.exp(params[0])
        return -np.sum(np.log(clayton_pdf(u, v, theta)))
    res = minimize(neg_ll, x0=[0.0], method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 500})
    theta = np.exp(res.x[0])
    return dict(theta=theta, loglik=-res.fun, k=1)


def fit_frank(u, v):
    def neg_ll(params):
        theta = params[0]
        return -np.sum(np.log(frank_pdf(u, v, theta)))
    res = minimize(neg_ll, x0=[1.0], method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 500})
    return dict(theta=res.x[0], loglik=-res.fun, k=1)


def fit_gaussian(u, v):
    def neg_ll(params):
        rho = np.tanh(params[0])
        return -np.sum(np.log(gaussian_pdf(u, v, rho)))
    res = minimize(neg_ll, x0=[0.0], method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 500})
    rho = np.tanh(res.x[0])
    return dict(rho=rho, loglik=-res.fun, k=1)


def fit_studentt(u, v):
    def neg_ll(params):
        rho = np.tanh(params[0])
        nu = 2.5 + np.exp(params[1])
        return -np.sum(np.log(studentt_pdf(u, v, rho, nu)))
    res = minimize(neg_ll, x0=[0.0, 2.0], method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 500})
    rho = np.tanh(res.x[0])
    nu = 2.5 + np.exp(res.x[1])
    return dict(rho=rho, nu=nu, loglik=-res.fun, k=2)


# ============================================================
# 混合 Copula MLE (Gumbel + Clayton + Frank)
# ============================================================
def softmax3(a1, a2):
    """三分量 softmax: w1, w2, w3 用两个无约束参数表达"""
    e = np.array([np.exp(a1), np.exp(a2), 1.0])
    w = e / e.sum()
    return w  # w1 w2 w3, sum=1


def mixed_pdf(u, v, w, theta_G, theta_C, theta_F):
    return w[0] * gumbel_pdf(u, v, theta_G) + \
           w[1] * clayton_pdf(u, v, theta_C) + \
           w[2] * frank_pdf(u, v, theta_F)


def fit_mixed(u, v, n_starts=5, verbose=False):
    """3 分量混合 Copula MLE, 多起点以避免局部最优"""
    best = None
    np.random.seed(42)
    starts = [[0.0, 0.0, 0.0, 0.0, 1.0]]  # (a1, a2, log(θ_G-1), log(θ_C), θ_F)
    for _ in range(n_starts - 1):
        starts.append([np.random.uniform(-1, 1),
                       np.random.uniform(-1, 1),
                       np.random.uniform(-1, 1),
                       np.random.uniform(-1, 1),
                       np.random.uniform(-2, 4)])

    def neg_ll(params):
        a1, a2, bG, bC, bF = params
        w = softmax3(a1, a2)
        theta_G = 1.0 + np.exp(bG)
        theta_C = np.exp(bC)
        theta_F = bF
        pdf = mixed_pdf(u, v, w, theta_G, theta_C, theta_F)
        return -np.sum(np.log(pdf))

    for x0 in starts:
        try:
            res = minimize(neg_ll, x0=x0, method="Nelder-Mead",
                           options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 1500})
            # 拒绝非有限值的解 (防御数值溢出)
            if not np.isfinite(res.fun):
                continue
            if best is None or res.fun < best.fun:
                best = res
        except Exception as e:
            if verbose:
                print(f"  fit_mixed start failed: {e}")
    if best is None:
        # 极少数极端情况下所有起点都不收敛, 返回退化的"独立"结果
        return dict(w_G=0.0, w_C=0.0, w_F=1.0,
                    theta_G=1.0, theta_C=1e-6, theta_F=0.0,
                    lambda_U=0.0, lambda_L=0.0,
                    loglik=0.0, k=5)

    a1, a2, bG, bC, bF = best.x
    w = softmax3(a1, a2)
    theta_G = 1.0 + np.exp(bG)
    theta_C = np.exp(bC)
    theta_F = bF
    ll = -best.fun
    # 上尾相依: λ_U = w_G · (2 - 2^(1/θ_G))
    # 下尾相依: λ_L = w_C · 2^(-1/θ_C)
    lambda_U = w[0] * (2 - 2 ** (1.0 / theta_G))
    lambda_L = w[1] * (2 ** (-1.0 / theta_C))
    return dict(w_G=w[0], w_C=w[1], w_F=w[2],
                theta_G=theta_G, theta_C=theta_C, theta_F=theta_F,
                lambda_U=lambda_U, lambda_L=lambda_L,
                loglik=ll, k=5)


# ============================================================
# AIC/BIC
# ============================================================
def aic(ll, k):
    return 2 * k - 2 * ll


def bic(ll, k, n):
    return k * np.log(n) - 2 * ll


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 70)
    print("第三步: 混合 Copula 建模")
    print("=" * 70)

    # ---------- 1. 加载 PIT 数据 ----------
    pit_dict = {}
    for code, name in ETFS:
        path = os.path.join(PIT_DIR, f"pit_{code}.csv")
        df = pd.read_csv(path, parse_dates=["date"])
        pit_dict[code] = df
        print(f"  {code} {name}: 已加载 {len(df)} 行")

    # ---------- 2. 单一 Copula 比较 ----------
    print("\n[1/4] 拟合 5 个单一 Copula + 静态混合 Copula ...")
    single_rows = []
    mixed_rows = []
    for code, name in ETFS:
        df = pit_dict[code].dropna(subset=["u_L", "u_R"])
        u, v = df["u_L"].values, df["u_R"].values
        n = len(u)
        print(f"\n  -- {code} {name} (n={n}) --")

        # 单 Copula
        models = {
            "Gumbel":  fit_gumbel(u, v),
            "Clayton": fit_clayton(u, v),
            "Frank":   fit_frank(u, v),
            "Gaussian": fit_gaussian(u, v),
            "Student-t": fit_studentt(u, v),
        }
        for mname, res in models.items():
            single_rows.append({
                "ETF": f"{code} {name}",
                "模型": mname,
                "参数": ({k: round(v_, 4) for k, v_ in res.items() if k not in ("loglik", "k")}),
                "对数似然": round(res["loglik"], 2),
                "k": res["k"],
                "AIC": round(aic(res["loglik"], res["k"]), 2),
                "BIC": round(bic(res["loglik"], res["k"], n), 2),
            })
            print(f"    {mname:10s} ll={res['loglik']:.2f}  AIC={aic(res['loglik'], res['k']):.2f}  BIC={bic(res['loglik'], res['k'], n):.2f}")

        # 混合 Copula
        mres = fit_mixed(u, v)
        mixed_rows.append({
            "ETF": f"{code} {name}", "样本": n,
            "w_G": round(mres["w_G"], 4),
            "w_C": round(mres["w_C"], 4),
            "w_F": round(mres["w_F"], 4),
            "θ_G": round(mres["theta_G"], 4),
            "θ_C": round(mres["theta_C"], 4),
            "θ_F": round(mres["theta_F"], 4),
            "λ_U": round(mres["lambda_U"], 4),
            "λ_L": round(mres["lambda_L"], 4),
            "对数似然": round(mres["loglik"], 2),
            "AIC": round(aic(mres["loglik"], mres["k"]), 2),
            "BIC": round(bic(mres["loglik"], mres["k"], n), 2),
        })
        single_rows.append({
            "ETF": f"{code} {name}",
            "模型": "Mixed (G+C+F)",
            "参数": f"w=({mres['w_G']:.2f},{mres['w_C']:.2f},{mres['w_F']:.2f}); θ=({mres['theta_G']:.2f},{mres['theta_C']:.2f},{mres['theta_F']:.2f})",
            "对数似然": round(mres["loglik"], 2),
            "k": mres["k"],
            "AIC": round(aic(mres["loglik"], mres["k"]), 2),
            "BIC": round(bic(mres["loglik"], mres["k"], n), 2),
        })
        print(f"    {'Mixed (G+C+F)':10s} ll={mres['loglik']:.2f}  AIC={aic(mres['loglik'], 5):.2f}  BIC={bic(mres['loglik'], 5, n):.2f}")
        print(f"      -> w_G={mres['w_G']:.3f}, w_C={mres['w_C']:.3f}, w_F={mres['w_F']:.3f}")
        print(f"      -> λ_U={mres['lambda_U']:.4f}, λ_L={mres['lambda_L']:.4f}")

    pd.DataFrame(single_rows).to_csv(
        os.path.join(TABLE_DIR, "07_single_copula_compare.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(mixed_rows).to_csv(
        os.path.join(TABLE_DIR, "08_mixed_copula_static.csv"), index=False, encoding="utf-8-sig")

    # ---------- 3. 子时段分析 ----------
    print("\n[2/4] 子时段混合 Copula 拟合 ...")
    sub_rows = []
    for code, name in ETFS:
        df = pit_dict[code]
        for pname, (start, end) in SUBPERIODS.items():
            sub = df[(df["date"] >= start) & (df["date"] <= end)].dropna(subset=["u_L", "u_R"])
            if len(sub) < 80:
                continue
            try:
                m = fit_mixed(sub["u_L"].values, sub["u_R"].values, n_starts=3)
                sub_rows.append({
                    "ETF": f"{code} {name}", "时段": pname, "天数": len(sub),
                    "w_G": round(m["w_G"], 3),
                    "w_C": round(m["w_C"], 3),
                    "w_F": round(m["w_F"], 3),
                    "θ_G": round(m["theta_G"], 3),
                    "θ_C": round(m["theta_C"], 3),
                    "λ_U": round(m["lambda_U"], 4),
                    "λ_L": round(m["lambda_L"], 4),
                    "Kendall τ": round(stats.kendalltau(sub["u_L"], sub["u_R"])[0], 4),
                    "对数似然": round(m["loglik"], 2),
                })
            except Exception as e:
                print(f"    [警告] {code} {pname} 拟合失败: {e}")
    sp_df = pd.DataFrame(sub_rows)
    sp_df.to_csv(os.path.join(TABLE_DIR, "09_subperiod_mixed_copula.csv"),
                 index=False, encoding="utf-8-sig")
    print(f"  -> 完成, 共 {len(sp_df)} 行")

    # ---------- 4. 时变混合 Copula (滚动窗口) ----------
    print("\n[3/4] 时变混合 Copula (250 日滚动窗口, 60 日步长) ...")
    rolling_dict = {}
    for code, name in ETFS:
        df = pit_dict[code].dropna(subset=["u_L", "u_R"]).reset_index(drop=True)
        window = 250
        step = 60
        rows = []
        t0 = _time.time()
        for end_idx in range(window, len(df), step):
            sub = df.iloc[end_idx - window:end_idx]
            try:
                m = fit_mixed(sub["u_L"].values, sub["u_R"].values, n_starts=2)
                rows.append({
                    "date": sub["date"].iloc[-1],
                    "w_G": m["w_G"], "w_C": m["w_C"], "w_F": m["w_F"],
                    "θ_G": m["theta_G"], "θ_C": m["theta_C"], "θ_F": m["theta_F"],
                    "λ_U": m["lambda_U"], "λ_L": m["lambda_L"],
                    "loglik": m["loglik"],
                })
            except Exception:
                continue
        rdf = pd.DataFrame(rows)
        rolling_dict[code] = rdf
        rdf.to_csv(os.path.join(TABLE_DIR, f"rolling_{code}.csv"), index=False, encoding="utf-8-sig")
        print(f"  {code} {name}: {len(rdf)} 个窗口, 用时 {_time.time() - t0:.1f}s")

    # ---------- 5. 与 docx 对比表 ----------
    docx_results = {
        "510050 上证50ETF": {"τ_docx": -0.0107, "λ_U_docx": 0.0146, "λ_L_docx": 0.0000},
        "510180 上证180ETF": {"τ_docx": 0.0224, "λ_U_docx": -0.0320, "λ_L_docx": 0.0000},
        "510300 沪深300ETF": {"τ_docx": 0.0135, "λ_U_docx": -0.0190, "λ_L_docx": 0.0000},
        "510500 中证500ETF": {"τ_docx": -0.0177, "λ_U_docx": 0.0239, "λ_L_docx": 0.0000},
        "159915 创业板ETF": {"τ_docx": -0.0039, "λ_U_docx": 0.0054, "λ_L_docx": 0.0000},
        "513050 中概互联ETF": {"τ_docx": np.nan, "λ_U_docx": np.nan, "λ_L_docx": np.nan},
    }
    compare_rows = []
    for row in mixed_rows:
        key = row["ETF"]
        d = docx_results.get(key, {})
        compare_rows.append({
            "ETF": key,
            "τ (docx)":         d.get("τ_docx", np.nan),
            "λ_U (docx-静态)":  d.get("λ_U_docx", np.nan),
            "λ_L (docx-静态)":  d.get("λ_L_docx", np.nan),
            "λ_U (本研究-混合)": row["λ_U"],
            "λ_L (本研究-混合)": row["λ_L"],
            "λ_U 改善":         (row["λ_U"] - d.get("λ_U_docx", 0))
                                if not pd.isna(d.get("λ_U_docx", np.nan)) else np.nan,
            "λ_L 改善":         (row["λ_L"] - d.get("λ_L_docx", 0))
                                if not pd.isna(d.get("λ_L_docx", np.nan)) else np.nan,
        })
    pd.DataFrame(compare_rows).to_csv(
        os.path.join(TABLE_DIR, "10_tail_dependence_compare.csv"),
        index=False, encoding="utf-8-sig")

    # ---------- 6. 可视化 ----------
    print("\n[4/4] 生成图表 ...")

    # 6.1 静态混合权重柱状图
    fig, ax = plt.subplots(figsize=(11, 5))
    etf_labels = [r["ETF"].split(" ")[1] for r in mixed_rows]
    w_G_arr = np.array([r["w_G"] for r in mixed_rows])
    w_C_arr = np.array([r["w_C"] for r in mixed_rows])
    w_F_arr = np.array([r["w_F"] for r in mixed_rows])
    x = np.arange(len(etf_labels))
    ax.bar(x, w_G_arr, label="Gumbel (上尾)", color="#d62728")
    ax.bar(x, w_C_arr, bottom=w_G_arr, label="Clayton (下尾)", color="#1f77b4")
    ax.bar(x, w_F_arr, bottom=w_G_arr + w_C_arr, label="Frank (对称)", color="#888888")
    ax.set_xticks(x)
    ax.set_xticklabels(etf_labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("权重", fontsize=11)
    ax.set_title("各 ETF 静态混合 Copula 的分量权重", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    for i, (g, c, f) in enumerate(zip(w_G_arr, w_C_arr, w_F_arr)):
        if g > 0.05: ax.text(i, g / 2, f"{g:.2f}", ha="center", va="center", color="white", fontsize=9)
        if c > 0.05: ax.text(i, g + c / 2, f"{c:.2f}", ha="center", va="center", color="white", fontsize=9)
        if f > 0.05: ax.text(i, g + c + f / 2, f"{f:.2f}", ha="center", va="center", color="white", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "09_copula_weights_bar.png"))
    plt.close()

    # 6.2 尾部相依对比 (docx vs 我们)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    labels = [r["ETF"].split(" ")[1] for r in compare_rows[:5]]  # 仅前 5 (docx 不含中概互联)
    x = np.arange(len(labels))
    width = 0.35
    lu_docx = [r["λ_U (docx-静态)"] for r in compare_rows[:5]]
    lu_ours = [r["λ_U (本研究-混合)"] for r in compare_rows[:5]]
    ll_docx = [r["λ_L (docx-静态)"] for r in compare_rows[:5]]
    ll_ours = [r["λ_L (本研究-混合)"] for r in compare_rows[:5]]

    axes[0].bar(x - width / 2, lu_docx, width, label="docx 静态 Gumbel", color="#888888")
    axes[0].bar(x + width / 2, lu_ours, width, label="本研究 混合", color="#d62728")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, rotation=15)
    axes[0].set_title("上尾相依系数 $\\lambda_U$ — 牛市极端", fontsize=11)
    axes[0].set_ylabel("$\\lambda_U$")
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend()

    axes[1].bar(x - width / 2, ll_docx, width, label="docx 静态 Clayton", color="#888888")
    axes[1].bar(x + width / 2, ll_ours, width, label="本研究 混合", color="#1f77b4")
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, rotation=15)
    axes[1].set_title("下尾相依系数 $\\lambda_L$ — 熊市极端", fontsize=11)
    axes[1].set_ylabel("$\\lambda_L$")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend()
    fig.suptitle("混合 Copula vs docx 静态 Copula 的尾部相依系数对比",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "10_tail_dep_compare.png"))
    plt.close()

    # 6.3 子时段 λ_L 热图
    if len(sp_df) > 0:
        pivot_L = sp_df.pivot_table(index="ETF", columns="时段", values="λ_L")
        pivot_U = sp_df.pivot_table(index="ETF", columns="时段", values="λ_U")
        # 列按 SUBPERIODS 顺序
        col_order = [k for k in SUBPERIODS.keys() if k in pivot_L.columns]
        pivot_L = pivot_L[col_order]
        pivot_U = pivot_U[col_order]

        fig, axes = plt.subplots(2, 1, figsize=(11, 7))
        for ax, mat, title, cmap in zip(axes, [pivot_L, pivot_U],
                                         ["下尾相依 $\\lambda_L$ (熊市极端联动)",
                                          "上尾相依 $\\lambda_U$ (牛市极端联动)"],
                                         ["Blues", "Reds"]):
            im = ax.imshow(mat.values, cmap=cmap, aspect="auto", vmin=0)
            ax.set_xticks(range(len(mat.columns)))
            ax.set_xticklabels(mat.columns, rotation=15, ha="right")
            ax.set_yticks(range(len(mat.index)))
            ax.set_yticklabels([s.split(" ")[1] for s in mat.index])
            for i in range(len(mat.index)):
                for j in range(len(mat.columns)):
                    v_ = mat.values[i, j]
                    if not np.isnan(v_):
                        color = "white" if v_ > mat.values[~np.isnan(mat.values)].mean() else "black"
                        ax.text(j, i, f"{v_:.3f}", ha="center", va="center",
                                fontsize=8, color=color)
            ax.set_title(title, fontsize=11, fontweight="bold")
            fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        fig.suptitle("各 ETF × 各时段的尾部相依系数热图", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "11_subperiod_lambda.png"))
        plt.close()

    # 6.4 时变 λ_L(t), λ_U(t) 曲线
    for code, name in ETFS:
        rdf = rolling_dict[code]
        if len(rdf) == 0:
            continue
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        axes[0].plot(rdf["date"], rdf["λ_L"], color="#1f77b4", linewidth=1.5, label="$\\lambda_L$ 下尾相依")
        axes[0].fill_between(rdf["date"], 0, rdf["λ_L"], color="#1f77b4", alpha=0.2)
        axes[0].set_ylabel("$\\lambda_L$ 下尾相依")
        axes[0].set_title(f"{name} ({code}) 时变混合 Copula 尾部相依系数 (250d 滚动窗口)",
                          fontsize=12, fontweight="bold")
        axes[0].grid(alpha=0.3)
        axes[0].axhline(0, color="black", linewidth=0.5)

        axes[1].plot(rdf["date"], rdf["λ_U"], color="#d62728", linewidth=1.5, label="$\\lambda_U$ 上尾相依")
        axes[1].fill_between(rdf["date"], 0, rdf["λ_U"], color="#d62728", alpha=0.2)
        axes[1].set_ylabel("$\\lambda_U$ 上尾相依")
        axes[1].set_xlabel("日期")
        axes[1].grid(alpha=0.3)
        axes[1].axhline(0, color="black", linewidth=0.5)

        # 事件标注
        for date_str, label in EVENTS:
            d = pd.to_datetime(date_str)
            if rdf["date"].min() <= d <= rdf["date"].max():
                for ax in axes:
                    ax.axvline(d, color="gray", linewidth=0.6, linestyle=":", alpha=0.6)
                axes[0].annotate(label, xy=(d, axes[0].get_ylim()[1]),
                                 xytext=(0, -8), textcoords="offset points",
                                 fontsize=7, color="gray", ha="center", rotation=90, va="top")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"12_rolling_lambda_{code}_{name}.png"))
        plt.close()

    # 6.5 综合时变 λ_L(t) 对比 (6 只 ETF 在同一张图)
    fig, ax = plt.subplots(figsize=(13, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for (code, name), color in zip(ETFS, colors):
        rdf = rolling_dict[code]
        if len(rdf) == 0:
            continue
        ax.plot(rdf["date"], rdf["λ_L"], color=color, linewidth=1.4, label=name, alpha=0.8)
    for date_str, label in EVENTS:
        d = pd.to_datetime(date_str)
        ax.axvline(d, color="gray", linewidth=0.7, linestyle="--", alpha=0.6)
        ax.text(d, ax.get_ylim()[1] * 0.95, label, fontsize=8, rotation=90,
                ha="center", va="top", color="gray")
    ax.set_ylabel("$\\lambda_L$ 下尾相依系数 (流动性-市场风险熊市联动)")
    ax.set_xlabel("日期")
    ax.set_title("6 只 ETF 时变下尾相依系数 $\\lambda_L(t)$ —— 揭示风险联动的时变特征",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", ncol=2, fontsize=9)
    ax.axhline(0, color="black", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "12_rolling_lambda_L_all_ETFs.png"))
    plt.close()

    # 6.6 经验密度 vs 混合 Copula 密度等高线
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for i, (code, name) in enumerate(ETFS):
        ax = axes[i]
        m = mixed_rows[i]
        df = pit_dict[code].dropna(subset=["u_L", "u_R"])
        u, v = df["u_L"].values, df["u_R"].values
        ax.scatter(u, v, s=2, alpha=0.2, color="#1f77b4")
        # 计算混合 Copula 在网格上的 PDF
        grid = np.linspace(0.02, 0.98, 50)
        U_, V_ = np.meshgrid(grid, grid)
        w = np.array([m["w_G"], m["w_C"], m["w_F"]])
        Z = (w[0] * gumbel_pdf(U_.flatten(), V_.flatten(), m["θ_G"]) +
             w[1] * clayton_pdf(U_.flatten(), V_.flatten(), m["θ_C"]) +
             w[2] * frank_pdf(U_.flatten(), V_.flatten(), m["θ_F"])).reshape(U_.shape)
        cs = ax.contour(U_, V_, np.log(Z + EPS), levels=8, colors="#d62728", linewidths=0.8)
        ax.set_xlabel("$u_L$"); ax.set_ylabel("$u_R$")
        ax.set_title(f"{name}\nw=({m['w_G']:.2f},{m['w_C']:.2f},{m['w_F']:.2f}); $\\lambda_L$={m['λ_L']:.3f}",
                     fontsize=9)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.suptitle("PIT 散点 + 混合 Copula log-密度等高线", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "13_copula_density_contour.png"))
    plt.close()

    # ---------- 终端总结 ----------
    print("\n" + "=" * 70)
    print("第三步完成! 关键结果摘要:")
    print("=" * 70)

    print("\n[静态混合 Copula] 各 ETF 的关键参数:")
    print(pd.DataFrame(mixed_rows)[
        ["ETF", "w_G", "w_C", "w_F", "θ_G", "θ_C", "λ_U", "λ_L", "AIC", "BIC"]
    ].to_string(index=False))

    print("\n[与 docx 对比] 尾部相依系数变化:")
    print(pd.DataFrame(compare_rows).to_string(index=False))

    print(f"\n输出位置:")
    print(f"  表格: {TABLE_DIR}/07_*..10_*.csv, rolling_*.csv")
    print(f"  图:   {FIG_DIR}/09_*..13_*.png")


if __name__ == "__main__":
    main()
