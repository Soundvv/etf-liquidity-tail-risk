"""
ETF流动性风险与市场风险研究 - 数据获取脚本
=================================================
功能:
    1. 用 akshare 下载 5 只 ETF (+ 1 只案例 ETF) 的日线 OHLCV 数据
    2. 下载 5 个对应指数的日线收盘数据
    3. 计算流动性风险指标 L_t (Amivest 流动性比率取对数)
    4. 计算市场风险指标 R_t (指数日收益率)
    5. 输出整洁的 CSV 文件，可直接用于 Python/R 建模

数据源: akshare (东方财富、新浪财经)
作者: 小组研究项目
"""

import os
import time
import numpy as np
import pandas as pd
import akshare as ak

# ============================================================
# 配置
# ============================================================
START_DATE = "20140101"
END_DATE = "20251231"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# (ETF代码, ETF名称, 对应指数代码, 指数名称)
ETF_INDEX_MAP = [
    ("510050", "上证50ETF",   "sh000016", "上证50"),
    ("510180", "上证180ETF",  "sh000010", "上证180"),
    ("510300", "沪深300ETF",  "sh000300", "沪深300"),
    ("510500", "中证500ETF",  "sh000905", "中证500"),
    ("159915", "创业板ETF",   "sz399006", "创业板指"),
    ("513050", "中概互联ETF", "sh000016", "上证50"),  # 案例: 中概互联无单一可比指数, 这里仅作占位
]

# ============================================================
# 下载函数
# ============================================================
def fetch_etf(symbol: str) -> pd.DataFrame:
    """下载单只 ETF 日线数据 (东方财富口径)"""
    df = ak.fund_etf_hist_em(
        symbol=symbol,
        period="daily",
        start_date=START_DATE,
        end_date=END_DATE,
        adjust="qfq",  # 前复权
    )
    df = df.rename(columns={
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "振幅": "amplitude",
        "涨跌幅": "pct_chg",
        "涨跌额": "change",
        "换手率": "turnover",
    })
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def fetch_index(symbol: str) -> pd.DataFrame:
    """下载单个指数日线数据 (新浪财经口径)"""
    df = ak.stock_zh_index_daily(symbol=symbol)
    df["date"] = pd.to_datetime(df["date"])
    mask = (df["date"] >= pd.to_datetime(START_DATE)) & (df["date"] <= pd.to_datetime(END_DATE))
    df = df.loc[mask].sort_values("date").reset_index(drop=True)
    return df


# ============================================================
# 风险指标计算
# ============================================================
def compute_liquidity_risk(df_etf: pd.DataFrame) -> pd.Series:
    """
    Amivest 流动性比率取对数:
        L_t = log( close * volume / (high - low) )
    指标越大 -> 流动性越好 -> 流动性风险越小
    成交量 volume 单位: 手 (1 手 = 100 股), 这里保留原值, 不影响时间序列性质
    """
    price_range = df_etf["high"] - df_etf["low"]
    price_range = price_range.replace(0, np.nan)  # 防止除零
    numerator = df_etf["close"] * df_etf["volume"]
    L = np.log(numerator / price_range)
    return L


def compute_market_risk(df_index: pd.DataFrame) -> pd.Series:
    """市场风险指标: 指数日收益率 R_t = (K_t - K_{t-1}) / K_{t-1}"""
    return df_index["close"].pct_change()


# ============================================================
# 主流程
# ============================================================
def main():
    summary_rows = []

    for etf_code, etf_name, idx_code, idx_name in ETF_INDEX_MAP:
        print(f"\n=== 处理: {etf_code} {etf_name}  (指数: {idx_code} {idx_name}) ===")

        try:
            df_etf = fetch_etf(etf_code)
            print(f"  ETF 原始数据: {len(df_etf)} 行, 区间 {df_etf['date'].min().date()} ~ {df_etf['date'].max().date()}")
        except Exception as e:
            print(f"  [跳过] ETF 下载失败: {e}")
            continue

        try:
            df_idx = fetch_index(idx_code)
            print(f"  指数原始数据: {len(df_idx)} 行, 区间 {df_idx['date'].min().date()} ~ {df_idx['date'].max().date()}")
        except Exception as e:
            print(f"  [跳过] 指数下载失败: {e}")
            continue

        # 单独保存原始数据
        df_etf.to_csv(os.path.join(OUT_DIR, f"raw_etf_{etf_code}.csv"), index=False, encoding="utf-8-sig")
        df_idx.to_csv(os.path.join(OUT_DIR, f"raw_index_{idx_code}.csv"), index=False, encoding="utf-8-sig")

        # 计算 L_t 和 R_t
        df_etf["L_t"] = compute_liquidity_risk(df_etf)
        df_idx_renamed = df_idx.rename(columns={"close": "index_close"})[["date", "index_close"]]
        df_idx_renamed["R_t"] = compute_market_risk(df_idx_renamed.rename(columns={"index_close": "close"}))

        # 合并 ETF 和指数
        merged = pd.merge(df_etf, df_idx_renamed, on="date", how="inner")
        merged = merged.dropna(subset=["L_t", "R_t"]).reset_index(drop=True)

        # 输出"建模就绪"的 CSV
        out_cols = ["date", "open", "high", "low", "close", "volume", "amount",
                    "L_t", "index_close", "R_t"]
        out_cols = [c for c in out_cols if c in merged.columns]
        out_path = os.path.join(OUT_DIR, f"ready_{etf_code}_{etf_name}.csv")
        merged[out_cols].to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"  -> 输出: {out_path}  ({len(merged)} 行)")

        # 描述性统计
        L_stat = merged["L_t"].describe()
        R_stat = merged["R_t"].describe()
        summary_rows.append({
            "ETF代码": etf_code,
            "ETF名称": etf_name,
            "指数代码": idx_code,
            "样本数": len(merged),
            "起始日": merged["date"].min().date(),
            "终止日": merged["date"].max().date(),
            "L_t均值": round(L_stat["mean"], 4),
            "L_t标准差": round(L_stat["std"], 4),
            "L_t偏度": round(merged["L_t"].skew(), 4),
            "L_t峰度": round(merged["L_t"].kurtosis(), 4),
            "R_t均值": round(R_stat["mean"], 6),
            "R_t标准差": round(R_stat["std"], 6),
            "R_t偏度": round(merged["R_t"].skew(), 4),
            "R_t峰度": round(merged["R_t"].kurtosis(), 4),
        })

        time.sleep(1)  # 避免请求过快被限流

    # 汇总表
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUT_DIR, "summary_descriptive_statistics.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\n汇总表已输出: {summary_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
