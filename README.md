# ETF Liquidity and Market Risk Linkage Under Tail Regimes

This repository contains a cleaned, reproducible version of a course group research project on nonlinear dependence between ETF liquidity risk and market risk in China. It studies six ETFs and their benchmark indices from 2014 to 2025 using GARCH-family marginal models and static, regime-specific, and rolling mixed Copulas.

## Highlights

- 17,762 daily ETF observations across six funds.
- GARCH, GJR-GARCH, Gaussian, Student-t, and skewed-t marginal specifications.
- Gaussian, Student-t, Gumbel, Clayton, Frank, and three-component mixed Copula comparisons.
- Downside tail-dependence estimates of 0.11–0.19 for broad-market A-share ETFs.
- Regime estimates approaching 0.30 during leveraged-bull and crash periods.
- A 250-day rolling-window analysis linking changes in dependence to major market events.

## Repository structure

```text
src/
  fetch_data.py
  step1_preprocess_describe.py
  step2_garch_marginal.py
  step3_mixed_copula.py
results/
  tables/
  figures/
requirements.txt
```

## Reproducing the analysis

Create a Python environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the scripts in order from the `src` directory:

```bash
cd src
python fetch_data.py
python step1_preprocess_describe.py
python step2_garch_marginal.py
python step3_mixed_copula.py
```

The data-download script uses public market-data interfaces exposed through AkShare. Availability and historical values may change with the upstream providers, so reproduced samples may differ slightly from the archived result tables.

## Selected findings

The three-component mixed Copula improved AIC by up to 55 points relative to the best single-Copula benchmark. For the five broad-market A-share ETFs, the Clayton component received most of the estimated mixture weight, indicating stronger downside than upside dependence. Dependence also varied materially across market regimes and rolling windows.

## Research limitations

- The China Concepts Internet ETF uses its own return as the market-risk proxy because a directly comparable public benchmark series was not used.
- Some liquidity-shock residuals retain serial correlation after an AR(1) mean specification.
- Results are historical research estimates and are not investment advice.

## Project note

This work originated as a course group project. The repository is a cleaned research artifact for portfolio review and omits documents containing group-member information.
