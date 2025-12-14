# bayesian_staggered_did_thermometer
Analyzes "AI Thermometer" yield impact using Bayesian Staggered DiD (MCMC). Handles varied adoption timings and outputs improvement probabilities. ベイズ流Staggered DiDによるAI温度計の歩留まり検証ツール。導入時期のバラつきに対応し、MCMCで改善確率や信用区間を算出します。

# Bayesian Staggered DiD for AI Thermometer Analysis 🌡️

**Evaluating Manufacturing Yield with Time-Varying Treatments using Bayesian Gaussian Models.**

> 🇯🇵 **[Click here for Japanese Description / 日本語の説明はこちら](https://www.google.com/search?q=%23-japanese-description)**

## 📖 Overview

This project simulates and analyzes the causal impact of an **"AI Thermometer"** introduction on manufacturing yield across multiple production lines.

Unlike traditional A/B tests or simple Difference-in-Differences (DiD), this tool is designed for **Staggered Adoption** scenarios, where different lines introduce the technology at different times (Early Adopters vs. Late Adopters). By utilizing **Bayesian Inference (Bayesian OLS)**, it estimates the treatment effect with probabilistic uncertainty, providing **95% Credible Intervals** and the **Probability of Improvement** for robust decision-making.

## 🚀 Key Features

  * **Staggered Design Support**: Handles complex timelines with Early Adopters, Late Adopters, and Control groups.
  * **Bayesian Gaussian Modeling**: Uses MCMC sampling (`Bambi` & `PyMC`) to estimate continuous yield outcomes (equivalent to Bayesian OLS with fixed effects).
  * **Probabilistic Insights**: Outputs "Probability of Positive Effect" (e.g., "98% chance of yield improvement") rather than just p-values.
  * **Counterfactual Visualization**: Plots the actual yield against the predicted "what-if" scenario (if AI were never introduced) to visually demonstrate the impact.

## 📊 Methodology

The model estimates the causal impact ($\delta$) while controlling for common time trends and line-specific baselines:

$$Yield_{it} \sim \mathcal{N}(\mu_{it}, \sigma)$$
$$\mu_{it} = \alpha + \beta_{Time}(\text{Time}_{t}) + \beta_{Line}(\text{Line}_{i}) + \delta(\text{Intervention}_{it})$$

  * $Yield_{it}$: Yield score (0-100) of line $i$ at time $t$.
  * $\text{Intervention}_{it}$: Binary indicator (1 if AI is active, 0 otherwise).
  * $\delta$: **The Causal Estimator** (Impact of the AI Thermometer).

## 📂 File Structure

| File | Description |
| :--- | :--- |
| `staggered_data_generator.py` | **Data Generator**: Creates dummy yield data for multiple lines (Early/Late/Control) with trend and noise. |
| `bayesian_staggered_analyzer.py` | **Analysis Tool**: A GUI application to load data, run MCMC, and visualize results. |

## 🛠️ Usage

### 1\. Generate Data

Run the generator script to create a staggered dataset (`Staggered_Yield_Data.xlsx`).

```bash
python staggered_data_generator.py
```

### 2\. Run Analysis

Launch the GUI application.

```bash
python bayesian_staggered_analyzer.py
```

### 3\. Execution

1.  Click **"Browse"** and select `Staggered_Yield_Data.xlsx`.
2.  Click **"Run Bayesian Analysis"**.
3.  **Check Results**:
      * A summary popup will show the estimated effect size and probability.
      * Comparison plots (Actual vs. Counterfactual) will be saved in the same folder.

## 📦 Requirements

  * Python 3.10+
  * `bambi`
  * `pymc`
  * `arviz`
  * `pandas`
  * `matplotlib`
  * `tkinter` (Standard library)

To install dependencies:

```bash
pip install bambi pymc arviz pandas matplotlib
```

## 👤 Author

**Go Sato (Data Scientist)**

  * Specializing in Causal Inference and Bayesian Statistics for manufacturing process optimization.

-----

<br>

# 🇯🇵 Japanese Description

## 📖 概要

本プロジェクトは、製造ラインへの\*\*「AI温度計」\*\*導入が歩留まり（Yield）に与える因果効果を検証するシミュレーションおよび分析ツールです。

単純なA/Bテストや通常の差分の差分法（DiD）とは異なり、ラインごとに導入時期が異なる\*\*「Staggered（時間差）導入」\*\*のシナリオ（早期導入、後期導入など）に対応しています。\*\*ベイズ推論（Bayesian OLS）\*\*を用いることで、単なる点推定ではなく、「95%信用区間」や「改善の確信度（確率）」を算出し、不確実性を考慮した高度な意思決定を支援します。

## 🚀 主な特徴

  * **Staggered導入への対応**: 早期導入、後期導入、未導入（Control）が混在する複雑なタイムラインを適切に処理します。
  * **ベイズ流ガウス回帰**: MCMCサンプリング（`Bambi` & `PyMC`）を用い、固定効果を含むベイズ流OLSとして連続値の歩留まりを推定します。
  * **確率的な示唆**: 単なるp値ではなく、「98%の確率で歩留まりが向上している」といった、ビジネス判断に直結する指標を提供します。
  * **反実仮想（Counterfactual）の可視化**: 「もしAIを導入していなかったらどうなっていたか」を予測・プロットし、実際のデータと比較することで効果を直感的に示します。

## 📊 分析手法

モデルは、共通の時間トレンドとラインごとのベースラインを制御しつつ、介入効果（$\delta$）を推定します。

$$Yield_{it} \sim \mathcal{N}(\mu_{it}, \sigma)$$
$$\mu_{it} = \alpha + \beta_{Time}(\text{Time}_{t}) + \beta_{Line}(\text{Line}_{i}) + \delta(\text{Intervention}_{it})$$

  * $Yield_{it}$: 時点 $t$ におけるライン $i$ の歩留まりスコア (0-100)。
  * $\text{Intervention}_{it}$: 介入フラグ（AI導入済みなら1、そうでなければ0）。
  * $\delta$: **因果効果の推定量**（AI温度計によるインパクト）。

## 📂 ファイル構成

| ファイル名 | 説明 |
| :--- | :--- |
| `staggered_data_generator.py` | **データ生成**: 早期・後期・未導入ラインのダミー歩留まりデータを生成します。 |
| `bayesian_staggered_analyzer.py` | **分析ツール**: データの読み込み、MCMCの実行、結果の可視化を行うGUIアプリです。 |

## 🛠️ 使用方法

### 1\. データの準備

スクリプトを実行し、時間差導入のダミーデータ（Excel）を作成します。

```bash
python staggered_data_generator.py
```

実行後、`Staggered_Yield_Data.xlsx` が生成されます。

### 2\. ツールの起動

分析ツールを起動します。

```bash
python bayesian_staggered_analyzer.py
```

### 3\. 分析の実行

1.  **"Browse"** ボタンを押し、生成されたExcelファイルを選択します。
2.  **"Run Bayesian Analysis"** をクリックすると、MCMCサンプリングが開始されます。
3.  **結果の確認**:
      * 推定された効果量と改善確率がポップアップで表示されます。
      * 各ラインのプロット画像（実績値 vs 反実仮想）がフォルダに保存されます。

## 📦 必要ライブラリ

  * Python 3.10+
  * `bambi`
  * `pymc`
  * `arviz`
  * `pandas`
  * `matplotlib`
  * `tkinter` (標準ライブラリ)

インストールコマンド:

```bash
pip install bambi pymc arviz pandas matplotlib
```

## 👤 著者

**Go Sato (Data Scientist)**

  * 製造プロセスの最適化における因果推論とベイズ統計解析を専門としています。
