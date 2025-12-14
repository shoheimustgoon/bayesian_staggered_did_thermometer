# -*- coding: utf-8 -*-
"""
Staggered DiD Analysis Tool (Bayesian Version - v36 Simplified)
- Data Prep: Pandas
- Parallel Trend Test: Frequentist GLM/OLS (for speed check)
- DiD Modeling: Bayesian (Bambi/PyMC)
- Visualization: Matplotlib with 95% Probability Range

Updates v36 (User Request):
- Terminology: "95% HDI" -> "95%確率範囲"
- Metric: "p-value" -> "改善しなかった確率" (Prob of No Improvement)
- Headers: Simplified for non-statisticians
"""
from __future__ import annotations
import os
import sys

# PyTensor警告を抑制（g++なしでも動作）
os.environ['PYTENSOR_FLAGS'] = 'device=cpu,floatX=float64,optimizer=fast_compile,cxx='

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import logging
import traceback
from datetime import datetime
import warnings

# Bayesian Libraries
import bambi as bmb
import arviz as az
import pymc as pm

# ==========================================
# Settings & Constants
# ==========================================
plt.rcParams['font.family'] = 'MS Gothic'
warnings.simplefilter('ignore')

# ベイズ設定（グローバル変数 - GUIから変更可能）
BAYES_DRAWS = 500       
BAYES_TUNE = 300        
BAYES_CHAINS = 2        
BAYES_PROGRESSBAR = True

COLOR_TREATED_PRE   = '#56B4E9'
COLOR_TREATED_POST  = '#E69F00'
COLOR_CONTROL       = '#AAAAAA'
COLOR_MEDIAN        = '#000000'
COLOR_MEAN          = 'green'

KPI_SHEET_PREFIX = 'DiD_KPI_'
DENOMINATORS_SHEET_PREFIX = 'Denominators_'

# ==========================================
# Helper Functions
# ==========================================
def _extract_denominators(df_merged: pd.DataFrame, group_name: str) -> tuple | None:
    effective_den_cols = ['Treated_Effective_Denominator_Total', 'Treated_Effective_Denominator_Intro', 'Control_Effective_Denominator_Total']
    legacy_den_cols = ['Treated_Total_Active_PM', 'Treated_After_PM_Introduced', 'Control_Total_Active_PM']

    if all(col in df_merged.columns for col in effective_den_cols):
        logging.info(f"[Data] Group {group_name}: Using 'Effective Denominator'.")
        t_denom = df_merged['Treated_Effective_Denominator_Total']
        t_dose = df_merged['Treated_Effective_Denominator_Intro']
        c_denom = df_merged['Control_Effective_Denominator_Total']
    elif all(col in df_merged.columns for col in legacy_den_cols):
        logging.info(f"[Data] Group {group_name}: Using 'Total Active PM' (Legacy).")
        t_denom = df_merged['Treated_Total_Active_PM']
        t_dose = df_merged['Treated_After_PM_Introduced']
        c_denom = df_merged['Control_Total_Active_PM']
    else:
        logging.error(f"Skipping {group_name}: Missing denominator columns.")
        return None
        
    t_dose_mtbf = df_merged.get('Treated_Effective_Denominator_Intro', df_merged.get('Treated_After_PM_Introduced'))
    return t_denom, t_dose, c_denom, t_dose_mtbf

# ==========================================
# Data Preparation
# ==========================================
def prepare_data_cr(xls: pd.ExcelFile, group_name: str) -> pd.DataFrame | None:
    try:
        df_kpi = pd.read_excel(xls, sheet_name=f'{KPI_SHEET_PREFIX}{group_name}')
        df_kpi['Month'] = pd.to_datetime(df_kpi['Month'], errors='coerce')
        try:
            df_den = pd.read_excel(xls, sheet_name=f'{DENOMINATORS_SHEET_PREFIX}{group_name}')
            df_den['Month'] = pd.to_datetime(df_den['Month'], errors='coerce')
            df_merged = pd.merge(df_kpi, df_den, on='Month', how='inner', suffixes=('', '_den'))
        except:
            df_merged = df_kpi

        for col in ['Treated_Count_Intro', 'Treated_Count_NotIntro', 'Control_Count_Total']:
            if col in df_merged.columns: df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce')
        df_merged = df_merged.fillna(0)

        den_res = _extract_denominators(df_merged, group_name)
        if not den_res: return None
        t_denom, t_dose, c_denom, _ = den_res

        df_t = pd.DataFrame({
            'Month': df_merged['Month'], 'Treated': 1,
            'Count': df_merged['Treated_Count_Intro'] + df_merged['Treated_Count_NotIntro'],
            'Denominator': t_denom, 'Dose': t_dose
        })
        df_c = pd.DataFrame({
            'Month': df_merged['Month'], 'Treated': 0,
            'Count': df_merged['Control_Count_Total'],
            'Denominator': c_denom, 'Dose': 0
        })

        df_long = pd.concat([df_t, df_c], ignore_index=True)
        df_long['Month'] = pd.to_datetime(df_long['Month'])
        for col in ['Treated', 'Count', 'Denominator', 'Dose']: 
            df_long[col] = df_long[col].astype(float)
        
        df_long = df_long.sort_values(by=['Treated', 'Month']).reset_index(drop=True)
        df_long['Time'] = df_long.groupby('Treated').cumcount()
        
        try:
            first_treat = df_long.loc[df_long['Dose'] > 0, 'Month'].min()
            df_long['Post'] = (df_long['Month'] >= first_treat).astype(int) if not pd.isna(first_treat) else 0
        except:
            df_long['Post'] = 0
            
        post_map = df_long.loc[df_long['Treated'] == 1, ['Month', 'Post']].set_index('Month')['Post']
        df_long.loc[df_long['Treated'] == 0, 'Post'] = \
            df_long.loc[df_long['Treated'] == 0, 'Month'].map(post_map).ffill().fillna(0).astype(int)
            
        return df_long
    except Exception as e:
        logging.error(f"Data Prep CR Error: {e}")
        return None

def _prepare_data_continuous(xls, group_name, mode):
    try:
        df_kpi = pd.read_excel(xls, sheet_name=f'{KPI_SHEET_PREFIX}{group_name}')
        df_kpi['Month'] = pd.to_datetime(df_kpi['Month'], errors='coerce')
        try:
            df_den = pd.read_excel(xls, sheet_name=f'{DENOMINATORS_SHEET_PREFIX}{group_name}')
            df_den['Month'] = pd.to_datetime(df_den['Month'], errors='coerce')
            df_merged = pd.merge(df_kpi, df_den, on='Month', how='inner', suffixes=('', '_den'))
        except:
            df_merged = df_kpi

        if mode == 'MTBF':
            t_hours = df_merged.get('Treated_Norm_MTBF_hours_Intro', 0) + df_merged.get('Treated_Norm_MTBF_hours_NotIntro', 0)
            c_hours = df_merged.get('Control_Norm_MTBF_hours_Total', 0)
            if isinstance(t_hours, int): 
                t_hours = df_merged['Treated_MTBF_hours_Intro'] + df_merged['Treated_MTBF_hours_NotIntro']
                c_hours = df_merged['Control_MTBF_hours_Total']
            t_cnt = df_merged['Treated_MTBF_Count_Intro'] + df_merged['Treated_MTBF_Count_NotIntro']
            c_cnt = df_merged['Control_MTBF_Count_Total']
            target_t, target_c = 'Treated_MTBF', 'Control_MTBF'
        elif mode == 'RF':
            t_hours = df_merged['Treated_RF_Hours_Intro'] + df_merged['Treated_RF_Hours_NotIntro']
            c_hours = df_merged['Control_RF_Hours_Total']
            t_cnt = df_merged['Treated_Count_Intro'] + df_merged['Treated_Count_NotIntro']
            c_cnt = df_merged['Control_Count_Total']
            target_t, target_c = 'Treated_RF_MTBF', 'Control_RF_MTBF'
        elif mode == 'WBF':
            t_hours = df_merged['Treated_Wafer_Count_Intro'] + df_merged['Treated_Wafer_Count_NotIntro']
            c_hours = df_merged['Control_Wafer_Count_Total']
            t_cnt = df_merged['Treated_Count_Intro'] + df_merged['Treated_Count_NotIntro']
            c_cnt = df_merged['Control_Count_Total']
            target_t, target_c = 'Treated_WBF', 'Control_WBF'
        else: return None

        df_merged[target_t] = np.where(t_cnt > 0, t_hours / t_cnt, np.nan)
        df_merged[target_c] = np.where(c_cnt > 0, c_hours / c_cnt, np.nan)

        den_res = _extract_denominators(df_merged, group_name)
        if not den_res: return None
        _, _, _, t_dose = den_res
        
        val_key = mode if mode != 'RF' else 'RF_MTBF'
        
        df_t = pd.DataFrame({'Month': df_merged['Month'], 'Treated': 1, val_key: df_merged[target_t], 'Dose': t_dose})
        df_c = pd.DataFrame({'Month': df_merged['Month'], 'Treated': 0, val_key: df_merged[target_c], 'Dose': 0})
        
        df_long = pd.concat([df_t, df_c], ignore_index=True)
        df_long['Month'] = pd.to_datetime(df_long['Month'])
        df_long = df_long.dropna(subset=[val_key]).sort_values(by=['Treated', 'Month']).reset_index(drop=True)
        
        for col in ['Treated', val_key, 'Dose']: df_long[col] = df_long[col].astype(float)
        
        df_long['Time'] = df_long.groupby('Treated').cumcount()
        
        try:
            first_treat = df_long.loc[df_long['Dose'] > 0, 'Month'].min()
            df_long['Post'] = (df_long['Month'] >= first_treat).astype(int) if not pd.isna(first_treat) else 0
        except: df_long['Post'] = 0
        
        post_map = df_long.loc[df_long['Treated'] == 1, ['Month', 'Post']].set_index('Month')['Post']
        df_long.loc[df_long['Treated'] == 0, 'Post'] = \
            df_long.loc[df_long['Treated'] == 0, 'Month'].map(post_map).ffill().fillna(0).astype(int)
            
        return df_long
    except Exception as e:
        logging.error(f"Data Prep {mode} Error: {e}")
        return None

def prepare_data_mtbf(xls, g): return _prepare_data_continuous(xls, g, 'MTBF')
def prepare_data_rf(xls, g): return _prepare_data_continuous(xls, g, 'RF')
def prepare_data_wbf(xls, g): return _prepare_data_continuous(xls, g, 'WBF')

# ==========================================
# Parallel Trend Test (Frequentist)
# ==========================================
def run_parallel_trend_test_cr(df_long):
    df_pre = df_long[df_long['Post'] == 0].dropna(subset=['Denominator'])
    df_pre = df_pre[df_pre['Denominator'] > 0]
    if len(df_pre['Treated'].unique()) < 2 or len(df_pre) < 10:
        return {'Passed/Failed': "Failed", 'p': 1.0, 'Reason': 'Not enough data'}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = smf.glm("Count ~ Time + Treated + Time:Treated", data=df_pre,
                            family=sm.families.NegativeBinomial(), 
                            offset=np.log(df_pre['Denominator'])).fit()
        p_val = model.pvalues.get('Time:Treated', 1.0)
        return {'Passed/Failed': "Passed" if p_val >= 0.10 else "Failed",
                'p': p_val, 'Model': 'GLM-NB(Freq)'}
    except Exception as e:
        return {'Passed/Failed': "Failed", 'p': 1.0, 'Reason': str(e)}

def run_parallel_trend_test_ols(df_long, target_var):
    df_pre = df_long[df_long['Post'] == 0].dropna(subset=[target_var])
    if len(df_pre['Treated'].unique()) < 2 or len(df_pre) < 6: 
        return {'Passed/Failed': "Failed", 'p': 1.0, 'Reason': 'Not enough data'}
    try:
        model = smf.ols(f"{target_var} ~ Time + Treated + Time:Treated", data=df_pre).fit()
        p_val = model.pvalues.get('Time:Treated', 1.0)
        return {'Passed/Failed': "Passed" if p_val >= 0.10 else "Failed",
                'p': p_val, 'Model': 'OLS(Freq)'}
    except Exception as e:
        return {'Passed/Failed': "Failed", 'p': 1.0, 'Reason': str(e)}

# ==========================================
# Bayesian Models (Bambi)
# ==========================================
def run_did_model_bayes_cr(df_long: pd.DataFrame):
    df_full = df_long[df_long['Denominator'] > 0].dropna().copy()
    if df_full.empty: return pd.Series({'Error': "Empty"}), None, df_full
    
    df_full['log_denom'] = np.log(df_full['Denominator'])
    try:
        print(f"  → Bayesian CR: draws={BAYES_DRAWS}, tune={BAYES_TUNE}, chains={BAYES_CHAINS}")
        model = bmb.Model("Count ~ Time + Treated + Dose + offset(log_denom)", 
                          data=df_full, family="negativebinomial", link="log")
        idata = model.fit(draws=BAYES_DRAWS, tune=BAYES_TUNE, chains=BAYES_CHAINS, 
                         random_seed=42, progressbar=BAYES_PROGRESSBAR)
        
        summary = az.summary(idata, var_names=['Dose'], hdi_prob=0.95)
        beta = summary.loc['Dose', 'mean']
        hdi_lower = summary.loc['Dose', 'hdi_2.5%']
        hdi_upper = summary.loc['Dose', 'hdi_97.5%']
        
        # 改善しなかった確率 (Count RateはRR < 1なら改善 => Beta < 0なら改善)
        # Beta > 0 (悪化) である確率を計算
        post_beta = idata.posterior['Dose'].values.flatten()
        prob_no_imp = (post_beta > 0).mean()

        stats = pd.Series({
            'Beta_Dose (logRR)': beta,
            'RR (Risk Ratio)': np.exp(beta),
            'Range_Lower': np.exp(hdi_lower),
            'Range_Upper': np.exp(hdi_upper),
            'Model': 'Bayesian-NB',
            'Prob_No_Improvement': prob_no_imp,
            'SE (logRR)': summary.loc['Dose', 'sd']
        })
        return stats, (model, idata), df_full
    except Exception as e:
        logging.error(f"Bayes CR Error: {e}")
        return pd.Series({'Error': str(e)}), None, df_full

def generate_predictions_bayes_cr(df_full, model_pack):
    if model_pack is None: return df_full
    model, idata = model_pack
    df = df_full.copy()
    
    # 1. CF Treated
    df_cf = df.copy(); df_cf.loc[df_cf['Treated']==1, 'Dose'] = 0
    model.predict(idata, data=df_cf, kind='mean', inplace=True)
    
    resp_name = "Count"
    post_mean_key = f"{resp_name}_mean"
    if post_mean_key not in idata.posterior and 'mu' in idata.posterior:
        post_mean_key = 'mu'
        
    post_mean = idata.posterior[post_mean_key]
    
    df['Pred_Count_CF_Treated'] = post_mean.mean(dim=["chain", "draw"]).values
    df['Pred_Count_CF_Treated_Lower'] = post_mean.quantile(0.025, dim=["chain", "draw"]).values
    df['Pred_Count_CF_Treated_Upper'] = post_mean.quantile(0.975, dim=["chain", "draw"]).values

    # 2. Hypo Control
    df_hypo = df.copy()
    dose_map = df[df['Treated']==1].set_index('Time')['Dose']
    df_hypo.loc[df_hypo['Treated']==0, 'Dose'] = df_hypo.loc[df_hypo['Treated']==0, 'Time'].map(dose_map).fillna(0)
    model.predict(idata, data=df_hypo, kind='mean', inplace=True)
    post_mean_h = idata.posterior[post_mean_key]
    
    df['Pred_Count_Hypo_Control'] = post_mean_h.mean(dim=["chain", "draw"]).values
    df['Pred_Count_Hypo_Control_Lower'] = post_mean_h.quantile(0.025, dim=["chain", "draw"]).values
    df['Pred_Count_Hypo_Control_Upper'] = post_mean_h.quantile(0.975, dim=["chain", "draw"]).values
    
    # Rate変換 (名前固定)
    df['Rate_CF_Treated'] = df['Pred_Count_CF_Treated'] / df['Denominator']
    df['Rate_Hypo_Control'] = df['Pred_Count_Hypo_Control'] / df['Denominator']
    df['Rate_CF_Treated_Lower'] = df['Pred_Count_CF_Treated_Lower'] / df['Denominator']
    df['Rate_CF_Treated_Upper'] = df['Pred_Count_CF_Treated_Upper'] / df['Denominator']
    df['Rate_Hypo_Control_Lower'] = df['Pred_Count_Hypo_Control_Lower'] / df['Denominator']
    df['Rate_Hypo_Control_Upper'] = df['Pred_Count_Hypo_Control_Upper'] / df['Denominator']
    
    df['Rate_Actual'] = df['Count'] / df['Denominator']
    return df

def run_did_model_bayes_ols(df_long: pd.DataFrame, target_var: str, log_prefix: str):
    df_full = df_long.dropna(subset=[target_var, 'Dose']).copy()
    if df_full.empty: return pd.Series({'Error': "Empty"}), None, df_full
    
    try:
        print(f"  → Bayesian {log_prefix}: draws={BAYES_DRAWS}, tune={BAYES_TUNE}, chains={BAYES_CHAINS}")
        model = bmb.Model(f"{target_var} ~ Time + Treated + Dose", data=df_full, family="gaussian")
        idata = model.fit(draws=BAYES_DRAWS, tune=BAYES_TUNE, chains=BAYES_CHAINS, 
                         random_seed=42, progressbar=BAYES_PROGRESSBAR)
        
        summary = az.summary(idata, var_names=['Dose'], hdi_prob=0.95)
        beta = summary.loc['Dose', 'mean']
        hdi_l = summary.loc['Dose', 'hdi_2.5%']
        hdi_u = summary.loc['Dose', 'hdi_97.5%']

        # 改善しなかった確率 (MTBF/RF/WBFは値が大きいほど改善 => Beta > 0なら改善)
        # Beta < 0 (悪化) である確率を計算
        post_beta = idata.posterior['Dose'].values.flatten()
        prob_no_imp = (post_beta < 0).mean()
        
        stats = pd.Series({
            f'Beta_Dose ({log_prefix} Effect)': beta,
            'Range_Lower': hdi_l, 
            'Range_Upper': hdi_u,
            'Model': 'Bayesian-Gaussian',
            'Prob_No_Improvement': prob_no_imp,
            'SE': summary.loc['Dose', 'sd']
        })
        return stats, (model, idata), df_full
    except Exception as e:
        logging.error(f"Bayes OLS Error: {e}")
        return pd.Series({'Error': str(e)}), None, df_full

def generate_predictions_bayes_ols(df_full, model_pack, target_var, pred_prefix):
    if model_pack is None: return df_full
    model, idata = model_pack
    df = df_full.copy()
    
    # 1. CF
    df_cf = df.copy()
    df_cf.loc[df_cf['Treated']==1, 'Dose'] = 0
    model.predict(idata, data=df_cf, kind='mean', inplace=True)
    
    var_name = f"{target_var}_mean"
    if var_name not in idata.posterior:
        if 'mu' in idata.posterior:
            var_name = 'mu'
        else:
            available = list(idata.posterior.data_vars.keys())
            logging.warning(f"Variable '{target_var}_mean' not found. Available: {available}")
            for v in available:
                if '_mean' in v or v == 'mu':
                    var_name = v
                    break
    
    pm_v = idata.posterior[var_name]
    df[f'Pred_{pred_prefix}_CF_Treated'] = pm_v.mean(dim=["chain", "draw"]).values
    df[f'Pred_{pred_prefix}_CF_Treated_Lower'] = pm_v.quantile(0.025, dim=["chain", "draw"]).values
    df[f'Pred_{pred_prefix}_CF_Treated_Upper'] = pm_v.quantile(0.975, dim=["chain", "draw"]).values
    
    # 2. Hypo
    df_hypo = df.copy()
    dose_map = df[df['Treated']==1].set_index('Time')['Dose']
    df_hypo.loc[df_hypo['Treated']==0, 'Dose'] = df_hypo.loc[df_hypo['Treated']==0, 'Time'].map(dose_map).fillna(0)
    model.predict(idata, data=df_hypo, kind='mean', inplace=True)
    pm_h = idata.posterior[var_name]
    df[f'Pred_{pred_prefix}_Hypo_Control'] = pm_h.mean(dim=["chain", "draw"]).values
    df[f'Pred_{pred_prefix}_Hypo_Control_Lower'] = pm_h.quantile(0.025, dim=["chain", "draw"]).values
    df[f'Pred_{pred_prefix}_Hypo_Control_Upper'] = pm_h.quantile(0.975, dim=["chain", "draw"]).values
    
    df[f'{pred_prefix}_Actual'] = df[target_var]
    return df

# ==========================================
# Visualization
# ==========================================
def _plot_did_generic(df_plot, stats, pt_passed, save_path, group_name, y_col, pred_col, y_label, title_metric, is_treated, is_cr, metric_key=None):
    fig, ax = plt.subplots(figsize=(14, 8), dpi=120)
    target_val = 1 if is_treated else 0
    df_sub = df_plot[df_plot['Treated'] == target_val].set_index('Month').copy()
    
    pred_vals = df_sub[pred_col].copy()
    pred_vals.loc[df_sub['Post'] == 0] = np.nan
    
    ci_l, ci_u = f"{pred_col}_Lower", f"{pred_col}_Upper"
    has_ci = (ci_l in df_sub.columns and ci_u in df_sub.columns)
    
    cf_color = COLOR_TREATED_POST if pt_passed == "Passed" else COLOR_CONTROL
    lbl_act = '実績値 (導入)' if is_treated else '実績値 (未導入)'
    lbl_pred = '予測値 (もし未導入なら)' if is_treated else '予測値 (もし導入なら)'
    
    ax.plot(df_sub.index, df_sub[y_col], color='#0000FF', label=lbl_act, marker='o', markersize=4)
    ax.plot(df_sub.index, pred_vals, color=cf_color, label=lbl_pred, linestyle='-', marker='x', markersize=4)
    
    if has_ci:
        l_v = df_sub[ci_l].copy(); l_v.loc[df_sub['Post']==0] = np.nan
        u_v = df_sub[ci_u].copy(); u_v.loc[df_sub['Post']==0] = np.nan
        # ラベル修正: 95% HDI -> 95%確率範囲
        ax.fill_between(df_sub.index, l_v, u_v, color=cf_color, alpha=0.15, label='95%確率範囲')

    # Fill
    if is_treated:
        good = (df_sub[pred_col] >= df_sub[y_col]) if is_cr else (df_sub[y_col] >= df_sub[pred_col])
        ax.fill_between(df_sub.index, df_sub[y_col], df_sub[pred_col], where=good & (df_sub['Post']==1), color='green', alpha=0.3, label='改善', interpolate=True)
        ax.fill_between(df_sub.index, df_sub[y_col], df_sub[pred_col], where=(~good) & (df_sub['Post']==1), color='red', alpha=0.3, label='悪化', interpolate=True)
    else:
        good = (df_sub[y_col] >= df_sub[pred_col]) if is_cr else (df_sub[pred_col] >= df_sub[y_col])
        ax.fill_between(df_sub.index, df_sub[y_col], df_sub[pred_col], where=good & (df_sub['Post']==1), color='green', alpha=0.3, label='想定改善', interpolate=True)
        ax.fill_between(df_sub.index, df_sub[y_col], df_sub[pred_col], where=(~good) & (df_sub['Post']==1), color='red', alpha=0.3, label='想定悪化', interpolate=True)

    # Text
    txt = f"Group: {group_name}\nEffect: "
    if is_cr: txt += f"RR={stats['RR (Risk Ratio)']:.3f} "
    else:
        val = stats.get(f'Beta_Dose ({metric_key} Effect)', 0)
        txt += f"Coeff={val:.3f} "
    
    # 表示修正: p(HDI) -> 改善しなかった確率
    prob = stats.get('Prob_No_Improvement', 1.0)
    txt += f"\n改善しなかった確率={prob:.1%}"
    if prob < 0.05:
        txt += "\n(高い確度で改善)"
        
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    ax.set_title(f"DiD (Bayes) {title_metric}: {group_name}")
    ax.set_ylabel(y_label)
    ax.legend(loc='upper right')
    ax.grid(True, linestyle=':')
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

def plot_did_treated_cr(df, m, s, p, path, g): _plot_did_generic(df, s, p, path, g, 'Rate_Actual', 'Rate_CF_Treated', 'Rate', 'CountRate', True, True)
def plot_did_control_cr(df, m, s, p, path, g): _plot_did_generic(df, s, p, path, g, 'Rate_Actual', 'Rate_Hypo_Control', 'Rate', 'CountRate', False, True)
def plot_did_treated_mtbf(df, m, s, p, path, g): _plot_did_generic(df, s, p, path, g, 'MTBF_Actual', 'Pred_MTBF_CF_Treated', 'MTBF(h)', 'MTBF', True, False, 'MTBF')
def plot_did_control_mtbf(df, m, s, p, path, g): _plot_did_generic(df, s, p, path, g, 'MTBF_Actual', 'Pred_MTBF_Hypo_Control', 'MTBF(h)', 'MTBF', False, False, 'MTBF')
def plot_did_treated_rf(df, m, s, p, path, g): _plot_did_generic(df, s, p, path, g, 'RF_MTBF_Actual', 'Pred_RF_MTBF_CF_Treated', 'RF-MTBF(h)', 'RF_MTBF', True, False, 'RF-MTBF')
def plot_did_control_rf(df, m, s, p, path, g): _plot_did_generic(df, s, p, path, g, 'RF_MTBF_Actual', 'Pred_RF_MTBF_Hypo_Control', 'RF-MTBF(h)', 'RF_MTBF', False, False, 'RF-MTBF')
def plot_did_treated_wbf(df, m, s, p, path, g): _plot_did_generic(df, s, p, path, g, 'WBF_Actual', 'Pred_WBF_CF_Treated', 'WBF(w)', 'WBF', True, False, 'WBF')
def plot_did_control_wbf(df, m, s, p, path, g): _plot_did_generic(df, s, p, path, g, 'WBF_Actual', 'Pred_WBF_Hypo_Control', 'WBF(w)', 'WBF', False, False, 'WBF')

# ==========================================
# Unified Boxplot
# ==========================================
def _draw_styled_boxplot(ax, data_list, labels, colors, title, ylabel):
    valid_data = []
    valid_labels = []
    valid_colors = []
    for d, l, c in zip(data_list, labels, colors):
        arr = np.array(d).flatten()
        arr = arr[~np.isnan(arr)]
        if arr.size > 0:
            valid_data.append(arr)
            valid_labels.append(l)
            valid_colors.append(c)
    
    if not valid_data:
        ax.set_title(title + "\n(データなし)")
        return
    
    bp = ax.boxplot(valid_data, labels=valid_labels, patch_artist=True,
                    showmeans=True, meanline=False, showfliers=False)
    
    for patch, c in zip(bp['boxes'], valid_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    
    for med in bp['medians']:
        med.set_color(COLOR_MEDIAN)
        med.set_linewidth(2)
    
    for mean_marker in bp['means']:
        mean_marker.set_marker('^')
        mean_marker.set_markersize(8)
        mean_marker.set_markeredgecolor('black')
        mean_marker.set_markerfacecolor(COLOR_MEAN)

    for whisker in bp['whiskers']:
        whisker.set_color('black')
        whisker.set_linewidth(1)
    for cap in bp['caps']:
        cap.set_color('black')
        cap.set_linewidth(1)

    for i, (vals, c) in enumerate(zip(valid_data, valid_colors)):
        y = vals
        x = np.random.normal(i + 1, 0.04, size=len(y))
        ax.plot(x, y, marker='o', linestyle='None', color='k', markersize=4,
                markerfacecolor=c, markeredgecolor='k', markeredgewidth=0.5, alpha=0.6)

    stats_lines = []
    for lab, vals in zip(valid_labels, valid_data):
        if len(vals) > 0:
            avg = np.mean(vals)
            med = np.median(vals)
            stats_lines.append(f"{lab}: n={len(vals)}, 平均値={avg:.3f}, 中央値={med:.3f}")
        else:
            stats_lines.append(f"{lab}: n=0")
    
    txt = "\n".join(stats_lines)
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, fontsize=8, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor='#CCCCCC'))

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', linestyle=':', alpha=0.4)

def plot_boxplot_cr(df_pred, save_path, group_name):
    _plot_boxplot_generic(df_pred, save_path, group_name, 'Rate_Actual', 'Rate_CF_Treated', 'Rate_Hypo_Control', 'Count Rate', 'レート')

def plot_boxplot_treated_mtbf(df, p, g): _plot_boxplot_generic(df, p, g, 'MTBF_Actual', 'Pred_MTBF_CF_Treated', '', 'MTBF', '時間', True)
def plot_boxplot_control_mtbf(df, p, g): _plot_boxplot_generic(df, p, g, 'MTBF_Actual', '', 'Pred_MTBF_Hypo_Control', 'MTBF', '時間', False)
def plot_boxplot_treated_rf(df, p, g): _plot_boxplot_generic(df, p, g, 'RF_MTBF_Actual', 'Pred_RF_MTBF_CF_Treated', '', 'RF-MTBF', '時間', True)
def plot_boxplot_control_rf(df, p, g): _plot_boxplot_generic(df, p, g, 'RF_MTBF_Actual', '', 'Pred_RF_MTBF_Hypo_Control', 'RF-MTBF', '時間', False)
def plot_boxplot_treated_wbf(df, p, g): _plot_boxplot_generic(df, p, g, 'WBF_Actual', 'Pred_WBF_CF_Treated', '', 'WBF', '枚数', True)
def plot_boxplot_control_wbf(df, p, g): _plot_boxplot_generic(df, p, g, 'WBF_Actual', '', 'Pred_WBF_Hypo_Control', 'WBF', '枚数', False)

def _plot_boxplot_generic(df_pred, save_path, group_name, act_col, pred_cf_t, pred_hypo_c, metric, unit, is_treated=None):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
    
    # CR (Dual)
    if is_treated is None:
        plt.close(fig)
        fig, (ax_t, ax_c) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Treated
        df_t = df_pred[df_pred['Treated'] == 1]
        data_t = [df_t[df_t['Post']==0][act_col], df_t[df_t['Post']==1][act_col], df_t[df_t['Post']==1][pred_cf_t]]
        _draw_styled_boxplot(ax_t, data_t, ['導入前(実績)', '導入後(実績)', '導入後(予測)'], [COLOR_TREATED_PRE, COLOR_TREATED_POST, COLOR_CONTROL], f"{metric} (Treated): {group_name}", unit)
        
        # Control
        df_c = df_pred[df_pred['Treated'] == 0]
        data_c = [df_c[df_c['Post']==0][act_col], df_c[df_c['Post']==1][act_col], df_c[df_c['Post']==1][pred_hypo_c]]
        _draw_styled_boxplot(ax_c, data_c, ['未導入(実績)', '未導入(実績)', '未導入(仮想予測)'], [COLOR_CONTROL, COLOR_CONTROL, COLOR_TREATED_POST], f"{metric} (Control): {group_name}", unit)
        
        fig.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)
        return

    # Single
    target = 1 if is_treated else 0
    df = df_pred[df_pred['Treated'] == target]
    pred_col = pred_cf_t if is_treated else pred_hypo_c
    
    data = [
        df[df['Post'] == 0][act_col].dropna().values,
        df[df['Post'] == 1][act_col].dropna().values,
        df[df['Post'] == 1][pred_col].dropna().values
    ]
    labels = ['導入前(実績)', '導入後(実績)', '導入後(予測)'] if is_treated else ['未導入(実績)', '未導入(実績)', '未導入(仮想予測)']
    colors = [COLOR_TREATED_PRE, COLOR_TREATED_POST, COLOR_CONTROL] if is_treated else [COLOR_CONTROL, COLOR_CONTROL, COLOR_TREATED_POST]
    
    _draw_styled_boxplot(ax, data, labels, colors, f"{metric} Boxplot: {group_name}", unit)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

# ==========================================
# Main App
# ==========================================
class DidAnalysisBayesianApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Bayesian DiD Analysis Tool v36 Simplified")
        self.root.geometry("700x450")
        self.excel_path = tk.StringVar()
        self.save_dir = tk.StringVar()
        
        self.draws_var = tk.IntVar(value=500)
        self.tune_var = tk.IntVar(value=300)
        self.chains_var = tk.IntVar(value=2)
        
        ttk.Label(root, text="1. RF_Generator_error出力 Excelファイルを選択:").pack(pady=(10,0), anchor='w', padx=20)
        f_frame = ttk.Frame(root); f_frame.pack(fill='x', padx=20)
        ttk.Entry(f_frame, textvariable=self.excel_path).pack(side='left', fill='x', expand=True)
        ttk.Button(f_frame, text="参照", command=self.browse_file).pack(side='left', padx=5)
        
        ttk.Label(root, text="2. 出力先フォルダを選択:").pack(pady=(10,0), anchor='w', padx=20)
        d_frame = ttk.Frame(root); d_frame.pack(fill='x', padx=20)
        ttk.Entry(d_frame, textvariable=self.save_dir).pack(side='left', fill='x', expand=True)
        ttk.Button(d_frame, text="参照", command=self.browse_dir).pack(side='left', padx=5)
        
        settings_frame = ttk.LabelFrame(root, text="ベイズ設定（値を小さくすると高速）", padding=10)
        settings_frame.pack(fill='x', padx=20, pady=10)
        
        # サンプル数
        ttk.Label(settings_frame, text="サンプル数:").grid(row=0, column=0, sticky='w', padx=(0,5))
        ttk.Entry(settings_frame, textvariable=self.draws_var, width=8).grid(row=0, column=1, padx=5)
        ttk.Label(settings_frame, text="(推奨: 500〜1000)").grid(row=0, column=2, sticky='w', padx=(5,20))
        
        # チューニング数
        ttk.Label(settings_frame, text="チューニング数:").grid(row=1, column=0, sticky='w', padx=(0,5), pady=(5,0))
        ttk.Entry(settings_frame, textvariable=self.tune_var, width=8).grid(row=1, column=1, padx=5, pady=(5,0))
        ttk.Label(settings_frame, text="(推奨: 300〜500)").grid(row=1, column=2, sticky='w', padx=(5,20), pady=(5,0))
        
        # チェーン数
        ttk.Label(settings_frame, text="チェーン数:").grid(row=2, column=0, sticky='w', padx=(0,5), pady=(5,0))
        ttk.Entry(settings_frame, textvariable=self.chains_var, width=8).grid(row=2, column=1, padx=5, pady=(5,0))
        ttk.Label(settings_frame, text="(推奨: 2)").grid(row=2, column=2, sticky='w', padx=(5,20), pady=(5,0))
        
        # 説明テキスト
        ttk.Label(root, text="※ 処理に時間がかかります。コンソール(黒い画面)で進捗を確認できます。", 
                 foreground='blue').pack(pady=(5,0))
        
        self.progress_label = ttk.Label(root, text="")
        self.progress_label.pack(pady=5)
        
        self.btn = ttk.Button(root, text="ベイズDiD解析開始", command=self.run)
        self.btn.pack(pady=10, ipadx=20, ipady=5)

    def browse_file(self):
        f = filedialog.askopenfilename(filetypes=[("Excel", "*.xlsx")])
        if f: 
            self.excel_path.set(f)
            if not self.save_dir.get(): self.save_dir.set(os.path.dirname(f))
    def browse_dir(self):
        d = filedialog.askdirectory()
        if d: self.save_dir.set(d)

    def run(self):
        global BAYES_DRAWS, BAYES_TUNE, BAYES_CHAINS
        ep, sd = self.excel_path.get(), self.save_dir.get()
        if not ep or not sd: return
        
        BAYES_DRAWS = self.draws_var.get()
        BAYES_TUNE = self.tune_var.get()
        BAYES_CHAINS = self.chains_var.get()
        
        self.btn.config(state='disabled')
        self.root.update()
        
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_f = os.path.join(sd, f'log_{ts}.txt')
        logging.basicConfig(filename=log_f, level=logging.INFO, force=True)
        
        try:
            xls = pd.ExcelFile(ep)
            groups = [s.replace(KPI_SHEET_PREFIX, '') for s in xls.sheet_names if s.startswith(KPI_SHEET_PREFIX)]
            out_f = os.path.join(sd, f'DiD_Results_Bayes_{ts}.xlsx')
            
            res_cr, res_mtbf, res_rf, res_wbf = [], [], [], []
            pt_cr, pt_mtbf, pt_rf, pt_wbf = [], [], [], []

            with pd.ExcelWriter(out_f) as writer:
                total_groups = len(groups)
                for idx, g in enumerate(groups):
                    print(f"\n=== Processing {idx+1}/{total_groups}: {g} ===")
                    self.progress_label.config(text=f'処理中: {g} ({idx+1}/{total_groups})')
                    self.root.update()
                    
                    # 1. CR
                    df = prepare_data_cr(xls, g)
                    if df is not None:
                        pt = run_parallel_trend_test_cr(df); pt['Group'] = g; pt_cr.append(pt)
                        st, m_pack, full = run_did_model_bayes_cr(df)
                        if m_pack:
                            st['Group'] = g; res_cr.append(st)
                            pred = generate_predictions_bayes_cr(full, m_pack)
                            plot_did_treated_cr(pred, m_pack, st, pt.get('Passed/Failed'), os.path.join(sd, f'CR_T_{g}.png'), g)
                            plot_did_control_cr(pred, m_pack, st, pt.get('Passed/Failed'), os.path.join(sd, f'CR_C_{g}.png'), g)
                            plot_boxplot_cr(pred, os.path.join(sd, f'Box_CR_{g}.png'), g)

                    # 2. MTBF
                    df = prepare_data_mtbf(xls, g)
                    if df is not None:
                        pt = run_parallel_trend_test_ols(df, 'MTBF'); pt['Group'] = g; pt_mtbf.append(pt)
                        st, m_pack, full = run_did_model_bayes_ols(df, 'MTBF', 'MTBF')
                        if m_pack:
                            st['Group'] = g; res_mtbf.append(st)
                            pred = generate_predictions_bayes_ols(full, m_pack, 'MTBF', 'MTBF')
                            plot_did_treated_mtbf(pred, m_pack, st, pt.get('Passed/Failed'), os.path.join(sd, f'MTBF_T_{g}.png'), g)
                            plot_did_control_mtbf(pred, m_pack, st, pt.get('Passed/Failed'), os.path.join(sd, f'MTBF_C_{g}.png'), g)
                            plot_boxplot_treated_mtbf(pred, os.path.join(sd, f'Box_MTBF_T_{g}.png'), g)
                            plot_boxplot_control_mtbf(pred, os.path.join(sd, f'Box_MTBF_C_{g}.png'), g)

                    # 3. RF
                    df = prepare_data_rf(xls, g)
                    if df is not None:
                        pt = run_parallel_trend_test_ols(df, 'RF_MTBF'); pt['Group'] = g; pt_rf.append(pt)
                        st, m_pack, full = run_did_model_bayes_ols(df, 'RF_MTBF', 'RF-MTBF')
                        if m_pack:
                            st['Group'] = g; res_rf.append(st)
                            pred = generate_predictions_bayes_ols(full, m_pack, 'RF_MTBF', 'RF_MTBF')
                            plot_did_treated_rf(pred, m_pack, st, pt.get('Passed/Failed'), os.path.join(sd, f'RF_T_{g}.png'), g)
                            plot_did_control_rf(pred, m_pack, st, pt.get('Passed/Failed'), os.path.join(sd, f'RF_C_{g}.png'), g)
                            plot_boxplot_treated_rf(pred, os.path.join(sd, f'Box_RF_T_{g}.png'), g)
                            plot_boxplot_control_rf(pred, os.path.join(sd, f'Box_RF_C_{g}.png'), g)

                    # 4. WBF
                    df = prepare_data_wbf(xls, g)
                    if df is not None:
                        pt = run_parallel_trend_test_ols(df, 'WBF'); pt['Group'] = g; pt_wbf.append(pt)
                        st, m_pack, full = run_did_model_bayes_ols(df, 'WBF', 'WBF')
                        if m_pack:
                            st['Group'] = g; res_wbf.append(st)
                            pred = generate_predictions_bayes_ols(full, m_pack, 'WBF', 'WBF')
                            plot_did_treated_wbf(pred, m_pack, st, pt.get('Passed/Failed'), os.path.join(sd, f'WBF_T_{g}.png'), g)
                            plot_did_control_wbf(pred, m_pack, st, pt.get('Passed/Failed'), os.path.join(sd, f'WBF_C_{g}.png'), g)
                            plot_boxplot_treated_wbf(pred, os.path.join(sd, f'Box_WBF_T_{g}.png'), g)
                            plot_boxplot_control_wbf(pred, os.path.join(sd, f'Box_WBF_C_{g}.png'), g)

                # Save Results (Header Renaming for Non-Statisticians)
                def save_sheet(data, pt_data, name, metric_type):
                    if not data: return
                    df = pd.merge(pd.DataFrame(data), pd.DataFrame(pt_data)[['Group', 'Passed/Failed']], on='Group', how='left')
                    
                    # カラム名のマッピング（わかりやすい日本語へ）
                    rename_map = {
                        'RR (Risk Ratio)': '改善効果 (RR < 1.0)',
                        'Range_Lower': '95%確率範囲(下限)',
                        'Range_Upper': '95%確率範囲(上限)',
                        'Prob_No_Improvement': '改善しなかった確率 (小さいほど良い)',
                        'Passed/Failed': 'トレンド検定(参考)'
                    }
                    if metric_type == 'CR':
                        # CRはRR
                        pass 
                    else:
                        # 他はBeta > 0が改善
                        rename_map[f'Beta_Dose ({metric_type} Effect)'] = f'改善効果 ({metric_type} > 0)'

                    df.rename(columns=rename_map, inplace=True)

                    # 出力列の整理
                    target_cols = ['Group']
                    if metric_type == 'CR':
                        target_cols.append('改善効果 (RR < 1.0)')
                    else:
                        target_cols.append(f'改善効果 ({metric_type} > 0)')
                        
                    target_cols.extend(['改善しなかった確率 (小さいほど良い)', '95%確率範囲(下限)', '95%確率範囲(上限)', 'トレンド検定(参考)'])
                    
                    # 存在する列だけ出力
                    final_cols = [c for c in target_cols if c in df.columns]
                    df[final_cols].to_excel(writer, sheet_name=name, index=False)

                save_sheet(res_cr, pt_cr, 'DiD_CountRate', 'CR')
                save_sheet(res_mtbf, pt_mtbf, 'DiD_MTBF', 'MTBF')
                save_sheet(res_rf, pt_rf, 'DiD_RFMTBF', 'RF-MTBF')
                save_sheet(res_wbf, pt_wbf, 'DiD_WBF', 'WBF')

            logging.info("Complete")
            messagebox.showinfo("Success", f"Done.\nFile: {out_f}")

        except Exception as e:
            logging.error(traceback.format_exc())
            messagebox.showerror("Error", str(e))
        finally:
            self.btn.config(state='normal')
            self.progress_label.config(text='')

if __name__ == "__main__":
    root = tk.Tk()
    app = DidAnalysisBayesianApp(root)
    root.mainloop()
