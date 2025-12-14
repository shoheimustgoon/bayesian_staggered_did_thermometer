# -*- coding: utf-8 -*-
"""
Bayesian Staggered DiD Analyzer
Author: Go Sato
Description: GUI tool to analyze AI Thermometer impact using Bayesian Gaussian Models.
"""
import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # GUIバックエンドとの競合を防ぐためAggを使用
import matplotlib.pyplot as plt
import logging
import traceback
import warnings

# Bayesian Libraries
import bambi as bmb
import arviz as az

# Settings
plt.rcParams['font.family'] = 'MS Gothic' # 日本語フォント設定
warnings.simplefilter('ignore')

# PyTensor (PyMCのバックエンド) の高速化フラグ
os.environ['PYTENSOR_FLAGS'] = 'device=cpu,floatX=float64,optimizer=fast_compile,cxx='

# MCMC設定 (計算時間と精度のバランス)
BAYES_DRAWS = 1000
BAYES_TUNE = 500
BAYES_CHAINS = 2

# プロット用カラー設定
COLOR_ACTUAL = '#1f77b4'  # 青 (実績)
COLOR_CF     = '#d62728'  # 赤 (反実仮想: AIなし)
COLOR_CI     = '#7f7f7f'  # グレー (信用区間)

class StaggeredApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Bayesian Staggered DiD Analyzer - Go Sato")
        self.root.geometry("650x450")
        
        # タイトル
        title_lbl = ttk.Label(root, text="Bayesian Staggered DiD Analysis Tool", font=("Arial", 14, "bold"))
        title_lbl.pack(pady=15)
        
        # ファイル選択エリア
        frame = ttk.LabelFrame(root, text="1. Select Data File (Excel)", padding=15)
        frame.pack(fill='x', padx=20, pady=10)
        
        self.path_var = tk.StringVar()
        entry = ttk.Entry(frame, textvariable=self.path_var)
        entry.pack(side='left', fill='x', expand=True, padx=(0, 5))
        
        btn_browse = ttk.Button(frame, text="Browse", command=self.browse)
        btn_browse.pack(side='left')
        
        # 実行ボタンエリア
        btn_frame = ttk.Frame(root)
        btn_frame.pack(pady=20)
        
        self.btn_run = ttk.Button(btn_frame, text="Run Bayesian Analysis", command=self.run_analysis)
        self.btn_run.pack(ipadx=20, ipady=10)
        
        # ステータス表示
        self.status_var = tk.StringVar(value="Ready")
        status_lbl = ttk.Label(root, textvariable=self.status_var, foreground="blue", font=("Arial", 10))
        status_lbl.pack(pady=5)
        
        # フッター
        footer_lbl = ttk.Label(root, text="Model: Yield ~ Time + Line_FE + Intervention (Gaussian)", foreground="gray", font=("Arial", 8))
        footer_lbl.pack(side='bottom', pady=10)
        
    def browse(self):
        f = filedialog.askopenfilename(filetypes=[("Excel", "*.xlsx")])
        if f: self.path_var.set(f)

    def load_data(self, path):
        """Excelの複数シートを読み込み、パネルデータとして統合する"""
        try:
            xls = pd.ExcelFile(path)
            # "Data_" で始まるシートのみを対象とする
            sheets = [s for s in xls.sheet_names if s.startswith('Data_')]
            
            if not sheets:
                return None, "No sheets starting with 'Data_' found."

            df_list = []
            for s in sheets:
                df = pd.read_excel(xls, sheet_name=s)
                # シート名をLine_IDとして列に追加
                df['Line_ID'] = s.replace('Data_', '')
                df_list.append(df)
            
            df_panel = pd.concat(df_list, ignore_index=True)
            df_panel['Month'] = pd.to_datetime(df_panel['Month'])
            
            # モデリング用の数値時間インデックス (0, 1, 2...) を作成
            start_date = df_panel['Month'].min()
            df_panel['Time_Index'] = ((df_panel['Month'] - start_date) / np.timedelta64(1, 'M')).astype(int)
            
            return df_panel, None
        except Exception as e:
            return None, str(e)

    def run_analysis(self):
        path = self.path_var.get()
        if not path:
            messagebox.showwarning("Warning", "Please select a file first.")
            return
        
        # UIロック & ステータス更新
        self.btn_run.config(state='disabled')
        self.status_var.set("Loading Data & Running MCMC... (This may take a minute)")
        self.root.update()
        
        try:
            save_dir = os.path.dirname(path)
            
            # 1. データ読み込み
            df_panel, err = self.load_data(path)
            if err: raise Exception(err)

            # 2. ベイズモデリング (Gaussian)
            # Formula: Yield_Score ~ 時間トレンド + ライン固定効果 + 介入効果
            print("Building Bayesian Model...")
            model = bmb.Model("Yield_Score ~ Time_Index + C(Line_ID) + Intervention_On", 
                              data=df_panel, family="gaussian")
            
            # MCMC実行
            idata = model.fit(draws=BAYES_DRAWS, tune=BAYES_TUNE, chains=BAYES_CHAINS, 
                              random_seed=42, progressbar=True)
            
            # 3. 統計量の抽出
            summary = az.summary(idata, var_names=['Intervention_On'], hdi_prob=0.95)
            beta = summary.loc['Intervention_On', 'mean']
            hdi_l = summary.loc['Intervention_On', 'hdi_2.5%']
            hdi_u = summary.loc['Intervention_On', 'hdi_97.5%']
            
            # 改善確率 (効果 > 0 である確率)
            post_samples = idata.posterior['Intervention_On'].values.flatten()
            prob_pos = (post_samples > 0).mean()

            # 4. 可視化 (反実仮想プロット)
            self.generate_plots(model, idata, df_panel, save_dir, beta, prob_pos)
            
            # 完了メッセージ
            msg = f"Analysis Complete!\n\n" \
                  f"Estimated AI Effect: +{beta:.2f} points\n" \
                  f"Prob of Improvement: {prob_pos*100:.1f}%\n" \
                  f"95% Credible Interval: [{hdi_l:.2f}, {hdi_u:.2f}]"
            
            messagebox.showinfo("Success", msg)
            self.status_var.set("Analysis Done. Check saved images.")

        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")
            self.status_var.set("Error Occurred")
        finally:
            self.btn_run.config(state='normal')

    def generate_plots(self, model, idata, df_panel, save_dir, beta, prob):
        """実績値と反実仮想(AIなし)を比較プロット"""
        
        # 反実仮想データの作成 (介入フラグを全員0にする)
        df_cf = df_panel.copy()
        df_cf['Intervention_On'] = 0 
        
        # 予測 (事後分布の平均)
        model.predict(idata, data=df_cf, kind="mean", inplace=True)
        
        # 事後分布から平均と信用区間を取得
        post_mean = idata.posterior['Yield_Score_mean']
        cf_mean = post_mean.mean(dim=["chain", "draw"]).values
        cf_low = post_mean.quantile(0.025, dim=["chain", "draw"]).values
        cf_high = post_mean.quantile(0.975, dim=["chain", "draw"]).values
        
        df_panel['CF_Mean'] = cf_mean
        df_panel['CF_Low'] = cf_low
        df_panel['CF_High'] = cf_high
        
        # ラインごとにプロット作成
        for line in df_panel['Line_ID'].unique():
            df_sub = df_panel[df_panel['Line_ID'] == line].sort_values('Month')
            
            fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
            
            # 実績値 (Actual)
            ax.plot(df_sub['Month'], df_sub['Yield_Score'], 'o-', color=COLOR_ACTUAL, label='Actual Yield', alpha=0.8, linewidth=2)
            
            # 反実仮想 (Counterfactual)
            ax.plot(df_sub['Month'], df_sub['CF_Mean'], '--', color=COLOR_CF, label='Counterfactual (No AI)')
            ax.fill_between(df_sub['Month'], df_sub['CF_Low'], df_sub['CF_High'], color=COLOR_CI, alpha=0.2, label='95% Credible Interval')
            
            # 介入開始線の描画
            intervention_start = df_sub[df_sub['Intervention_On'] == 1]['Month'].min()
            
            if pd.notna(intervention_start):
                ax.axvline(intervention_start, color='black', linestyle=':', label='AI Introduction')
                
                # 効果の塗りつぶし (実績 > 反実仮想 の部分)
                mask = df_sub['Month'] >= intervention_start
                ax.fill_between(df_sub['Month'], df_sub['Yield_Score'], df_sub['CF_Mean'], 
                                where=mask & (df_sub['Yield_Score'] > df_sub['CF_Mean']),
                                color='green', alpha=0.2, label='Estimated Improvement')

            # グラフ装飾
            ax.set_title(f"Bayesian Staggered DiD: {line}\nEffect: +{beta:.2f} (Prob: {prob*100:.0f}%)")
            ax.set_ylabel("Yield Score (0-100)")
            ax.legend(loc='lower left')
            ax.grid(True, linestyle=':', alpha=0.6)
            
            # 保存
            fname = os.path.join(save_dir, f'Result_Plot_{line}.png')
            fig.tight_layout()
            fig.savefig(fname)
            plt.close(fig)

if __name__ == "__main__":
    root = tk.Tk()
    app = StaggeredApp(root)
    root.mainloop()
