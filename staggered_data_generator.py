# -*- coding: utf-8 -*-
"""
Staggered Data Generator for AI Thermometer Analysis
Author: Go Sato
Description: Generates dummy manufacturing yield data for Staggered DiD analysis.
"""
import pandas as pd
import numpy as np
import os

def generate_staggered_data(filename='Staggered_Yield_Data.xlsx'):
    # 設定: 24ヶ月分のデータ (2023/01 - 2024/12)
    dates = pd.date_range(start='2023-01-01', periods=24, freq='MS')
    
    # シナリオ設定: 3つのラインで導入時期をずらす
    # Line_A: 早期導入 (Month 8 = 2023/08から)
    # Line_B: 後期導入 (Month 16 = 2024/04から)
    # Line_C: 未導入 (Control)
    lines_config = {
        'Line_A_Early': {'start_idx': 8, 'base_yield': 82.0, 'effect': 5.0},
        'Line_B_Late':  {'start_idx': 16, 'base_yield': 78.0, 'effect': 5.5},
        'Line_C_Control': {'start_idx': 999, 'base_yield': 80.0, 'effect': 0.0}
    }
    
    print(f"Generating data to: {filename} ...")
    
    with pd.ExcelWriter(filename) as writer:
        for line_name, config in lines_config.items():
            n = len(dates)
            start_idx = config['start_idx']
            base = config['base_yield']
            effect = config['effect']
            
            # 共通の時間トレンド (季節性や経年変化)
            # 少しずつ良くなりつつ、周期的な波がある
            time_trend = np.linspace(0, 2.0, n) + np.sin(np.linspace(0, 6, n)) * 0.5
            
            yield_scores = []
            intervention_flags = []
            
            for i in range(n):
                # 介入フラグ (導入済みなら1, 未導入なら0)
                is_treated = 1 if i >= start_idx else 0
                intervention_flags.append(is_treated)
                
                # スコア生成: ベース + トレンド + 介入効果 + ノイズ
                mu = base + time_trend[i] + (effect * is_treated)
                
                # ガウシアンノイズ (正規分布)
                noise = np.random.normal(0, 1.2)
                score = mu + noise
                
                # 現実的な範囲 (0-100%) に収める
                score = min(100, max(0, score))
                yield_scores.append(score)
            
            # データフレーム作成
            df = pd.DataFrame({
                'Month': dates,
                'Line_ID': line_name,
                'Yield_Score': yield_scores,
                'Intervention_On': intervention_flags,
                # 生産数 (今回は重み付けには使わないが、データのリアリティのため追加)
                'Production_Volume': np.random.randint(1000, 1200, n)
            })
            
            # Excelのシートとして保存 (シート名 = ライン名)
            df.to_excel(writer, sheet_name=f'Data_{line_name}', index=False)
            
    print("✅ Done! Data generation complete.")
    print("   - Line_A (Early): AI introduced at Month 8")
    print("   - Line_B (Late):  AI introduced at Month 16")
    print("   - Line_C (Ctrl):  No AI introduction")

if __name__ == "__main__":
    generate_staggered_data()
