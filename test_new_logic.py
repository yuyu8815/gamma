import pandas as pd
import numpy as np

# Load data
df_vix = pd.read_pickle('data/vix_1min_2025.pkl')[['time', 'vix_like']].sort_values('time')
df_txf = pd.read_pickle('data/TXF_H2.pkl').reset_index().sort_values('time')
df_merged = pd.merge_asof(df_txf, df_vix, on='time', direction='backward')
df_merged = df_merged[df_merged['time'].dt.month != 9]

def gamma_scalping_new(price_df, gamma=10, cost_per_contract=10, hedge_threshold_delta=200):
    """
    全新的Gamma Scalping計算
    
    重點:
    - ds = 當前價格 - 上次hedge價格 (每次hedge後重置)
    - gamma_pnl = 0.5 * gamma * ds^2
    - futures hedge時才發生交易成本
    """
    data = price_df.sort_values('time').reset_index(drop=True)
    
    results = []
    
    # 初始狀態
    start_price = data['mkt_mid'].iloc[0]
    last_hedge_price = start_price  # 上次hedge的價格
    
    fut_pos = 0  # 期貨部位 (contracts)
    fut_pnl_total = 0  # 期貨累積損益
    total_cost = 0  # 累積成本
    
    TXF_POINT_VALUE = 200
    prev_price = start_price
    
    for row in data.itertuples():
        current_price = row.mkt_mid
        price_change = current_price - prev_price
        
        # 1. 期貨部位的MtM損益
        fut_pnl_change = fut_pos * price_change * TXF_POINT_VALUE
        fut_pnl_total += fut_pnl_change
        
        # 2. 計算ds (從上次hedge到現在的價格變化)
        ds = current_price - last_hedge_price
        
        # 3. Gamma PnL (基於ds)
        gamma_pnl = 0.5 * gamma * (ds ** 2)
        
        # 4. 計算選擇權的delta
        opt_delta = gamma * (current_price - start_price)
        
        # 5. 計算net delta
        fut_delta = fut_pos * TXF_POINT_VALUE
        net_delta = opt_delta + fut_delta
        
        # 6. 檢查是否需要hedge
        trade_contracts = 0
        if abs(net_delta) >= hedge_threshold_delta:
            # 需要hedge
            trade_contracts = -int(round(net_delta / TXF_POINT_VALUE))
            
            if trade_contracts != 0:
                fut_pos += trade_contracts
                trade_cost = abs(trade_contracts) * cost_per_contract
                total_cost += trade_cost
                
                # *** 重點: hedge後重置last_hedge_price ***
                last_hedge_price = current_price
                
                # 重新計算net_delta
                fut_delta = fut_pos * TXF_POINT_VALUE
                net_delta = opt_delta + fut_delta
        
        # 7. 總損益
        total_pnl = gamma_pnl + fut_pnl_total - total_cost
        
        results.append({
            'time': row.time,
            'price': current_price,
            'ds': ds,
            'gamma_pnl': gamma_pnl,
            'fut_pos': fut_pos,
            'fut_pnl': fut_pnl_total,
            'net_delta': net_delta,
            'total_pnl': total_pnl,
            'trade': trade_contracts,
            'vix': getattr(row, 'vix_like', np.nan)
        })
        
        prev_price = current_price
    
    return pd.DataFrame(results)

# Run the function
result = gamma_scalping_new(df_merged, gamma=10, cost_per_contract=10, hedge_threshold_delta=200)

print(f"Total Trades: {result['trade'].abs().sum()}")
print(f"Final Total PnL: {result['total_pnl'].iloc[-1]:,.2f}")
print(f"\n前20筆交易 (只顯示有hedge的時間點):")

trades_only = result[result['trade'] != 0].head(20)
print(trades_only[['time', 'price', 'ds', 'gamma_pnl', 'fut_pos', 'fut_pnl', 'total_pnl', 'trade']].to_string())
