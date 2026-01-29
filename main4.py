import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta
from math import sqrt
import os

plt.style.use("default")

# ==========================
# KONFIG: MARKNADER & FILER
# ==========================

markets = [
    {
        "name": "EURJPY",
        "csv": "EURJPY_5M_2012-2026.csv",
        "pip_size": 0.0001,
        # USDJPY: 0.01, NZDUSD och AUDUSD och USDCHF och USDCAD och EURUSD och GBPUSD och USDCAD: 0.0001 pip
        "spread_points_per_pip": 10.0,  # om spread_points är pipetter (vanligast). Om er kolumn redan är pips: sätt 1.0
    },
]

# ==========================
# COST MODEL (POINTS)
# ==========================
HALF = 0.5

SLIPPAGE_PIPS = 0.10  # AUDJPY: 0.10, NZDUSD: 0.00, USDCAD: 0.10, AUDUSD och USDCHF och USDJPY och GBPUSD: 0.08, EURUSD:  0.05 pip (0.5 pipette)
FIXED_SPREAD_PIPS = 0.25 # AUDJPY: 0.30, NZDUSD: 1.2, AUDUSD och USDCHF: 0.18, USDCAD: 0.22, USDJPY och GBPUSD: 0.15, EURUSD: 0.10 pip (1 pipette) - anpassa!
COMM_PIPS_PER_SIDE = 0.25


def compute_stats_from_trades(trades_df: pd.DataFrame) -> dict:
    """
    Samma logik som era totala stats, men på en subset av trades.
    trades_df måste ha kolumner: pnl, equity (valfritt), is_win (valfritt)
    """
    if trades_df.empty:
        return {}

    df_ = trades_df.copy()

    # equity behövs för DD
    df_["equity"] = df_["pnl"].cumsum()

    df_["is_win"] = df_["pnl"] > 0

    gross_profit = df_.loc[df_["pnl"] > 0, "pnl"].sum()
    gross_loss = df_.loc[df_["pnl"] < 0, "pnl"].sum()  # negativt tal
    profit_factor = (gross_profit / abs(gross_loss)) if gross_loss != 0 else np.inf

    avg_win = df_.loc[df_["pnl"] > 0, "pnl"].mean()
    avg_loss = df_.loc[df_["pnl"] < 0, "pnl"].mean()

    winrate = df_["is_win"].mean()
    expectancy = df_["pnl"].mean()

    roll_max = df_["equity"].cummax()
    dd = df_["equity"] - roll_max
    max_dd_points = float(abs(dd.min())) if len(dd) > 0 else 0.0

    # Losing streak
    loss_streak = 0
    max_loss_streak = 0
    for is_win in df_["is_win"]:
        if not is_win:
            loss_streak += 1
            max_loss_streak = max(max_loss_streak, loss_streak)
        else:
            loss_streak = 0

    pnl_std = df_["pnl"].std(ddof=1)
    sharpe_trade = (expectancy / pnl_std) * sqrt(len(df_)) if pnl_std and pnl_std > 0 else np.nan

    return {
        "Trades": int(len(df_)),
        "Total PnL (points)": float(df_["pnl"].sum()),
        "Gross Profit": float(gross_profit),
        "Gross Loss": float(gross_loss),
        "Profit Factor": float(profit_factor),
        "Winrate": float(winrate),
        "Avg Win": float(avg_win) if not np.isnan(avg_win) else np.nan,
        "Avg Loss": float(avg_loss) if not np.isnan(avg_loss) else np.nan,
        "Expectancy (avg/trade)": float(expectancy),
        "Max Drawdown (points)": float(max_dd_points),
        "Max Losing Streak (trades)": int(max_loss_streak),
        "Sharpe (trade-level)": float(sharpe_trade) if not np.isnan(sharpe_trade) else np.nan,
    }


def run_backtest_for_market(market_name: str, csv_path: str, pip_size: float, spread_points_per_pip: float = 10.0):
    global close_price
    print("\n" + "=" * 70)
    print(f" BACKTEST FÖR MARKNAD: {market_name} ")
    print("=" * 70 + "\n")

    # 1) Ladda data
    df = pd.read_csv(csv_path)

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
    elif 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')
    else:
        raise ValueError("Hittar ingen 'timestamp' eller 'datetime'-kolumn i CSV.")

    df = df.sort_index()
    # ---- DEDUPE INDEX (viktigt för groupby/transform) ----
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="last")].sort_index()

    required_cols = {'open', 'high', 'low', 'close'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV måste innehålla kolumnerna: {required_cols}")

    def pips_to_price(pips: float) -> float:
        return float(pips) * float(pip_size)

    def commission_round_turn_price() -> float:
        return 2.0 * pips_to_price(COMM_PIPS_PER_SIDE)

    # EMA 50
    df['ema_fast'] = df['close'].ewm(span=50, adjust=False).mean()
    # EMA 200
    df['ema_slow'] = df['close'].ewm(span=200, adjust=False).mean()

    # ADX
    period = 14  # klassisk ADX-inställning

    high = df['high']
    low = df['low']
    close = df['close']

    # 1. True Range (TR)
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # 2. Directional Movement (+DM / -DM)
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)

    # 3. Wilder-smoothing (exponentiell men med alpha = 1/period)
    tr_smooth = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_dm_smooth = plus_dm.ewm(alpha=1 / period, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=1 / period, adjust=False).mean()

    # 4. DI+
    plus_di = 100 * (plus_dm_smooth / tr_smooth)
    minus_di = 100 * (minus_dm_smooth / tr_smooth)

    # 5. DX och ADX
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()

    df['ADX'] = adx
    df['PLUS_DI'] = plus_di
    df['MINUS_DI'] = minus_di

    USE_SPREAD_PIPS_COL = 'spread_pips' in df.columns
    USE_SPREAD_POINTS_COL = 'spread_points' in df.columns

    def get_spread_pips(row) -> float:
        if USE_SPREAD_PIPS_COL:
            return float(row['spread_pips'])
        if USE_SPREAD_POINTS_COL:
            return float(row['spread_points']) / float(spread_points_per_pip)
        return float(FIXED_SPREAD_PIPS)


    # 4) Backtest-loop
    trades = []
    in_position = False
    pos_direction = None
    entry_price = None
    entry_time = None


    idx_list = df.index.to_list()

    for i in range(1, len(df) - 1):

        ts = idx_list[i]
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]
        next_row = df.iloc[i + 1]
        close_price = row["close"]
        ema_fast = row["ema_fast"]
        ema_slow = row["ema_slow"]
        prev_ema_fast = prev_row["ema_fast"]
        prev_ema_slow = prev_row["ema_slow"]
        # ======================
        # EXIT-logik (signal på bar i, fill på bar i+1 open)
        # ======================
        if in_position:
            exit_price = None
            exit_reason = None

            if pos_direction == "LONG":
                if (prev_ema_fast > prev_ema_slow) and (ema_fast < ema_slow):

                    spread_pips = get_spread_pips(next_row)
                    spread_px = pips_to_price(spread_pips)
                    slip_px = pips_to_price(SLIPPAGE_PIPS)

                    exit_price = next_row["open"] - HALF * spread_px - slip_px
                    exit_reason = "ema"

            else:  # SHORT
                if (prev_ema_fast < prev_ema_slow) and (ema_fast > ema_slow):

                    spread_pips = get_spread_pips(next_row)
                    spread_px = pips_to_price(spread_pips)
                    slip_px = pips_to_price(SLIPPAGE_PIPS)

                    exit_price = next_row["open"] + HALF * spread_px + slip_px
                    exit_reason = "ema"

            if exit_price is not None:
                exit_time = idx_list[i + 1]
                comm_px = commission_round_turn_price()

                if pos_direction == "LONG":
                    pnl = (exit_price - entry_price) - comm_px
                else:
                    pnl = (entry_price - exit_price) - comm_px

                trades.append({
                    "Entry Time": entry_time,
                    "Exit Time": exit_time,
                    "Direction": pos_direction,
                    "Entry Price": entry_price,
                    "Exit Price": exit_price,
                    "Exit Reason": exit_reason,
                    "pnl": pnl,
                })

                in_position = False
                pos_direction = None
                entry_price = None
                entry_time = None
                continue

        # ======================
        # ENTRY-logik (NY)
        # ======================

        next_open = next_row["open"]
        prev_ema_fast = prev_row["ema_fast"]
        prev_ema_slow = prev_row["ema_slow"]
        adx = row["ADX"]
        adx_threshold = 25

        long_signal = (prev_ema_fast < prev_ema_slow) and (ema_fast > ema_slow)
        short_signal = (prev_ema_fast > prev_ema_slow) and (ema_fast < ema_slow)
        trend_filter = adx > adx_threshold

        if long_signal and trend_filter:
            pos_direction = "LONG"
            entry_time = idx_list[i + 1]

            spread_pips = get_spread_pips(next_row)
            spread_px = pips_to_price(spread_pips)
            slip_px = pips_to_price(SLIPPAGE_PIPS)

            entry_price = next_open + HALF * spread_px + slip_px
            in_position = True

        elif short_signal and trend_filter:
            pos_direction = "SHORT"
            entry_time = idx_list[i + 1]

            spread_pips = get_spread_pips(next_row)
            spread_px = pips_to_price(spread_pips)
            slip_px = pips_to_price(SLIPPAGE_PIPS)

            entry_price = next_open - HALF * spread_px - slip_px
            in_position = True

    # ==========================
    # 5. Resultatsammanställning
    # ==========================
    trades_df = pd.DataFrame(trades)
    '''
    def stats_basic(df):
        gross_profit = df.loc[df["pnl"] > 0, "pnl"].sum()
        gross_loss = df.loc[df["pnl"] < 0, "pnl"].sum()  # negativt
        pf = (gross_profit / abs(gross_loss)) if gross_loss != 0 else np.inf
        return {
            "Trades": len(df),
            "TotalPnL": df["pnl"].sum(),
            "Expectancy": df["pnl"].mean(),
            "Winrate": (df["pnl"] > 0).mean(),
            "ProfitFactor": pf
        }

    def trim_top_winners(trades_df, top_pct):
        df = trades_df.copy()
        # Tröskel för top X% vinnare
        cutoff = df["pnl"].quantile(1 - top_pct)
        # Ta bort endast trades som är i toppskiktet (stora vinnare)
        df_trim = df[df["pnl"] <= cutoff].copy()
        return df_trim, cutoff

    base = stats_basic(trades_df)

    for pct in [0.01, 0.05]:
        trimmed, cutoff = trim_top_winners(trades_df, pct)
        s = stats_basic(trimmed)
        print("\n" + "-" * 60)
        print(f"REMOVE TOP {int(pct * 100)}% WINNERS | cutoff pnl <= {cutoff:.4f}")
        print(
            f"BASE:   Trades={base['Trades']}  TotalPnL={base['TotalPnL']:.4f}  Exp={base['Expectancy']:.4f}  PF={base['ProfitFactor']:.4f}")
        print(
            f"TRIM:   Trades={s['Trades']}     TotalPnL={s['TotalPnL']:.4f}  Exp={s['Expectancy']:.4f}  PF={s['ProfitFactor']:.4f}")
    '''
    if trades_df.empty:
        print("Inga trades hittades.")
        return None, trades_df

    trades_df = trades_df.sort_values("Exit Time").reset_index(drop=True)
    trades_df["equity"] = trades_df["pnl"].cumsum()

    # --- Extra statistik ---
    trades_df["is_win"] = trades_df["pnl"] > 0

    gross_profit = trades_df.loc[trades_df["pnl"] > 0, "pnl"].sum()
    gross_loss = trades_df.loc[trades_df["pnl"] < 0, "pnl"].sum()  # negativt tal
    profit_factor = (gross_profit / abs(gross_loss)) if gross_loss != 0 else np.inf

    avg_win = trades_df.loc[trades_df["pnl"] > 0, "pnl"].mean()
    avg_loss = trades_df.loc[trades_df["pnl"] < 0, "pnl"].mean()  # negativt

    winrate = trades_df["is_win"].mean()

    # Expectancy per trade
    expectancy = trades_df["pnl"].mean()

    # Drawdown
    roll_max = trades_df["equity"].cummax()
    dd = trades_df["equity"] - roll_max
    max_dd = dd.min()  # negativt
    max_dd_points = abs(max_dd)  # positivt för rapportering

    # Longest losing streak (räknat i trades)
    loss_streak = 0
    max_loss_streak = 0
    for is_win in trades_df["is_win"]:
        if not is_win:
            loss_streak += 1
            max_loss_streak = max(max_loss_streak, loss_streak)
        else:
            loss_streak = 0

    # “Sharpe” på trade-nivå (inte tidsnormaliserad)
    pnl_std = trades_df["pnl"].std(ddof=1)
    sharpe_trade = (expectancy / pnl_std) * sqrt(len(trades_df)) if pnl_std and pnl_std > 0 else np.nan

    stats = {
        "Market": market_name,
        "Trades": int(len(trades_df)),
        "Total PnL (points)": float(trades_df["pnl"].sum()),
        "Gross Profit": float(gross_profit),
        "Gross Loss": float(gross_loss),
        "Profit Factor": float(profit_factor),
        "Winrate": float(winrate),
        "Avg Win": float(avg_win) if not np.isnan(avg_win) else np.nan,
        "Avg Loss": float(avg_loss) if not np.isnan(avg_loss) else np.nan,
        "Expectancy (avg/trade)": float(expectancy),
        "Max Drawdown (points)": float(max_dd_points),
        "Max Losing Streak (trades)": int(max_loss_streak),
        "Sharpe (trade-level)": float(sharpe_trade) if not np.isnan(sharpe_trade) else np.nan,
    }

    print("\n--- STATS ---")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

        # ==========================
        # PER-ÅR STATS (per market)
        # ==========================
    print("\n--- PER-ÅR STATS ---")

    # vi använder Exit Time som "trade year" (rekommenderat)
    trades_df["Year"] = pd.to_datetime(trades_df["Exit Time"]).dt.year

    years = sorted(trades_df["Year"].unique().tolist())
    for y in years:
        sub = trades_df[trades_df["Year"] == y].copy()
        y_stats = compute_stats_from_trades(sub)

        if not y_stats:
            continue

        print(f"\n{market_name} - {y}:")
        print(f"Trades: {y_stats['Trades']}")
        print(f"Total PnL (points): {y_stats['Total PnL (points)']:.4f}")
        print(f"Gross Profit: {y_stats['Gross Profit']:.4f}")
        print(f"Gross Loss: {y_stats['Gross Loss']:.4f}")
        print(f"Profit Factor: {y_stats['Profit Factor']:.4f}")
        print(f"Winrate: {y_stats['Winrate']:.4f}")
        print(f"Avg Win: {y_stats['Avg Win']:.4f}")
        print(f"Avg Loss: {y_stats['Avg Loss']:.4f}")
        print(f"Expectancy (avg/trade): {y_stats['Expectancy (avg/trade)']:.4f}")
        print(f"Max Drawdown (points): {y_stats['Max Drawdown (points)']:.4f}")
        print(f"Max Losing Streak (trades): {y_stats['Max Losing Streak (trades)']}")
        print(f"Sharpe (trade-level): {y_stats['Sharpe (trade-level)']:.4f}")

    # PLOT (som du redan får)
    plt.figure(figsize=(12, 5))
    plt.plot(trades_df["Exit Time"], trades_df["equity"])
    plt.title(f"Equity curve - {market_name}")
    plt.xlabel("Time")
    plt.ylabel("Cumulative PnL (points)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return stats, trades_df


# ==========================
# KÖR BACKTEST + SLUTSUMMERING + COMBINED EQUITY & STATS
# ==========================

all_results = []
all_trades = []

for m in markets:
    try:
        stats, trades_df = run_backtest_for_market(
            m["name"],
            m["csv"],
            m["pip_size"],
            m.get("spread_points_per_pip", 10.0),
        )
        if stats is not None and trades_df is not None:
            trades_df["Market"] = m["name"]
            all_results.append(stats)
            all_trades.append(trades_df)
    except Exception as e:
        print(f"\n*** FEL för {m['name']} ({m['csv']}): {e}\n")