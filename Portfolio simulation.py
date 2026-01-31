import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

plt.style.use("default")

# ==========================
# KONFIG: MARKNADER & FILER
# ==========================


markets = [
    # --- FX ---
    {
        "name": "USDJPY", "asset_class": "fx",
        "csv": "USDJPY_5M_2012-now.csv", "pip_size": 0.01,
        "spread_points_per_pip": 10.0,
        "cost_model": {"slippage_pips": 0.15, "fixed_spread_pips": 0.25, "comm_pips_per_side": 0.25},
        "session_start": "08:00:00", "session_end": "16:00:00",
    },
    {
        "name": "GBPJPY", "asset_class": "fx",
        "csv": "GBPJPY_5M_2012-2026.csv", "pip_size": 0.01,
        "spread_points_per_pip": 10.0,
        "cost_model": {"slippage_pips": 0.30, "fixed_spread_pips": 0.50, "comm_pips_per_side": 0.25},
        "session_start": "08:00:00", "session_end": "16:00:00",
    },
    {
        "name": "USDCHF", "asset_class": "fx",
        "csv": "USDCHF_5M_2012-now.csv", "pip_size": 0.0001,
        "spread_points_per_pip": 10.0,
        "cost_model": {"slippage_pips": 0.20, "fixed_spread_pips": 0.35, "comm_pips_per_side": 0.25},
        "session_start": "08:00:00", "session_end": "16:00:00",
    },

    # --- INDEX (LONG ONLY) ---
    {
        "name": "NASDAQ", "asset_class": "index", "direction_mode": "long_only",
        "csv": "USATECH_5M_Stockholm_2012to2025_11.csv",
        "point_size": 1.0,  # 1 index-point i pris
        "quote_ccy": "USD",
        "quote_per_point_per_contract": 1.0,  # 1 USD per point per contract (vanligt)
        "cost_model": {"fixed_spread_points": 2.5, "slippage_points": 1.2, "comm_quote_per_side": 0.0},
        "session_start": "12:00:00", "session_end": "22:00:00",
    },
    {
        "name": "DAX40", "asset_class": "index", "direction_mode": "long_only",
        "csv": "DEU_IDX_EUR_5M_2012-now.csv",
        "point_size": 1.0,
        "quote_ccy": "EUR",
        "quote_per_point_per_contract": 1.0,  # <-- 1 EUR per point per contract (typisk CFD-variant)
        "cost_model": {"fixed_spread_points": 2.5, "slippage_points": 1.5, "comm_quote_per_side": 0.0},
        "session_start": "08:00:00", "session_end": "12:00:00",
    },
]

fx_rates = {
    "USDJPY": {"csv": "USDJPY_5M_2012-now.csv", "price_col": "close"},
    "USDCHF": {"csv": "USDCHF_5M_2012-now.csv", "price_col": "close"},
    "GBPUSD": {"csv": "GBPUSD_5M_2012-2026.csv", "price_col": "close"},  # <-- behövs för GBPJPY pricing
    "EURUSD": {"csv": "EURUSD_5M_2012-now.csv", "price_col": "close"},
}


HALF = 0.5


def clamp_time_series_index_unique(df: pd.DataFrame) -> pd.DataFrame:
    """Säkerställ unik, sorterad datetime-index."""
    df = df.sort_index()
    if df.index.has_duplicates:
        df = (df.groupby(df.index)
              .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
              .sort_index())
    return df


def annualize_factor_from_resample(rule: str) -> float:
    """Ann-faktor för returns baserat på resample-regel."""
    rule = rule.upper()
    if rule in ("D", "1D"):
        return 365.0
    if rule in ("B", "1B"):
        return 252.0
    # fallback: tolka som dagar
    return 365.0


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

def load_fx_rates(fx_rates: dict, start_time=None, end_time=None):
    """
    fx_rates format:
      {
        "USDJPY": {"csv":"...", "price_col":"close"},
        ...
      }
    Return:
      dict[symbol] -> pd.Series (price), datetimeindex
    """
    out = {}

    for sym, cfg in fx_rates.items():
        if not isinstance(cfg, dict) or "csv" not in cfg:
            raise ValueError(f"{sym}: fx_rates[{sym}] måste vara dict med 'csv'")

        csv_path = cfg["csv"]
        price_col = cfg.get("price_col", "close")

        df = pd.read_csv(csv_path)

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
        elif "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.set_index("datetime")
        else:
            raise ValueError(f"{sym}: saknar 'timestamp' eller 'datetime' i rates CSV")

        df = df.sort_index()
        if df.index.has_duplicates:
            df = df[~df.index.duplicated(keep="last")].sort_index()

        s = df[price_col].astype(float)

        if start_time is not None:
            s = s[s.index >= start_time]
        if end_time is not None:
            s = s[s.index <= end_time]

        out[sym] = s

    return out

def build_rates_df_for_portfolio(portfolio_trades: pd.DataFrame, fx_rates: dict, freq="5min") -> pd.DataFrame:
    t0 = pd.to_datetime(portfolio_trades["Entry Time"]).min()
    t1 = pd.to_datetime(portfolio_trades["Exit Time"]).max()

    rates_raw = load_fx_rates(fx_rates, start_time=t0, end_time=t1)  # dict[sym] -> Series

    start = pd.Timestamp(t0).floor(freq)
    end = pd.Timestamp(t1).ceil(freq)
    idx = pd.date_range(start=start, end=end, freq=freq)

    rates_df = pd.DataFrame(index=idx)

    for sym, s in rates_raw.items():
        rates_df[sym] = s.reindex(idx).ffill()

    return rates_df

def _rate_at(sym: str, t: pd.Timestamp, rates_df: pd.DataFrame) -> float:
    """
    Hämtar rate för sym vid tid t.
    rates_df index antas vara 5min och forward-filled.
    """
    t = pd.Timestamp(t).floor("5min")
    if t in rates_df.index:
        v = rates_df.at[t, sym]
        if pd.notna(v):
            return float(v)

    # fallback: närmaste tidigare tid
    idx = rates_df.index.get_indexer([t], method="ffill")
    if idx.size == 0 or idx[0] < 0:
        raise KeyError(f"No rate available for {sym} at {t}")
    v = rates_df.iloc[idx[0]][sym]
    if pd.isna(v):
        raise KeyError(f"Rate is NaN for {sym} at {rates_df.index[idx[0]]}")
    return float(v)


def quote_to_usd_rate(quote: str, t: pd.Timestamp, rates_df: pd.DataFrame) -> float:
    """
    Returnerar USD per 1 quote-valuta.
    Ex:
      quote='JPY' -> USD/JPY = 1 / (JPY per USD) = 1 / USDJPY
      quote='CHF' -> USD/CHF = 1 / USDCHF
      quote='USD' -> 1
      quote='GBP' -> USD/GBP? Nej, vi vill USD per GBP => GBPUSD (USD per GBP)
                     (denna fall hanteras av "quoteUSD" grenen)
    """
    quote = quote.upper()
    if quote == "USD":
        return 1.0

    # Om vi har quoteUSD (t.ex. GBPUSD, EURUSD, AUDUSD): det är USD per 1 quote
    sym_direct = f"{quote}USD"
    if sym_direct in rates_df.columns:
        return _rate_at(sym_direct, t, rates_df)

    # Om vi har USDquote (t.ex. USDJPY, USDCHF, USDCAD): det är quote per 1 USD
    sym_inv = f"USD{quote}"
    if sym_inv in rates_df.columns:
        r = _rate_at(sym_inv, t, rates_df)
        return 1.0 / r

    raise KeyError(f"Missing FX rate to convert {quote}->USD. Need {quote}USD or USD{quote} in rates_df.")

def approx_usd_per_pip_general(sym: str, pip_size: float, units: float, t: pd.Timestamp, rates_df: pd.DataFrame) -> float:
    """
    USD value of 1 pip for a position size 'units' (base units).
    Works for ALL FX pairs, including crosses.
    """
    quote = sym[3:]
    q2usd = quote_to_usd_rate(quote, t, rates_df)   # USD per 1 quote
    return float(units) * float(pip_size) * float(q2usd)

def base_to_usd_rate(base: str, t: pd.Timestamp, rates_df: pd.DataFrame) -> float:
    """
    Returnerar USD per 1 base-valuta.
    base='USD' -> 1
    base='GBP' -> GBPUSD
    base='JPY' -> JPYUSD? (vanligtvis saknas) => 1 / USDJPY
    """
    base = base.upper()
    if base == "USD":
        return 1.0

    sym_direct = f"{base}USD"
    if sym_direct in rates_df.columns:
        return _rate_at(sym_direct, t, rates_df)

    sym_inv = f"USD{base}"
    if sym_inv in rates_df.columns:
        r = _rate_at(sym_inv, t, rates_df)
        return 1.0 / r

    raise KeyError(f"Missing FX rate to convert {base}->USD. Need {base}USD or USD{base} in rates_df.")


def pip_value_usd_per_unit(sym: str, pip_size: float, t: pd.Timestamp, rates_df: pd.DataFrame) -> float:
    """
    USD value of 1 pip for 1 unit base currency.
    pip is in quote currency -> convert quote->USD.
    """
    base = sym[:3]
    quote = sym[3:]
    q2usd = quote_to_usd_rate(quote, t, rates_df)
    return float(pip_size) * q2usd


def size_units_from_usd_notional(sym: str, notional_usd: float, t: pd.Timestamp, rates_df: pd.DataFrame) -> float:
    """
    Konvertera USD-notional till base units.
    units = USD_notional / (USD per 1 base).
    """
    base = sym[:3]
    b2usd = base_to_usd_rate(base, t, rates_df)
    return float(notional_usd) / float(b2usd)

def size_units_from_usd_notional_generic(sym, notional_usd, t, market_cfg, rates_df):
    cfg = market_cfg[sym]
    cls = cfg.get("asset_class", "fx")

    if cls == "fx":
        return size_units_from_usd_notional(sym, notional_usd, t, rates_df)

    elif cls == "index":
        # units = contracts
        # Notional per contract approx = price * usd_per_point (om point_size=1)
        # Mer robust: notional_per_contract = (price / point_size) * usd_per_point
        point_size = float(cfg.get("point_size", 1.0))
        usd_per_point = float(cfg.get("usd_per_point_per_contract", 1.0))

        # behövs en prisserie för index i rates_df? enklast: använd entry price från trade
        # men här har vi inte row. Så: du kan skicka in entry_price eller ha en index_prices_df.
        # Minimal fix: sizea på “1 contract = $X riskbudget” -> använd en konstant contract_notional.
        contract_notional = float(cfg.get("contract_notional_usd", 100_000.0))
        return notional_usd / contract_notional

    else:
        raise ValueError(f"Unknown asset_class for {sym}: {cls}")

def run_backtest_for_market(
        market_name: str,
        csv_path: str,
        pip_size: float,
        spread_points_per_pip: float = 10.0,
        cost_model: dict | None = None,
        session_start: str = "08:00:00",
        session_end: str = "16:00:00",
        direction_mode: str = "both"  # "both" | "long_only" | "short_only"
):
    global entry_mid
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

    if cost_model is None:
        cost_model = {}

    slippage_pips = float(cost_model.get("slippage_pips"))
    fixed_spread_pips = float(cost_model.get("fixed_spread_pips"))
    comm_pips_per_side = float(cost_model.get("comm_pips_per_side"))

    def commission_round_turn_price() -> float:
        return 2.0 * pips_to_price(comm_pips_per_side)


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

    session_start_t = pd.to_datetime(session_start).time()
    session_end_t = pd.to_datetime(session_end).time()

    def in_session(ts) -> bool:
        t = ts.time()

        # Normal session (ex 01:00 -> 07:00)
        if session_start_t < session_end_t:
            return (t >= session_start_t) and (t < session_end_t)

        # Wrap session (ex 23:00 -> 03:00)
        # Då är det "i session" om tiden är >= start ELLER < end
        return (t >= session_start_t) or (t < session_end_t)

    USE_SPREAD_PIPS_COL = 'spread_pips' in df.columns
    USE_SPREAD_POINTS_COL = 'spread_points' in df.columns

    def get_spread_pips(row) -> float:
        if USE_SPREAD_PIPS_COL:
            return float(row['spread_pips'])
        if USE_SPREAD_POINTS_COL:
            return float(row['spread_points']) / float(spread_points_per_pip)
        return float(fixed_spread_pips)

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
                    slip_px = pips_to_price(slippage_pips)

                    exit_price = next_row["open"] - HALF * spread_px - slip_px
                    exit_reason = "ema"

            else:  # SHORT
                if (prev_ema_fast < prev_ema_slow) and (ema_fast > ema_slow):
                    spread_pips = get_spread_pips(next_row)
                    spread_px = pips_to_price(spread_pips)
                    slip_px = pips_to_price(slippage_pips)

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

        # Sessionfilter: både signalbar och fillbar måste vara i session
        if not in_session(ts):
            continue

        # ======================
        # ENTRY-logik (NY)
        # ======================

        next_open = next_row["open"]
        prev_ema_fast = prev_row["ema_fast"]
        prev_ema_slow = prev_row["ema_slow"]
        adx = row["ADX"]
        adx_threshold = 25

        allow_long = direction_mode in ("both", "long_only")
        allow_short = direction_mode in ("both", "short_only")
        long_signal = (prev_ema_fast < prev_ema_slow) and (ema_fast > ema_slow)
        short_signal = (prev_ema_fast > prev_ema_slow) and (ema_fast < ema_slow)
        trend_filter = adx > adx_threshold

        if allow_long and long_signal and trend_filter:
            pos_direction = "LONG"
            entry_time = idx_list[i + 1]

            spread_pips = get_spread_pips(next_row)
            spread_px = pips_to_price(spread_pips)
            slip_px = pips_to_price(slippage_pips)

            entry_price = next_open + HALF * spread_px + slip_px
            in_position = True

        elif allow_short and short_signal and trend_filter:
            pos_direction = "SHORT"
            entry_time = idx_list[i + 1]

            spread_pips = get_spread_pips(next_row)
            spread_px = pips_to_price(spread_pips)
            slip_px = pips_to_price(slippage_pips)

            entry_price = next_open - HALF * spread_px - slip_px
            in_position = True

    # ==========================
    # 5. Resultatsammanställning
    # ==========================
    trades_df = pd.DataFrame(trades)

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


def simulate_portfolio_equal_risk(
        trades: pd.DataFrame,
        market_cfg: dict,
        rates_df: pd.DataFrame,   # <-- NY
        start_capital: float = 50_000.0,
        max_one_position_per_market: bool = True,
        exposure_scale: float = 1.0,
        weights: dict | None = None,
):
    """
    Equal risk per market (Variant B): varje marknad får lika stor notional-budget vid entry.
    - Vi antar att trades har Entry/Exit Price som redan inkluderar spread+slippage.
    - 'pnl' i trades är price-delta per 1 unit (netto efter comm_px).
    """

    df = trades.copy()
    df = df.sort_values("Entry Time").reset_index(drop=True)

    df["Entry Time"] = pd.to_datetime(df["Entry Time"])
    df["Exit Time"] = pd.to_datetime(df["Exit Time"])
    df = df.sort_values("Entry Time").reset_index(drop=True)

    markets_list = sorted(df["Market"].unique().tolist())
    n_markets = len(markets_list)
    if n_markets == 0:
        return None, None, None

    if weights is None:
        market_budget = {m: 1.0 / n_markets for m in markets_list}
    else:
        market_budget = {m: float(weights.get(m, 0.0)) for m in markets_list}
        s = sum(market_budget.values())
        if s <= 0:
            market_budget = {m: 1.0 / n_markets for m in markets_list}
        else:
            market_budget = {m: market_budget[m] / s for m in markets_list}

    equity = start_capital
    equity_curve = []
    open_pos = {m: None for m in markets_list}  # store dict with units, entry, etc.

    # Event queue: vi itererar på entries men stänger allt som exit:ar före nästa entry
    # Skapa exits per market som min-heap-ish via scanning
    # Enkelt: vi processar i tidsordning med en lista över alla exit-events.
    exit_events = df[["Exit Time", "Market"]].copy()
    exit_events["idx"] = exit_events.index
    exit_events = exit_events.sort_values("Exit Time").reset_index(drop=True)
    exit_ptr = 0
    exit_events["Exit Time"] = pd.to_datetime(exit_events["Exit Time"])
    exit_events = exit_events.sort_values("Exit Time").reset_index(drop=True)

    def usd_pnl_for_trade_generic(row, units: float) -> float:
        sym = row["Market"]
        cfg = market_cfg[sym]
        cls = cfg.get("asset_class", "fx")

        t_exit = pd.Timestamp(row["Exit Time"]).floor("5min")
        if t_exit not in rates_df.index:
            t_exit = rates_df.index[rates_df.index.get_indexer([t_exit], method="ffill")[0]]

        if cls == "fx":
            pip_size = float(cfg["pip_size"])
            pip_val = pip_value_usd_per_unit(sym, pip_size, t_exit, rates_df)

            pnl_price = float(row["pnl"])  # price-delta per 1 unit base (inkl comm_px i price)
            pips_move = pnl_price / pip_size
            return pips_move * pip_val * units

        elif cls == "index":
            point_size = float(cfg.get("point_size", 1.0))
            quote_ccy = str(cfg.get("quote_ccy", "USD"))
            quote_per_point = float(cfg.get("quote_per_point_per_contract", 1.0))

            # row["pnl"] ska vara price-delta per 1 contract (executed, netto efter comm om ni lägger den i pnl)
            pnl_price = float(row["pnl"])
            pnl_points = pnl_price / point_size

            pnl_quote = pnl_points * quote_per_point * units  # units = contracts
            q2usd = quote_to_usd_rate(quote_ccy, t_exit, rates_df)  # EUR->USD för DAX
            return pnl_quote * float(q2usd)

        else:
            raise ValueError(f"Unknown asset_class for {sym}: {cls}")

    trade_log = []

    for i, row in df.iterrows():
        t_entry = row["Entry Time"]

        # 1) Close any positions whose exit time <= this entry time
        while exit_ptr < len(exit_events) and exit_events.loc[exit_ptr, "Exit Time"] <= t_entry:

            idx_to_close = int(exit_events.loc[exit_ptr, "idx"])
            r_close = df.loc[idx_to_close]
            mkt = r_close["Market"]

            pos = open_pos.get(mkt)

            if pos is not None and pos.get("trade_idx") == idx_to_close:
                units = pos["units"]
                pnl_usd = usd_pnl_for_trade_generic(r_close, units)
                equity += pnl_usd

                trade_log.append({
                    "Market": mkt,
                    "Entry Time": r_close["Entry Time"],
                    "Exit Time": r_close["Exit Time"],
                    "Direction": r_close.get("Direction", None),
                    "Entry Price": r_close.get("Entry Price", np.nan),
                    "Exit Price": r_close.get("Exit Price", np.nan),
                    "Entry Mid": r_close.get("Entry Mid", np.nan),
                    "Units": units,
                    "PnL_USD": pnl_usd,
                    "Equity": equity,
                })

                open_pos[mkt] = None
                equity_curve.append({"Time": r_close["Exit Time"], "Equity": equity})
            else:
                # Exit-event för en trade som aldrig öppnades i portföljen (skippad entry)
                # eller mismatch -> ignorera
                pass

            exit_ptr += 1

        mkt = row["Market"]
        if max_one_position_per_market and open_pos.get(mkt) is not None:
            # redan i position, skip entry
            continue

        # 2) Size: equal budget per market at time of entry
        notional_usd = equity * market_budget[mkt] * exposure_scale

        t_entry = pd.Timestamp(row["Entry Time"]).floor("5min")
        if t_entry not in rates_df.index:
            t_entry = rates_df.index[rates_df.index.get_indexer([t_entry], method="ffill")[0]]

        cfg = market_cfg[mkt]
        cls = cfg.get("asset_class", "fx")

        if cls == "fx":
            units = size_units_from_usd_notional(mkt, notional_usd, t_entry, rates_df)
        else:
            # INDEX: units = contracts
            entry_price = float(row["Entry Price"])
            point_size = float(cfg.get("point_size", 1.0))
            quote_ccy = str(cfg.get("quote_ccy", "USD"))
            quote_per_point = float(cfg.get("quote_per_point_per_contract", 1.0))

            # notional per contract i quote: (price/point_size)*quote_per_point
            notional_quote_per_contract = (entry_price / point_size) * quote_per_point

            # convert quote notional -> USD notional
            q2usd = float(quote_to_usd_rate(quote_ccy, t_entry, rates_df))
            notional_usd_per_contract = notional_quote_per_contract * q2usd

            units = notional_usd / notional_usd_per_contract

        open_pos[mkt] = {"units": units, "trade_idx": i, "notional_usd": notional_usd}

        equity_curve.append({"Time": t_entry, "Equity": equity})

    # 3) Close remaining positions at their exits
    # process remaining exits in chronological order
    while exit_ptr < len(exit_events):
        idx_to_close = int(exit_events.loc[exit_ptr, "idx"])
        r_close = df.loc[idx_to_close]
        mkt = r_close["Market"]

        pos = open_pos.get(mkt)

        # Stäng bara om den trade som exit:ar är exakt den du öppnade
        if pos is not None and pos.get("trade_idx") == idx_to_close:
            units = pos["units"]
            pnl_usd = usd_pnl_for_trade_generic(r_close, units, market_cfg, rates_df)
            equity += pnl_usd

            trade_log.append({
                "Market": mkt,
                "Entry Time": r_close["Entry Time"],
                "Exit Time": r_close["Exit Time"],
                "Direction": r_close.get("Direction", None),
                "Entry Price": r_close.get("Entry Price", np.nan),
                "Exit Price": r_close.get("Exit Price", np.nan),
                "Units": units,
                "PnL_USD": pnl_usd,
                "Equity": equity,
                "Entry Mid": r_close.get("Entry Mid", np.nan),
                "Direction": r_close.get("Direction", None),
            })

            open_pos[mkt] = None

            # NYTT: logga equity vid exit (där equity faktiskt ändras)
            equity_curve.append({"Time": r_close["Exit Time"], "Equity": equity})
        exit_ptr += 1

    eq_df = pd.DataFrame(equity_curve).drop_duplicates(subset=["Time"]).sort_values("Time")
    log_df = pd.DataFrame(trade_log).sort_values("Exit Time")

    # Drawdown stats
    if not eq_df.empty:
        eq_df["RollMax"] = eq_df["Equity"].cummax()
        eq_df["DD_$"] = eq_df["Equity"] - eq_df["RollMax"]
        eq_df["DD_%"] = eq_df["DD_$"] / eq_df["RollMax"]
        max_dd_usd = float(eq_df["DD_$"].min())
        max_dd_pct = float(eq_df["DD_%"].min())
    else:
        max_dd_usd = 0.0
        max_dd_pct = 0.0

    # ==========================
    # PORTFÖLJ-RATIO: Sharpe / Sortino / Calmar
    # ==========================
    days = 0.0
    years = 0.0

    sharpe = np.nan
    sortino = np.nan
    calmar = np.nan
    cagr = np.nan

    if not eq_df.empty and len(eq_df) >= 2:
        eq_ts = eq_df.copy()
        eq_ts["Time"] = pd.to_datetime(eq_ts["Time"])
        eq_ts = eq_ts.sort_values("Time").set_index("Time")

        # Daglig equity (kalenderdagar) och forward fill
        daily_eq = eq_ts["Equity"].resample("D").last().ffill()
        # Dagliga returns
        daily_ret = daily_eq.pct_change().dropna()

        # print("Days total:", len(daily_ret))
        # print("Days non-zero:", (daily_ret != 0).sum())
        # print("Share non-zero:", (daily_ret != 0).mean())

        if len(daily_ret) >= 30:
            ann_factor = 365.0  # eftersom vi använder kalenderdagar ("D")

            ret_mean = daily_ret.mean()
            ret_std = daily_ret.std(ddof=1)

            # Sharpe (RF=0)
            sharpe = (ret_mean / ret_std) * np.sqrt(ann_factor) if ret_std and ret_std > 0 else np.nan

            # Sortino (MAR = 0): downside deviation = sqrt(mean(min(0, r)^2))
            mar = 0.0
            downside_sq = np.minimum(0.0, daily_ret - mar) ** 2
            downside_dev = np.sqrt(downside_sq.mean())

            sortino = (ret_mean / downside_dev) * np.sqrt(ann_factor) if downside_dev and downside_dev > 0 else np.nan

            # CAGR (använd exakt tidslängd)
            start_val = float(daily_eq.iloc[0])
            end_val = float(daily_eq.iloc[-1])
            days = (daily_eq.index[-1] - daily_eq.index[0]).total_seconds() / 86400.0

            if days > 0 and start_val > 0:
                cagr = (end_val / start_val) ** (ann_factor / days) - 1.0
                years = days / 365.0 if days > 0 else 0.0

            # Calmar = CAGR / MaxDD (fraction)
            max_dd_frac = abs(max_dd_pct)  # max_dd_pct är en fraction (t.ex. -0.013)
            calmar = (cagr / max_dd_frac) if (np.isfinite(cagr) and max_dd_frac and max_dd_frac > 0) else np.nan

    summary = {
        "Start Capital": start_capital,
        "End Equity": float(equity),
        "Net PnL ($)": float(equity - start_capital),
        "Return (%)": float((equity / start_capital - 1.0) * 100.0),
        "Max Drawdown ($)": abs(max_dd_usd),
        "Max Drawdown (%)": abs(max_dd_pct) * 100.0,
        "Markets": n_markets,
        "Trades Closed": int(len(log_df)),
        "Days": float(days),
        "Years": float(years),
        "Sharpe (daily, ann.)": float(sharpe) if np.isfinite(sharpe) else np.nan,
        "Sortino (daily, ann.)": float(sortino) if np.isfinite(sortino) else np.nan,
        "CAGR (%)": float(cagr * 100.0) if np.isfinite(cagr) else np.nan,
        "Calmar": float(calmar) if np.isfinite(calmar) else np.nan,
    }

    return summary, eq_df, log_df


def build_price_series(market_cfg: dict, start_time=None, end_time=None, price_col="close"):
    """
    Returnerar dict: symbol -> pd.Series (price_col) med datetimeindex.
    Antag att CSV har 'timestamp' eller 'datetime' och 'close'.
    """
    out = {}
    for sym, cfg in market_cfg.items():
        df = pd.read_csv(cfg["csv"])
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
        elif "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.set_index("datetime")
        else:
            raise ValueError(f"{sym}: saknar timestamp/datetime")

        df = df.sort_index()
        if df.index.has_duplicates:
            df = df[~df.index.duplicated(keep="last")].sort_index()

        s = df[price_col].astype(float)

        if start_time is not None:
            s = s[s.index >= start_time]
        if end_time is not None:
            s = s[s.index <= end_time]

        out[sym] = s
    return out


def usd_pnl_from_price_delta(sym: str, pip_size: float, price_now: float, pnl_price_per_unit: float,
                             units: float) -> float:
    """
    Konverterar price-delta per unit till USD:
    - XXXUSD: 1 pip = pip_size USD per 1 unit => pip_value_per_unit = pip_size
    - USDXXX: pip_value_per_unit ~ pip_size / price_now
    """
    if sym.endswith("USD") and not sym.startswith("USD"):
        pip_value = pip_size
    else:
        pip_value = pip_size / price_now

    pips_move = pnl_price_per_unit / pip_size
    return pips_move * pip_value * units


def approx_usd_per_pip(sym: str, pip_size: float, units: float, price: float) -> float:
    """
    Approx USD value of 1 pip for a given position size (units).
    - XXXUSD: 1 pip = pip_size USD per unit
    - USDXXX: 1 pip in quote -> convert to USD using price => pip_size / price USD per unit
    """
    if sym.endswith("USD") and not sym.startswith("USD"):
        # EURUSD, GBPUSD, AUDUSD, NZDUSD...
        return units * pip_size
    else:
        # USDJPY, USDCHF, USDCAD...
        return units * (pip_size / price)


def compute_portfolio_mtm_equity_and_intraday_dd(
        port_log: pd.DataFrame,
        market_cfg: dict,
        start_capital: float = 50_000.0,
        freq: str = "5min",  # ni har 5m data
        price_col: str = "close",  # MTM på close (kan byta till open)
):
    """
    Bygger MTM equitykurva inklusive orealiserad PnL och räknar intraday DD.
    port_log måste ha: Market, Entry Time, Exit Time, Units, Direction, Entry Price, Entry Mid (minst)
    """

    if port_log.empty:
        return None, None, None

    log = port_log.copy()
    log["Entry Time"] = pd.to_datetime(log["Entry Time"])
    log["Exit Time"] = pd.to_datetime(log["Exit Time"])

    t0 = log["Entry Time"].min()
    t1 = log["Exit Time"].max()

    # 1) Ladda prisserier och resampla till gemensam frekvens
    prices = build_price_series(market_cfg, start_time=t0, end_time=t1, price_col=price_col)

    # 2) Skapa master timeline (union) och forward-fill priser
    # --- NYTT: validera tider ---
    if pd.isna(t0) or pd.isna(t1):
        raise ValueError("MTM: t0/t1 är NaT. Kontrollera att port_log har giltiga Entry/Exit Time.")

    start = pd.Timestamp(t0).floor(freq)
    end = pd.Timestamp(t1).ceil(freq)

    if end < start:
        raise ValueError(f"MTM: end < start ({end} < {start}). Kontrollera tiderna i port_log.")

    master_index = pd.date_range(start=start, end=end, freq=freq)

    if len(master_index) == 0:
        raise ValueError("MTM: master_index blev tom. Kontrollera freq och tidsintervall.")

    price_df = pd.DataFrame(index=master_index)
    for sym, s in prices.items():
        # align/resample till master
        aligned = s.reindex(master_index).ffill()
        price_df[sym] = aligned

    # 3) Event lists för att öppna/stänga positioner
    #    Vi itererar tid och håller en dict med öppna positioner per market (eller flera om ni tillåter)
    opens = log.sort_values("Entry Time").reset_index(drop=True)
    closes = log.sort_values("Exit Time").reset_index(drop=True)

    o_ptr = 0
    c_ptr = 0

    open_pos = {}  # sym -> list of positions (om ni vill tillåta flera)
    cash_equity = start_capital  # realiserad equity

    mtm_records = []

    for t in master_index:
        # close events
        while c_ptr < len(closes) and closes.loc[c_ptr, "Exit Time"] <= t:
            r = closes.loc[c_ptr]
            sym = r["Market"]

            # realiserad PnL finns redan i er simulator som PnL_USD per trade i port_log?
            # I er port_log har ni PnL_USD och Equity efter stängning.
            # Om den finns: använd den för cash_equity så cash matchar simulatorn.
            if "PnL_USD" in r:
                cash_equity += float(r["PnL_USD"])
            else:
                # fallback: räkna från executed pnl price per unit (om ni sparar den)
                raise ValueError("port_log saknar PnL_USD. Lägg till det i simulate_portfolio_equal_risk.")

            # ta bort positionen ur open_pos
            if sym in open_pos and len(open_pos[sym]) > 0:
                open_pos[sym].pop(0)
                if len(open_pos[sym]) == 0:
                    del open_pos[sym]

            c_ptr += 1

        # open events
        while o_ptr < len(opens) and opens.loc[o_ptr, "Entry Time"] <= t:
            r = opens.loc[o_ptr]
            sym = r["Market"]
            cm = market_cfg[sym].get("cost_model", {})
            pos = {
                "Direction": r.get("Direction", None),
                "Units": float(r["Units"]),
                "EntryPrice": float(r["Entry Price"]) if "Entry Price" in r else np.nan,  # executed
                "EntryMid": float(r["Entry Mid"]) if "Entry Mid" in r else np.nan,  # valfritt, kan behållas
                "SpreadPips": float(cm.get("fixed_spread_pips", 0.0)),
                "SlipPips": float(cm.get("slippage_pips", 0.0)),
            }
            open_pos.setdefault(sym, []).append(pos)
            o_ptr += 1

        # 4) MTM orealiserad PnL
        unreal = 0.0
        for sym, plist in open_pos.items():
            mid_now = float(price_df.loc[t, sym])  # vi behandlar close som mid approx i MTM
            pip_size = float(market_cfg[sym]["pip_size"])

            for pos in plist:
                units = float(pos["Units"])
                direction = pos.get("Direction", None)

                entry_exec = float(pos.get("EntryPrice", np.nan))  # executed entry (inkl costs)
                if not np.isfinite(entry_exec) or not np.isfinite(mid_now):
                    continue

                # Kostnader för att "likvidera" just nu (konservativt)
                spread_pips = float(pos.get("SpreadPips", 0.0))
                slip_pips = float(pos.get("SlipPips", 0.0))

                spread_px = spread_pips * pip_size
                slip_px = slip_pips * pip_size

                if direction == "LONG":
                    # Om vi stänger long nu: vi säljer på bid ≈ mid - halfspread, plus slippage
                    liquidation = mid_now - 0.5 * spread_px - slip_px
                    pnl_price_per_unit = liquidation - entry_exec
                else:
                    # Om vi stänger short nu: vi köper på ask ≈ mid + halfspread, plus slippage
                    liquidation = mid_now + 0.5 * spread_px + slip_px
                    pnl_price_per_unit = entry_exec - liquidation

                unreal += usd_pnl_from_price_delta(sym, pip_size, mid_now, pnl_price_per_unit, units)

        mtm_equity = cash_equity + unreal
        mtm_records.append({
            "Time": t,
            "Equity_MTM": mtm_equity,
            "Cash": cash_equity,
            "Unreal": unreal
        })

    mtm_df = pd.DataFrame(mtm_records).set_index("Time")

    # 5) Intraday drawdown:
    #    (a) Total max DD på 5m
    mtm_df["RollMax"] = mtm_df["Equity_MTM"].cummax()
    mtm_df["DD_$"] = mtm_df["Equity_MTM"] - mtm_df["RollMax"]
    mtm_df["DD_%"] = mtm_df["DD_$"] / mtm_df["RollMax"]

    max_dd_usd = float(mtm_df["DD_$"].min())
    max_dd_pct = float(mtm_df["DD_%"].min())

    #    (b) Intraday max DD: reset peak varje dag
    g = mtm_df.groupby(mtm_df.index.date)
    daily_peak = g["Equity_MTM"].cummax()
    intraday_dd = mtm_df["Equity_MTM"] - daily_peak
    mtm_df["Intraday_DD_$"] = intraday_dd
    mtm_df["Intraday_DD_%"] = intraday_dd / daily_peak

    max_intraday_dd_usd = float(mtm_df["Intraday_DD_$"].min())
    max_intraday_dd_pct = float(mtm_df["Intraday_DD_%"].min())

    dd_summary = {
        "Max DD MTM ($)": abs(max_dd_usd),
        "Max DD MTM (%)": abs(max_dd_pct) * 100.0,
        "Max Intraday DD MTM ($)": abs(max_intraday_dd_usd),
        "Max Intraday DD MTM (%)": abs(max_intraday_dd_pct) * 100.0,
    }
    print(sym, cm)
    return dd_summary, mtm_df


def compute_risk_metrics_from_equity(eq_series: pd.Series, resample_rule: str = "D") -> dict:
    """
    eq_series: pd.Series med datetimeindex och equity-värden.
    Returnerar sharpe/sortino/cagr/calmar + maxdd.
    """
    out = {
        "Sharpe (daily, ann.)": np.nan,
        "Sortino (daily, ann.)": np.nan,
        "CAGR (%)": np.nan,
        "Calmar": np.nan,
        "Max Drawdown ($)": np.nan,
        "Max Drawdown (%)": np.nan,
        "Days": np.nan,
        "Years": np.nan,
    }
    if eq_series is None or eq_series.empty or len(eq_series) < 2:
        return out

    eq = eq_series.copy()
    eq = eq[~eq.index.duplicated(keep="last")].sort_index()

    # Resample till daglig equity (eller valfri regel), forward fill
    eq_r = eq.resample(resample_rule).last().ffill()
    ret = eq_r.pct_change().dropna()
    if len(ret) < 30:
        # för få datapunkter för stabila annualiserade mått
        # men vi kan ändå räkna DD och CAGR
        pass

    # Drawdown
    roll_max = eq_r.cummax()
    dd = eq_r - roll_max
    max_dd_usd = float(dd.min())
    max_dd_pct = float((dd / roll_max).min())

    out["Max Drawdown ($)"] = abs(max_dd_usd)
    out["Max Drawdown (%)"] = abs(max_dd_pct) * 100.0

    ann_factor = annualize_factor_from_resample(resample_rule)

    # Sharpe / Sortino
    if len(ret) >= 30:
        mu = ret.mean()
        sd = ret.std(ddof=1)
        sharpe = (mu / sd) * np.sqrt(ann_factor) if sd and sd > 0 else np.nan

        mar = 0.0
        downside_sq = np.minimum(0.0, ret - mar) ** 2
        downside_dev = np.sqrt(downside_sq.mean())
        sortino = (mu / downside_dev) * np.sqrt(ann_factor) if downside_dev and downside_dev > 0 else np.nan

        out["Sharpe (daily, ann.)"] = float(sharpe) if np.isfinite(sharpe) else np.nan
        out["Sortino (daily, ann.)"] = float(sortino) if np.isfinite(sortino) else np.nan

    # CAGR
    start_val = float(eq_r.iloc[0])
    end_val = float(eq_r.iloc[-1])
    days = (eq_r.index[-1] - eq_r.index[0]).total_seconds() / 86400.0
    years = days / 365.0 if days > 0 else np.nan
    out["Days"] = float(days) if np.isfinite(days) else np.nan
    out["Years"] = float(years) if np.isfinite(years) else np.nan

    if days > 0 and start_val > 0:
        cagr = (end_val / start_val) ** (ann_factor / days) - 1.0
        out["CAGR (%)"] = float(cagr * 100.0) if np.isfinite(cagr) else np.nan

        max_dd_frac = abs(max_dd_pct)
        calmar = (cagr / max_dd_frac) if (np.isfinite(cagr) and max_dd_frac and max_dd_frac > 0) else np.nan
        out["Calmar"] = float(calmar) if np.isfinite(calmar) else np.nan

    return out


def build_daily_returns_matrix_from_port_log(
        port_log: pd.DataFrame,
        markets: list[str],
        start_capital: float = 50_000.0,
        date_col: str = "Exit Time",
        pnl_col: str = "PnL_USD",
) -> pd.DataFrame:
    """
    Skapar daily returns per market från port_log (realiserad PnL vid Exit Time).
    Returnerar df: index=Date, columns=markets.
    """
    log = port_log.copy()
    log[date_col] = pd.to_datetime(log[date_col])
    log["Date"] = log[date_col].dt.floor("D")

    daily_pnl = (
        log.groupby(["Date", "Market"])[pnl_col]
        .sum()
        .unstack("Market")
        .reindex(columns=markets)
        .fillna(0.0)
    )

    # Enkel normalisering (Carver-style baseline): dela med startkapital
    daily_ret = daily_pnl / float(start_capital)
    return daily_ret

# ==========================
# KÖR BACKTEST + SLUTSUMMERING + COMBINED EQUITY & STATS
# ==========================

all_results = []
all_trades = []


def iter_market_items(markets_obj):
    # Tillåt både list-of-dicts och dict(name->cfg)
    if isinstance(markets_obj, dict):
        for name, cfg in markets_obj.items():
            # om cfg redan är dict med csv/pip_size osv
            if isinstance(cfg, dict):
                cfg = cfg.copy()
                cfg.setdefault("name", name)
                yield cfg
            else:
                # om någon råkat lägga en str här
                yield {"name": str(name), "csv": None, "pip_size": None}
    else:
        for item in markets_obj:
            yield item


for m in iter_market_items(markets):
    try:
        if not isinstance(m, dict):
            raise TypeError(f"Market config är inte dict: {type(m)} value={m}")

        stats, trades_df = run_backtest_for_market(
            m["name"],
            m["csv"],
            m["pip_size"],
            m.get("spread_points_per_pip", 10.0),
            cost_model=m.get("cost_model", None),
            session_start=m.get("session_start", "08:00:00"),
            session_end=m.get("session_end", "16:00:00"),
        )

        if stats is not None and trades_df is not None:
            trades_df["Market"] = m["name"]
            all_results.append(stats)
            all_trades.append(trades_df)

    except Exception as e:
        name = m.get("name", str(m)) if isinstance(m, dict) else str(m)
        csv_ = m.get("csv", "?") if isinstance(m, dict) else "?"
        print(f"\n*** FEL för {name} ({csv_}): {e}\n")

# ==========================
# PORTFÖLJ: samla trades + market lookup
# ==========================

if all_trades:
    portfolio_trades = pd.concat(all_trades, ignore_index=True)
    portfolio_trades["Entry Time"] = pd.to_datetime(portfolio_trades["Entry Time"])
    portfolio_trades["Exit Time"] = pd.to_datetime(portfolio_trades["Exit Time"])

    market_cfg = {m["name"]: m for m in markets}

    rates_df = build_rates_df_for_portfolio(portfolio_trades, fx_rates, freq="5min")

    t = rates_df.index[len(rates_df) // 2]

    print("USD per JPY (should be ~0.006-0.01):", quote_to_usd_rate("JPY", t, rates_df))
    print("USD per CHF (should be ~1.0-1.2):", quote_to_usd_rate("CHF", t, rates_df))
    print("USD per GBP (should be ~1.1-1.5):", base_to_usd_rate("GBP", t, rates_df))

    # Pip values per 1 unit base:
    print("USDJPY pip $ per 1 USD:", pip_value_usd_per_unit("USDJPY", 0.01, t, rates_df))  # ~0.00006-0.0001
    print("USDCHF pip $ per 1 USD:", pip_value_usd_per_unit("USDCHF", 0.0001, t, rates_df))  # ~0.00008-0.0001
    print("GBPJPY pip $ per 1 GBP:", pip_value_usd_per_unit("GBPJPY", 0.01, t, rates_df))  # ~0.00006-0.0001

    print("rates_df columns:", rates_df.columns.tolist())
    print("rates_df index head/tail:", rates_df.index.min(), rates_df.index.max())
    print(rates_df.tail())

    port_summary, port_eq, port_log = simulate_portfolio_equal_risk(
        portfolio_trades,
        market_cfg,
        rates_df=rates_df,
        start_capital=50_000.0,
        max_one_position_per_market=True,
    )


    # ==========================
    # SANITY CHECKS (måste matcha)
    # ==========================

    plt.figure(figsize=(12, 5))
    plt.plot(port_eq["Time"], port_eq["Equity"])
    plt.title("Portfolio Equity Curve ($)")
    plt.xlabel("Time")
    plt.ylabel("Equity ($)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    dd_summary, mtm_df = compute_portfolio_mtm_equity_and_intraday_dd(
        port_log,
        market_cfg,
        start_capital=50_000.0,
        freq="5min",
        price_col="close",
    )

    # Lägg in i port_summary
    port_summary.update(dd_summary)

    print("\n--- MTM / INTRADAY DD ---")
    for k, v in dd_summary.items():
        print(f"{k}: {v:.4f}")

    # ===== NYTT: Riskmått från MTM equity istället för "stegig" realiserad equity =====
    mtm_metrics = compute_risk_metrics_from_equity(mtm_df["Equity_MTM"], resample_rule="D")

    print("\n" + "=" * 70)
    print("SANITY CHECKS")
    print("=" * 70)

    print("Sum PnL_USD:", float(port_log["PnL_USD"].sum()))
    print("Equity change:", float(port_summary["End Equity"] - port_summary["Start Capital"]))
    print("Diff:", float(port_log["PnL_USD"].sum() - (port_summary["End Equity"] - port_summary["Start Capital"])))

    print("Avg Units:", port_log["Units"].mean())
    print("Median Units:", port_log["Units"].median())

    tmp = portfolio_trades.copy()
    tmp["pip_size"] = tmp["Market"].map(lambda s: float(market_cfg[s]["pip_size"]))
    tmp["pips"] = tmp["pnl"] / tmp["pip_size"]

    print("Avg pips/trade:", tmp["pips"].mean())
    print("Median pips/trade:", tmp["pips"].median())

    print("\n$ per pip (approx) per market @ avg units (using median Entry Price):")
    for sym in sorted(port_log["Market"].unique()):
        sub = port_log[port_log["Market"] == sym].dropna(subset=["Entry Price"])
        if sub.empty:
            continue

        avg_units_sym = float(sub["Units"].mean())
        pip_size_sym = float(market_cfg[sym]["pip_size"])
        price_sym = float(sub["Entry Price"].median())

        t_ref = rates_df.index[len(rates_df) // 2]  # eller median exit/entry tid om du vill
        usd_per_pip_sym = approx_usd_per_pip_general(sym, pip_size_sym, avg_units_sym, t_ref, rates_df)
        print(f"{sym}: {usd_per_pip_sym:.4f}")

    # --- Correct $/pip sanity check (works for all majors) ---
    if not port_log.empty:
        avg_units = float(port_log["Units"].mean())

        # välj en representativ trade för att få pris (median entry price)
        sample = port_log.dropna(subset=["Entry Price"]).copy()
        if not sample.empty:
            sym0 = sample["Market"].iloc[0]
            pip_size0 = float(market_cfg[sym0]["pip_size"])
            price0 = float(sample["Entry Price"].median())

            usd_per_pip = approx_usd_per_pip(sym0, pip_size0, avg_units, price0)
            print("Approx $ per pip @ avg units:", usd_per_pip)
        else:
            print("Approx $ per pip @ avg units: N/A (no Entry Price)")

    print("\n" + "=" * 70)
    print(" PORTFÖLJ-RESULTAT (USD) ")
    print("=" * 70)

    # Skriv över portföljens riskmått så de blir realistiska
    for k in ["Sharpe (daily, ann.)", "Sortino (daily, ann.)", "CAGR (%)", "Calmar", "Days", "Years",
              "Max Drawdown ($)", "Max Drawdown (%)"]:
        if k in mtm_metrics:
            port_summary[k] = mtm_metrics[k]

    for k, v in port_summary.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

else:
    print("Inga trades att simulera i portföljen.")