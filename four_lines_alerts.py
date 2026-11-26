#!/usr/bin/env python3
"""
FOUR ADAPTIVE LINES PRO -> Telegram alerts (free alternative to TradingView alerts)

Port of the Pine v6 strategy:
  BUY  = close crosses above Line 2 + bull stack + all slopes up + +DI > -DI
         + ADX >= 22 + higher-timeframe (last CLOSED HTF bar) above its Line 4
  SELL = mirror image
  Signals are evaluated on closed bars only, with the same 4-bar cooldown.

Setup
  pip install pandas numpy requests ccxt yfinance
  export TELEGRAM_BOT_TOKEN="123456:ABC..."   # from @BotFather
  export TELEGRAM_CHAT_ID="123456789"         # your chat id

Examples
  python four_lines_alerts.py --ping
  python four_lines_alerts.py --source ccxt --exchange binance --symbol BTC/USDT --tf 5m,15m --htf 5m
  python four_lines_alerts.py --source yf --symbol RELIANCE.NS --tf 5m --htf 5m
  python four_lines_alerts.py --source yf --symbol EURUSD=X --tf 5m,15m --htf 5m --once
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
np.seterr(divide="ignore", invalid="ignore")

# ---------------- settings (same as the Pine inputs) ----------------
LEN1, LEN2, LEN3, LEN4 = 5, 9, 31, 39
ADX_LEN = 22
MIN_ADX = 21.0
USE_HTF = True
USE_ADX = True
TP_PCT = 2.0
SL_PCT = 1.0
COOLDOWN = 4
LOOKBACK = 6  # on every run, also check this many most recent closed candles (catches late runs)
DRY_RUN = False  # --dry-run: print what would be sent, send nothing, save nothing
DEBUG_ROWS = 12
BARS = 1000  # history to load; the adaptive lines need a few hundred bars to settle

UNITS = {"m": 60, "h": 3600, "d": 86400}


def px(x: float) -> str:
    """Price text: 2 decimals for big prices (BTC, gold), 5 significant digits for small ones."""
    return f"{x:.2f}" if x >= 100 else f"{x:.5g}"


def tf_seconds(tf: str) -> int:
    return int(tf[:-1]) * UNITS[tf[-1]]


# ---------------- indicator math ----------------
def adaptive(close: pd.Series, length: int) -> pd.Series:
    """Kaufman-style adaptive MA, identical to the Pine adaptive() function."""
    change = (close - close.shift(length)).abs()
    vol = close.diff().abs().rolling(length).sum()  # sma(...) * length
    eff = (change / vol.where(vol > 0)).fillna(0.0)
    fast, slow = 2.0 / 3.0, 2.0 / 31.0
    sc = ((eff * (fast - slow) + slow) ** 2).to_numpy(float)
    c = close.to_numpy(float)
    out = np.empty(len(c))
    out[0] = c[0]
    for i in range(1, len(c)):
        out[i] = out[i - 1] + sc[i] * (c[i] - out[i - 1])
    return pd.Series(out, index=close.index)


def rma(x, n: int) -> np.ndarray:
    """Wilder's RMA, seeded with an SMA like Pine's ta.rma."""
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    valid = np.flatnonzero(~np.isnan(x))
    if len(valid) < n:
        return out
    start = valid[0] + n - 1
    out[start] = x[valid[0]:start + 1].mean()
    for i in range(start + 1, len(x)):
        out[i] = (out[i - 1] * (n - 1) + x[i]) / n
    return out


def dmi(df: pd.DataFrame, n: int):
    h, l, c = df["high"], df["low"], df["close"]
    up, down = h.diff(), -l.diff()
    plus_dm = np.where(up.isna(), np.nan, np.where((up > down) & (up > 0), up, 0.0))
    minus_dm = np.where(down.isna(), np.nan, np.where((down > up) & (down > 0), down, 0.0))
    pc = c.shift()
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    trur = rma(tr, n)
    plus = pd.Series(100 * rma(plus_dm, n) / trur, index=df.index).ffill()
    minus = pd.Series(100 * rma(minus_dm, n) / trur, index=df.index).ffill()
    s = plus + minus
    dx = (plus - minus).abs() / s.where(s != 0, 1.0)
    adx = 100 * rma(dx.to_numpy(), n)
    return plus.to_numpy(), minus.to_numpy(), adx


def compute(df: pd.DataFrame, htf: pd.DataFrame, htf_sec: int, details: dict = None):
    """Return (buy, sell) boolean arrays aligned with df. df/htf need a UTC 'time' column (bar open)."""
    c = df["close"]
    l1, l2, l3, l4 = (adaptive(c, n) for n in (LEN1, LEN2, LEN3, LEN4))

    bull_stack = ((l1 > l2) & (l2 > l3) & (l3 > l4)).to_numpy()
    bear_stack = ((l1 < l2) & (l2 < l3) & (l3 < l4)).to_numpy()
    bull_slope = ((l1 > l1.shift()) & (l2 > l2.shift()) & (l3 > l3.shift()) & (l4 > l4.shift())).to_numpy()
    bear_slope = ((l1 < l1.shift()) & (l2 < l2.shift()) & (l3 < l3.shift()) & (l4 < l4.shift())).to_numpy()

    # Higher timeframe: Pine uses close[1] / line[1] with lookahead_on, i.e. the last CLOSED HTF bar.
    h = htf.copy()
    h["line"] = adaptive(h["close"], LEN4)
    h["avail"] = h["time"] + pd.Timedelta(seconds=htf_sec)
    left = pd.DataFrame({"time": df["time"].astype("datetime64[ns, UTC]")})
    right = h[["avail", "close", "line"]].copy()
    right["avail"] = right["avail"].astype("datetime64[ns, UTC]")
    m = pd.merge_asof(left, right, left_on="time", right_on="avail", direction="backward")
    htf_bull = (m["close"] > m["line"]).to_numpy()
    htf_bear = (m["close"] < m["line"]).to_numpy()
    htf_buy_ok = (not USE_HTF) | htf_bull
    htf_sell_ok = (not USE_HTF) | htf_bear

    plus, minus, adx = dmi(df, ADX_LEN)
    adx_ok = (not USE_ADX) | (adx >= MIN_ADX)

    cross_up = ((c > l2) & (c.shift() <= l2.shift())).to_numpy()
    cross_dn = ((c < l2) & (c.shift() >= l2.shift())).to_numpy()

    buy_setup = cross_up & bull_stack & bull_slope & (plus > minus) & adx_ok & htf_buy_ok
    sell_setup = cross_dn & bear_stack & bear_slope & (minus > plus) & adx_ok & htf_sell_ok

    n = len(df)
    buy, sell = np.zeros(n, bool), np.zeros(n, bool)
    last = None
    for i in np.flatnonzero(buy_setup | sell_setup):
        if last is None or i - last > COOLDOWN:
            (buy if buy_setup[i] else sell)[i] = True
            last = i
    if details is not None:
        details.update(l2=l2.to_numpy(), cross_up=cross_up, cross_dn=cross_dn,
                       bull_stack=bull_stack, bear_stack=bear_stack,
                       bull_slope=bull_slope, bear_slope=bear_slope,
                       plus=plus, minus=minus, adx=adx,
                       htf_bull=htf_bull, htf_bear=htf_bear,
                       htf_close=m["close"].to_numpy(), htf_line=m["line"].to_numpy())
    return buy, sell


# ---------------- data sources ----------------
def closed_only(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    now = pd.Timestamp.now(tz="UTC")
    keep = df["time"] + pd.Timedelta(seconds=tf_seconds(tf)) <= now
    return df[keep].reset_index(drop=True)


def fetch_ccxt(exchange: str, symbol: str, tf: str) -> pd.DataFrame:
    import ccxt
    ex = getattr(ccxt, exchange)({"enableRateLimit": True})
    raw = ex.fetch_ohlcv(symbol, tf, limit=BARS)
    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df


def fetch_yf(symbol: str, tf: str) -> pd.DataFrame:
    import yfinance as yf
    interval = {"1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m", "1h": "60m", "1d": "1d"}[tf]
    period = {"1m": "7d", "5m": "59d", "15m": "59d", "30m": "59d", "60m": "730d", "1d": "5y"}[interval]
    df = yf.download(symbol, interval=interval, period=period, auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower).reset_index()
    df["time"] = pd.to_datetime(df.iloc[:, 0], utc=True)
    return df[["time", "open", "high", "low", "close"]].dropna().tail(BARS)


def fetch_twelvedata(symbol: str, tf: str) -> pd.DataFrame:
    key = os.environ.get("TWELVEDATA_API_KEY")
    if not key:
        raise RuntimeError("TWELVEDATA_API_KEY is not set")
    interval = {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min",
                "1h": "1h", "4h": "4h", "1d": "1day"}[tf]
    try:
        r = requests.get("https://api.twelvedata.com/time_series",
                         params={"symbol": symbol, "interval": interval, "outputsize": BARS,
                                 "timezone": "UTC", "apikey": key}, timeout=20)
        data = r.json()
    except (requests.RequestException, ValueError) as e:
        # never include the request URL in the message (it contains the API key)
        raise RuntimeError(f"Twelve Data request failed ({type(e).__name__})") from None
    if data.get("status") == "error" or "values" not in data:
        raise RuntimeError(f"Twelve Data: {data.get('message', 'no data returned')}")
    df = pd.DataFrame(data["values"])
    df["time"] = pd.to_datetime(df["datetime"], utc=True)  # start time of each candle
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)
    return df[["time", "open", "high", "low", "close"]].sort_values("time").reset_index(drop=True)


_CACHE = {}  # one download per symbol/timeframe per run (saves free-plan requests)


def fetch(args, symbol: str, tf: str) -> pd.DataFrame:
    k = (args.source, args.exchange, symbol, tf)
    if k not in _CACHE:
        if args.source == "ccxt":
            df = fetch_ccxt(args.exchange, symbol, tf)
        elif args.source == "twelvedata":
            df = fetch_twelvedata(symbol, tf)
        else:
            df = fetch_yf(symbol, tf)
        _CACHE[k] = closed_only(df, tf)
    return _CACHE[k]


# ---------------- notifications ----------------
def telegram(text: str) -> None:
    """Send to every chat id in TELEGRAM_CHAT_ID (comma separated). Only fails if NO chat received it."""
    if DRY_RUN:
        print("[dry run] would send:\n" + text.replace("\n", "\n    "))
        return
    token, chats = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID", "")
    chat_ids = [c.strip() for c in chats.split(",") if c.strip()]
    if not token or not chat_ids:
        print("[Telegram not configured]\n" + text)
        return
    delivered = 0
    for chat in chat_ids:
        try:
            r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                              data={"chat_id": chat, "text": text}, timeout=15)
            if r.ok:
                delivered += 1
            else:
                print(f"Telegram: could not send to the chat ending ...{chat[-3:]} (HTTP {r.status_code})")
        except requests.RequestException as e:  # never print the URL (it contains the bot token)
            print(f"Telegram: could not send to the chat ending ...{chat[-3:]} ({type(e).__name__})")
    if delivered == 0:
        raise RuntimeError("Telegram: the message could not be delivered to any chat")


# ---------------- main loop ----------------
def print_debug(symbol: str, tf: str, ltf: pd.DataFrame, d: dict, buy, sell, rows: int = 12) -> None:
    """Prints why each recent candle did or did not produce a signal (shows up in the GitHub run log)."""
    tf_sec = tf_seconds(tf)
    n = len(ltf)
    print(f"--- debug {symbol} {tf}: candle CLOSE time in UTC ---")
    print("close  | price     | line2     | cross | stack | slope | DI | adx  | htf | htfClose  | htfLine   | signal")
    for i in range(max(0, n - rows), n):
        t = ltf["time"].iloc[i] + pd.Timedelta(seconds=tf_sec)
        cross = "UP" if d["cross_up"][i] else ("DN" if d["cross_dn"][i] else "-")
        stack = "B" if d["bull_stack"][i] else ("S" if d["bear_stack"][i] else "-")
        slope = "B" if d["bull_slope"][i] else ("S" if d["bear_slope"][i] else "-")
        di = "P>M" if d["plus"][i] > d["minus"][i] else "M>P"
        htf = "B" if d["htf_bull"][i] else ("S" if d["htf_bear"][i] else "-")
        sig = "BUY" if buy[i] else ("SELL" if sell[i] else "-")
        print(f"{t:%H:%M}  | {ltf['close'].iloc[i]:<9.1f} | {d['l2'][i]:<9.1f} | {cross:<5} | {stack:<5} | {slope:<5} | {di} | "
              f"{d['adx'][i]:<4.1f} | {htf:<3} | {d['htf_close'][i]:<9.1f} | {d['htf_line'][i]:<9.1f} | {sig}")


def check(args, symbol: str, tf: str, htf_tf: str) -> None:
    state_file = Path(f".state_{symbol.replace('/', '_')}_{tf}.json")
    ltf, htf = fetch(args, symbol, tf), fetch(args, symbol, htf_tf)
    dbg = {}
    buy, sell = compute(ltf, htf, tf_seconds(htf_tf), dbg)
    print_debug(symbol, tf, ltf, dbg, buy, sell, DEBUG_ROWS)

    state = {} if DRY_RUN or not state_file.exists() else json.loads(state_file.read_text())
    sent = set(state.get("sent", []))
    if state.get("last"):  # older state format
        sent.add(state["last"])
    trade = state.get("trade")  # the currently open trade, if any

    def save() -> None:
        if DRY_RUN:
            return
        state_file.write_text(json.dumps({"sent": sorted(sent)[-20:], "trade": trade}))

    tf_sec = tf_seconds(tf)
    now = pd.Timestamp.now(tz="UTC")
    n = len(ltf)
    price_now = float(ltf["close"].iloc[-1])
    keys = [str(t) for t in ltf["time"]]
    hi = ltf["high"].to_numpy(float)
    lo = ltf["low"].to_numpy(float)
    cl = ltf["close"].to_numpy(float)

    start = max(0, n - LOOKBACK)
    trade_idx = -1
    if trade:
        if trade["bar"] in keys:
            trade_idx = keys.index(trade["bar"])
            start = min(start, trade_idx + 1)
        else:
            print("open trade dropped: its entry candle is no longer in the loaded data")
            trade = None
            save()

    for i in range(start, n):  # oldest first
        bar_open = ltf["time"].iloc[i]
        bar_close = bar_open + pd.Timedelta(seconds=tf_sec)
        age_min = max(0.0, (now - bar_close).total_seconds() / 60)
        when = f"candle closed {age_min:.0f} min ago ({bar_close:%H:%M} UTC)"

        # 1) did the open trade reach its take profit or stop loss on this candle?
        if trade is not None and i > trade_idx:
            is_long = trade["side"] == "BUY"
            hit_tp = hi[i] >= trade["tp"] if is_long else lo[i] <= trade["tp"]
            hit_sl = lo[i] <= trade["sl"] if is_long else hi[i] >= trade["sl"]
            if hit_tp or hit_sl:
                if hit_tp and hit_sl:
                    head, result = "TP AND SL BOTH TOUCHED IN ONE CANDLE (order unknown, check your chart)", ""
                elif hit_tp:
                    head, result = "TAKE PROFIT HIT", f"\nResult: +{TP_PCT}%"
                else:
                    head, result = "STOP LOSS HIT", f"\nResult: -{SL_PCT}%"
                telegram(f"{head} ({trade['side']})\n{symbol} ({tf})\n"
                         f"Entry {px(trade['entry'])} | TP {px(trade['tp'])} | SL {px(trade['sl'])}\n"
                         f"Hit on the {when}{result}")
                trade = None
                save()

        # 2) new signal on this candle?
        side = "BUY" if buy[i] else ("SELL" if sell[i] else None)
        if side is None:
            continue

        # the strategy reverses on an opposite signal: close the open trade at this candle's close
        if trade is not None and i > trade_idx and trade["side"] != side:
            pnl = (cl[i] / trade["entry"] - 1) * 100
            if trade["side"] == "SELL":
                pnl = -pnl
            telegram(f"OPPOSITE SIGNAL - strategy closes the {trade['side']} ({symbol} {tf})\n"
                     f"Entry {px(trade['entry'])}, closed at {px(cl[i])}\n"
                     f"Result: {pnl:+.2f}%\n{when[0].upper() + when[1:]}")
            trade = None
            save()

        key = str(bar_open)
        if key in sent or i < n - LOOKBACK:
            continue  # already alerted, or too old to alert

        price = cl[i]
        if side == "BUY":
            tp, sl = price * (1 + TP_PCT / 100), price * (1 - SL_PCT / 100)
        else:
            tp, sl = price * (1 - TP_PCT / 100), price * (1 + SL_PCT / 100)
        moved = (price_now / price - 1) * 100
        note = ""
        if trade is not None:
            note = f"\nNote: a {trade['side']} from earlier is still open (the strategy ignores repeat signals)"

        telegram(f"FOUR LINE SYSTEM {side}\n{symbol} ({tf})\n"
                 f"Signal {when}\n"
                 f"Entry ~ {px(price)}\n"
                 f"Price now ~ {px(price_now)} ({moved:+.2f}% since entry)\n"
                 f"TP {px(tp)} (+{TP_PCT}%)\nSL {px(sl)} (-{SL_PCT}%){note}")
        sent.add(key)
        if trade is None:
            trade = {"side": side, "entry": price, "tp": tp, "sl": sl, "bar": key}
            trade_idx = i
        save()


def main() -> None:
    global LEN1, LEN2, LEN3, LEN4, ADX_LEN, MIN_ADX, TP_PCT, SL_PCT, COOLDOWN, LOOKBACK, USE_HTF, USE_ADX, DRY_RUN, DEBUG_ROWS
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["ccxt", "twelvedata", "yf"], default="ccxt")
    p.add_argument("--exchange", default="binance", help="ccxt exchange id")
    p.add_argument("--symbol", default="BTC/USDT", help="one or more, comma separated")
    p.add_argument("--tf", default="5m,15m", help="chart timeframes, comma separated: 1m 5m 15m 30m 1h 1d")
    p.add_argument("--htf", default="15m", help="higher timeframe filter (yours is set to 15 min)")
    # per-asset indicator settings (defaults = your BTC settings)
    p.add_argument("--lens", default="5,11,14,33", help="Line 1-4 lengths, e.g. 5,11,14,33")
    p.add_argument("--adx-len", type=int, default=20)
    p.add_argument("--min-adx", type=float, default=22.0)
    p.add_argument("--tp", type=float, default=1.0, help="take profit %%")
    p.add_argument("--sl", type=float, default=1.0, help="stop loss %%")
    p.add_argument("--cooldown", type=int, default=4)
    p.add_argument("--lookback", type=int, default=6, help="closed candles re-checked on every run")
    p.add_argument("--no-htf", action="store_true", help="turn the higher timeframe filter off")
    p.add_argument("--no-adx", action="store_true", help="turn the ADX filter off")
    p.add_argument("--dry-run", action="store_true", help="print alerts instead of sending; save nothing")
    p.add_argument("--debug-rows", type=int, default=12, help="candles shown in the log table")
    p.add_argument("--poll", type=int, default=60, help="seconds between checks")
    p.add_argument("--once", action="store_true", help="check once and exit (for cron)")
    p.add_argument("--ping", action="store_true", help="send a test Telegram message and exit")
    args = p.parse_args()
    LEN1, LEN2, LEN3, LEN4 = [int(x) for x in args.lens.split(",")]
    ADX_LEN, MIN_ADX, TP_PCT, SL_PCT = args.adx_len, args.min_adx, args.tp, args.sl
    COOLDOWN, LOOKBACK = args.cooldown, args.lookback
    USE_HTF, USE_ADX = not args.no_htf, not args.no_adx
    DRY_RUN, DEBUG_ROWS = args.dry_run, args.debug_rows

    if args.ping:
        telegram("Four Line alerts: test message OK")
        return

    symbols = [x.strip() for x in args.symbol.split(",")]
    tfs = [x.strip() for x in args.tf.split(",")]
    print(f"Watching {symbols} on {tfs} (HTF {args.htf}) via {args.source}. Ctrl+C to stop.")
    while True:
        _CACHE.clear()
        for symbol in symbols:
            for tf in tfs:
                try:
                    check(args, symbol, tf, args.htf)
                except Exception as e:  # network blips etc. shouldn't kill the watcher
                    print(f"error {symbol} {tf}:", e)
        if args.once:
            break
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
