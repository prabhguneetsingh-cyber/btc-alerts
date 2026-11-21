#!/usr/bin/env python3
"""
FOUR ADAPTIVE LINES PRO -> Telegram alerts (free alternative to TradingView alerts)
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ---------------- settings (same as the Pine inputs) ----------------
LEN1, LEN2, LEN3, LEN4 = 5, 11, 14, 33
ADX_LEN = 20
MIN_ADX = 22.0
USE_HTF = True
USE_ADX = True
TP_PCT = 1.0
SL_PCT = 1.0
COOLDOWN = 4
LOOKBACK = 3  # on every run, also check this many most recent closed candles (catches late runs)
BARS = 1000  # history to load; the adaptive lines need a few hundred bars to settle

UNITS = {"m": 60, "h": 3600, "d": 86400}


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


def compute(df: pd.DataFrame, htf: pd.DataFrame, htf_sec: int):
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


def fetch(args, symbol: str, tf: str) -> pd.DataFrame:
    df = fetch_ccxt(args.exchange, symbol, tf) if args.source == "ccxt" else fetch_yf(symbol, tf)
    return closed_only(df, tf)


# ---------------- notifications ----------------
def telegram(text: str) -> None:
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print("[Telegram not configured]\n" + text)
        return
    r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      data={"chat_id": chat, "text": text}, timeout=15)
    r.raise_for_status()


# ---------------- main loop ----------------
def check(args, symbol: str, tf: str, htf_tf: str) -> None:
    state_file = Path(f".state_{symbol.replace('/', '_')}_{tf}.json")
    ltf, htf = fetch(args, symbol, tf), fetch(args, symbol, htf_tf)
    buy, sell = compute(ltf, htf, tf_seconds(htf_tf))

    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    sent = set(state.get("sent", []))
    if state.get("last"):  # older state format
        sent.add(state["last"])

    tf_sec = tf_seconds(tf)
    now = pd.Timestamp.now(tz="UTC")
    n = len(ltf)
    price_now = float(ltf["close"].iloc[-1])

    for i in range(max(0, n - LOOKBACK), n):  # oldest first
        if not (buy[i] or sell[i]):
            continue
        bar_open = ltf["time"].iloc[i]
        key = str(bar_open)
        if key in sent:
            continue  # already alerted for this bar

        price = float(ltf["close"].iloc[i])
        if buy[i]:
            side, tp, sl = "BUY", price * (1 + TP_PCT / 100), price * (1 - SL_PCT / 100)
        else:
            side, tp, sl = "SELL", price * (1 - TP_PCT / 100), price * (1 + SL_PCT / 100)
        bar_close = bar_open + pd.Timedelta(seconds=tf_sec)
        age_min = max(0.0, (now - bar_close).total_seconds() / 60)
        moved = (price_now / price - 1) * 100

        telegram(f"FOUR LINE SYSTEM {side}\n{symbol} ({tf})\n"
                 f"Signal candle closed {age_min:.0f} min ago ({bar_close:%H:%M} UTC)\n"
                 f"Entry ~ {price:.5g}\n"
                 f"Price now ~ {price_now:.5g} ({moved:+.2f}% since entry)\n"
                 f"TP {tp:.5g} (+{TP_PCT}%)\nSL {sl:.5g} (-{SL_PCT}%)")
        sent.add(key)
        state_file.write_text(json.dumps({"sent": sorted(sent)[-20:]}))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["ccxt", "yf"], default="ccxt")
    p.add_argument("--exchange", default="binance", help="ccxt exchange id")
    p.add_argument("--symbol", default="BTC/USDT", help="one or more, comma separated")
    p.add_argument("--tf", default="5m,15m", help="chart timeframes, comma separated: 1m 5m 15m 30m 1h 1d")
    p.add_argument("--htf", default="15m", help="higher timeframe filter (yours is set to 15 min)")
    p.add_argument("--poll", type=int, default=60, help="seconds between checks")
    p.add_argument("--once", action="store_true", help="check once and exit (for cron)")
    p.add_argument("--ping", action="store_true", help="send a test Telegram message and exit")
    args = p.parse_args()

    if args.ping:
        telegram("Four Line alerts: test message OK")
        return

    symbols = [x.strip() for x in args.symbol.split(",")]
    tfs = [x.strip() for x in args.tf.split(",")]
    print(f"Watching {symbols} on {tfs} (HTF {args.htf}) via {args.source}. Ctrl+C to stop.")
    while True:
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
