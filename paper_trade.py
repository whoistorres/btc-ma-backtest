"""
Paper trading for the BTC MA5 / MA100 strategy. FAKE MONEY ONLY - no orders are placed.

Run once a day, after Bitcoin's daily close (00:00 UTC = 8:00 PM in Puerto Rico).
Each run:
  1. Downloads daily BTC closes and drops today's unfinished candle.
  2. Processes every completed day since the last run (catches up if the PC was off).
  3. Updates three fake $1,000 portfolios:
       - MA strategy:  hold BTC when MA5 > MA100, otherwise cash earning T-bill interest
       - Buy & hold:   100% BTC, never trades
       - Fixed mix:    always 59% BTC / 41% cash (the strategy's average exposure in the backtest)
  4. Appends one row per day to paper_log.csv and prints a summary.

Files (created next to this script):
  paper_log.csv   one row per completed day - the record to commit to GitHub
  paper_run.log   what each run printed - useful when it runs unattended

Run:              python paper_trade.py
Try it safely:    python paper_trade.py --demo    (fake prices, writes paper_log_demo.csv)
Testing only:     python paper_trade.py --demo --asof 2026-09-01   (pretend today is that date)
"""
import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ---- settings (fixed BEFORE paper trading starts - don't tune them mid-test) ----
TICKER = "BTC-USD"
FAST = 5                     # winner picked on the backtest's train period
SLOW = 100
FEE = 0.001                  # 0.1% per buy or sell
START_CASH = 1000.0
FIXED_MIX_WEIGHT = 0.59      # share of time the strategy held BTC on the backtest's test period
RATE_TICKER = "^IRX"         # 13-week US T-bill yield, annual %
FALLBACK_CASH_RATE = 0.0     # annual %, used only if the T-bill download fails

HERE = Path(__file__).resolve().parent
COLUMNS = ["date", "close", "ma_fast", "ma_slow", "signal", "action", "position_after",
           "btc_day_return_pct", "cash_rate_pct", "strategy_value", "buyhold_value",
           "fixedmix_value", "trades_total", "logged_at_utc"]

_output = []


def out(msg=""):
    """Print and remember, so the run can also be saved to the run log."""
    print(msg)
    _output.append(msg)


def fail(msg, run_log):
    out(f"ERROR: {msg}")
    save_run_log(run_log)
    sys.exit(1)


def save_run_log(run_log):
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    with open(run_log, "a", encoding="utf-8") as f:
        f.write(f"\n===== run at {stamp} =====\n" + "\n".join(_output) + "\n")


# ---------------------------------------------------------------- data
def download_close(ticker, period):
    import yfinance as yf
    data = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)
    if data.empty:
        return None
    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna()
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    return close


def load_prices(demo, today):
    """Completed daily closes only (anything dated today or later is dropped)."""
    if demo:
        dates = pd.date_range("2024-01-01", "2026-12-31")
        rng = np.random.default_rng(7)
        close = pd.Series(60000 * np.cumprod(1 + rng.normal(0.0005, 0.03, len(dates))), index=dates)
    else:
        close = download_close(TICKER, "2y")
        if close is None:
            return None
    return close[close.index < today]


def load_rates(demo):
    if demo:
        return pd.Series(3.0, index=pd.date_range("2023-01-01", "2026-12-31"))
    try:
        return download_close(RATE_TICKER, "1y")
    except Exception:
        return None


def rate_on(rates, date):
    """Latest known T-bill rate (annual %) on or before date."""
    if rates is None:
        return FALLBACK_CASH_RATE
    known = rates[rates.index <= date]
    return max(float(known.iloc[-1]), 0.0) if len(known) else FALLBACK_CASH_RATE


# ---------------------------------------------------------------- report
def worst_drop(values):
    v = np.r_[START_CASH, values.astype(float).values]
    return (v / np.maximum.accumulate(v) - 1).min() * 100


def summary(log, stale_note):
    last = log.iloc[-1]
    start = log.iloc[0]["date"]
    days = (pd.Timestamp(last["date"]) - pd.Timestamp(start)).days
    out(f"\n=== Paper trading: {TICKER} MA{FAST} / MA{SLOW}  (FAKE money) ===")
    out(f"Latest completed day: {last['date']}   BTC close ${last['close']:,.2f}")
    out(f"MA{FAST} ${last['ma_fast']:,.2f}  vs  MA{SLOW} ${last['ma_slow']:,.2f}"
        f"  ->  signal: {'IN BTC' if last['signal'] == 1 else 'CASH'}")
    out(f"Action at that close: {last['action']}")
    out(f"Running since {start} ({days} days), trades so far: {int(last['trades_total'])}")
    if stale_note:
        out(stale_note)

    out(f"\n  {'':<16}{'value':>12}{'since start':>13}{'worst drop':>12}")
    for name, col in [("MA strategy", "strategy_value"),
                      ("Buy & hold", "buyhold_value"),
                      (f"Fixed {FIXED_MIX_WEIGHT:.0%} BTC", "fixedmix_value")]:
        val = float(last[col])
        out(f"  {name:<16}{'$' + format(val, ',.2f'):>12}{(val / START_CASH - 1) * 100:+12.2f}%"
            f"{worst_drop(log[col]):11.2f}%")

    out("\nReminder: weeks or months of paper trading prove very little. Judge the strategy")
    out("against buy & hold AND the fixed mix, the same way the backtest did.")
    out("Not financial advice.")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Daily paper trading for the BTC MA strategy.")
    ap.add_argument("--demo", action="store_true", help="fake prices, separate demo log")
    ap.add_argument("--asof", help="testing only: pretend today is YYYY-MM-DD")
    args = ap.parse_args()

    log_path = HERE / ("paper_log_demo.csv" if args.demo else "paper_log.csv")
    run_log = HERE / ("paper_run_demo.log" if args.demo else "paper_run.log")
    if args.demo:
        out("*** DEMO MODE: fake prices. Writes paper_log_demo.csv, not the real log. ***")

    today = (pd.Timestamp(args.asof) if args.asof
             else pd.Timestamp(datetime.now(timezone.utc).date()))

    close = load_prices(args.demo, today)
    if close is None:
        fail("No BTC data downloaded. Check your internet connection and try again.", run_log)
    if len(close) < SLOW + 1:
        fail(f"Need at least {SLOW + 1} days of prices, got {len(close)}.", run_log)
    rates = load_rates(args.demo)
    if rates is None:
        out(f"WARNING: T-bill rates not downloaded. Cash assumed to earn {FALLBACK_CASH_RATE}%/yr today.")

    ma_f = close.rolling(FAST).mean()
    ma_s = close.rolling(SLOW).mean()
    latest = close.index[-1]
    stale_note = ""
    if latest < today - pd.Timedelta(days=1):
        stale_note = (f"NOTE: newest completed day in the data is {latest.date()}, not "
                      f"{(today - pd.Timedelta(days=1)).date()}. Yahoo may be late; run again later.")

    log = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame(columns=COLUMNS)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    new_rows = []

    if log.empty:
        # ---- first run: open the three fake portfolios at the latest completed close ----
        d = latest
        sig = int(ma_f[d] > ma_s[d])
        strat = START_CASH * (1 - FEE) if sig else START_CASH
        bh = START_CASH * (1 - FEE)                       # buy & hold pays the buy fee too
        mix = START_CASH * (1 - FEE * FIXED_MIX_WEIGHT)   # fee only on the BTC part
        new_rows.append([d.date(), close[d], ma_f[d], ma_s[d], sig,
                         "START: BUY" if sig else "START: STAY IN CASH", sig,
                         np.nan, rate_on(rates, d), strat, bh, mix, sig, now])
        out(f"Started paper trading at the {d.date()} close with ${START_CASH:,.0f} per portfolio.")
    else:
        last_date = pd.Timestamp(log.iloc[-1]["date"])
        new_days = close.index[close.index > last_date]
        if len(new_days) == 0:
            out("Nothing new to log: no completed day since the last run.")
        else:
            if len(new_days) > 1:
                out(f"Catching up {len(new_days)} days. Trades on missed days assume they happened "
                    f"at those closes; in real life you'd have missed them.")
            logged_prev = float(log.iloc[-1]["close"])
            if last_date in close.index:
                fresh_prev = float(close[last_date])
                if abs(fresh_prev / logged_prev - 1) > 0.005:
                    out(f"WARNING: Yahoo's close for {last_date.date()} changed from "
                        f"${logged_prev:,.2f} (logged) to ${fresh_prev:,.2f}. Using the new value.")
                prev_close = fresh_prev
            else:
                prev_close = logged_prev

            pos = int(log.iloc[-1]["position_after"])
            strat, bh, mix = (float(log.iloc[-1][c]) for c in
                              ("strategy_value", "buyhold_value", "fixedmix_value"))
            trades = int(log.iloc[-1]["trades_total"])
            prev_date = last_date
            w = FIXED_MIX_WEIGHT

            for d in new_days:
                gap = (d - prev_date).days
                if gap > 1:
                    out(f"WARNING: {gap - 1} day(s) missing in Yahoo's data before {d.date()}. "
                        f"Their price move is folded into this day.")
                r = close[d] / prev_close - 1
                rate = rate_on(rates, prev_date)
                c = rate / 100 * gap / 365

                strat *= 1 + pos * r + (1 - pos) * c          # held position earns the day's move
                bh *= 1 + r
                port = w * r + (1 - w) * c
                mix *= 1 + port - w * (1 - w) * abs(r - c) / (1 + port) * FEE   # daily rebalance

                sig = int(ma_f[d] > ma_s[d])                  # decide at this close
                if sig != pos:
                    strat *= 1 - FEE
                    trades += 1
                    action = "BUY" if sig else "SELL"
                else:
                    action = "HOLD BTC" if sig else "STAY IN CASH"
                pos = sig

                new_rows.append([d.date(), close[d], ma_f[d], ma_s[d], sig, action, pos,
                                 r * 100, rate, strat, bh, mix, trades, now])
                prev_close, prev_date = close[d], d

    if new_rows:
        add = pd.DataFrame(new_rows, columns=COLUMNS)
        log = add if log.empty else pd.concat([log, add], ignore_index=True)
        tmp = log_path.with_suffix(".tmp")
        log.to_csv(tmp, index=False, float_format="%.6f")
        try:
            os.replace(tmp, log_path)                         # all-or-nothing: no half-written log
        except PermissionError:
            tmp.unlink(missing_ok=True)
            fail(f"Couldn't save {log_path.name}. Is it open in Excel? Close it and run again.", run_log)
        out(f"Logged {len(new_rows)} new day(s) to {log_path.name}.")

    summary(log, stale_note)
    save_run_log(run_log)


if __name__ == "__main__":
    main()
