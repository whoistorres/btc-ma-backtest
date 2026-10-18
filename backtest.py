"""
BTC moving-average strategy lab (backtest only, no real money).

What it does:
  1. Downloads all available daily BTC-USD history, plus US T-bill rates.
  2. Tests many fast/slow moving-average combinations.
     When out of the market, cash earns the real T-bill rate of that time.
  3. Picks the best combo using ONLY the first 70% of BTC data (train).
  4. Checks that pick on the last 30% of BTC (test), which it never saw.
  5. Judges overfitting fairly: edge over buy & hold in train vs test.
  6. Asks the key question: is the timing real, or is the strategy only
     safer because it holds cash part of the time? It compares against a
     FIXED mix (e.g. 60% BTC / 40% cash, no signals) with the same average
     amount invested.
  7. Runs the same winning settings on ETH and SPY (never used to pick them).

Run on your PC:   python backtest.py
Check code only:  python backtest.py --demo   (FAKE random prices, results mean nothing)
"""
import sys
from itertools import product

import numpy as np
import pandas as pd

# ---- settings you can change ----
FEE = 0.001                          # 0.1% fee per buy or sell
FAST_OPTIONS = [5, 10, 15, 20, 30, 40]
SLOW_OPTIONS = [30, 50, 75, 100, 150, 200]
TRAIN_FRACTION = 0.70                # first 70% = pick the winner, last 30% = unseen test
MAIN_ASSET = ("BTC-USD", 365)        # (ticker, trading days per year)
OTHER_ASSETS = [("ETH-USD", 365),    # crypto, moves a lot like BTC
                ("SPY", 252)]        # S&P 500 ETF, a very different market
RATE_TICKER = "^IRX"                 # 13-week US T-bill yield (annual %), used for cash interest
FALLBACK_CASH_RATE = 0.0             # annual %, used only if T-bill download fails

DEMO = "--demo" in sys.argv


# ---------------------------------------------------------------- data
def download_close(ticker):
    import yfinance as yf
    data = yf.download(ticker, period="max", interval="1d", progress=False, auto_adjust=True)
    if data.empty:
        return None
    close = data["Close"]
    if isinstance(close, pd.DataFrame):          # newer yfinance returns a 1-column table
        close = close.iloc[:, 0]
    return close.dropna()


def get_prices(ticker, seed=0):
    """Daily closing prices for a ticker, or None if the download fails."""
    if DEMO:
        rng = np.random.default_rng(seed)
        dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=3000)
        rets = rng.normal(0.0005, 0.03, len(dates))
        return pd.Series(100 * np.cumprod(1 + rets), index=dates)
    return download_close(ticker)


def get_rates():
    """T-bill yield in annual %, or None."""
    if DEMO:
        dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=4000)
        return pd.Series(3.0, index=dates)
    return download_close(RATE_TICKER)


def cash_returns(index, rates):
    """Daily return of cash for each date in index, accrued by calendar days
    since the previous date, using the previous day's T-bill rate."""
    if rates is None:
        r = pd.Series(FALLBACK_CASH_RATE, index=index)
    else:
        r = rates.reindex(rates.index.union(index)).ffill().reindex(index)
        r = r.fillna(FALLBACK_CASH_RATE).clip(lower=0)
    gap_days = index.to_series().diff().dt.days.fillna(1).values
    return (r.shift(1).bfill() / 100 * gap_days / 365).rename("cash")


# ---------------------------------------------------------------- math
def metrics(ret, cash, trades=None, days_per_year=365):
    """Return %, max drawdown %, Sharpe (on returns ABOVE cash), trade count."""
    equity = (1 + ret).cumprod().values
    peak = np.maximum.accumulate(np.r_[1.0, equity])[1:]
    excess = ret - cash
    sd = excess.std()
    return {
        "ret": (equity[-1] - 1) * 100,
        "dd": (equity / peak - 1).min() * 100,
        "sharpe": excess.mean() / sd * np.sqrt(days_per_year) if sd > 0 else 0.0,
        "trades": int(trades.sum()) if trades is not None else 0,
    }


def all_combos():
    return [(f, s) for f, s in product(FAST_OPTIONS, SLOW_OPTIONS) if f < s]


def strategy_returns(close, cash):
    """For every combo: hold the asset when fast MA > slow MA, else cash (earning interest).
    Acts on yesterday's signal (no peeking at today's close)."""
    ma = {w: close.rolling(w).mean() for w in set(FAST_OPTIONS + SLOW_OPTIONS)}
    market = close.pct_change().fillna(0.0)
    out = {}
    for f, s in all_combos():
        pos = (ma[f] > ma[s]).astype(float).shift(1).fillna(0.0)
        trades = pos.diff().abs().fillna(0.0)
        ret = pos * market + (1 - pos) * cash - trades * FEE
        out[(f, s)] = (ret, trades, pos)
    return out, market, ma


def fixed_mix(market, cash, weight):
    """Always hold `weight` in the asset and the rest in cash, rebalanced daily.
    Pays the fee on the small amount traded each day to rebalance."""
    port = weight * market + (1 - weight) * cash
    turnover = weight * (1 - weight) * (market - cash).abs() / (1 + port)
    return port - turnover * FEE


# ---------------------------------------------------------------- printing
def print_compare(strat, bh, mix, exposure, asset_name):
    rows = [("MA strategy", strat),
            ("Buy & hold", bh),
            (f"Fixed {exposure:.0%} {asset_name}", mix)]
    print(f"  {'':<22}{'return':>12}{'Sharpe':>8}{'worst drop':>12}")
    for name, m in rows:
        print(f"  {name:<22}{m['ret']:11.1f}%{m['sharpe']:8.2f}{m['dd']:11.1f}%")
    print(f"  (fixed mix = always {exposure:.0%} in, {1 - exposure:.0%} in cash, no signals;"
          f"\n   same average exposure as the strategy)")


def timing_verdict(strat, mix):
    """Is the timing adding anything beyond simply holding less?"""
    d_sh = strat["sharpe"] - mix["sharpe"]
    d_dd = strat["dd"] - mix["dd"]              # positive = strategy's worst drop is smaller
    if d_sh > 0.1:
        line = f"Timing ADDED value: better risk-adjusted than the fixed mix ({d_sh:+.2f} Sharpe)."
        real = True
    elif d_sh > -0.1:
        line = f"Timing added little: about the same as the fixed mix ({d_sh:+.2f} Sharpe)."
        real = False
    else:
        line = f"Timing HURT: worse than just holding the fixed mix ({d_sh:+.2f} Sharpe)."
        real = False
    dd_line = (f"Worst drop {strat['dd']:.1f}% vs fixed mix {mix['dd']:.1f}% "
               f"({'smaller' if d_dd > 0 else 'bigger'} by {abs(d_dd):.1f} points).")
    return real, d_dd > 5, [line, dd_line]


def bh_verdict(strat, bh):
    edge = strat["sharpe"] - bh["sharpe"]
    if edge > 0.1:
        return f"Beat buy & hold risk-adjusted ({edge:+.2f} Sharpe)."
    if edge > -0.1:
        return f"Roughly tied buy & hold risk-adjusted ({edge:+.2f} Sharpe)."
    return f"WORSE than buy & hold risk-adjusted ({edge:+.2f} Sharpe)."


# ---------------------------------------------------------------- main
def main():
    if DEMO:
        print("*** DEMO MODE: FAKE random prices. Results are meaningless. ***")

    rates = get_rates()
    if rates is None:
        print(f"WARNING: T-bill rates not downloaded. Cash assumed to earn {FALLBACK_CASH_RATE}%/yr.")

    # ================= PART 1: BTC train / test =================
    ticker, dpy = MAIN_ASSET
    close = get_prices(ticker, seed=42)
    if close is None:
        sys.exit("No BTC data downloaded. Check your internet connection.")
    cash = cash_returns(close.index, rates)
    strats, market, ma = strategy_returns(close, cash)

    idx = close.index[max(SLOW_OPTIONS):]        # same dates for every combo
    if len(idx) < 500:
        sys.exit("Not enough price history for a train/test split.")
    split = int(len(idx) * TRAIN_FRACTION)
    train_idx, test_idx = idx[:split], idx[split:]

    bh_train = metrics(market.loc[train_idx], cash.loc[train_idx], days_per_year=dpy)
    bh_test = metrics(market.loc[test_idx], cash.loc[test_idx], days_per_year=dpy)

    rows = []
    for (f, s), (ret, trades, pos) in strats.items():
        tr = metrics(ret.loc[train_idx], cash.loc[train_idx], trades.loc[train_idx], dpy)
        te = metrics(ret.loc[test_idx], cash.loc[test_idx], trades.loc[test_idx], dpy)
        rows.append({"fast": f, "slow": s,
                     "train_ret": tr["ret"], "train_Sh": tr["sharpe"],
                     "test_ret": te["ret"], "test_Sh": te["sharpe"],
                     "trades": te["trades"], "test_dd": te["dd"]})
    res = pd.DataFrame(rows).sort_values("train_Sh", ascending=False).reset_index(drop=True)

    avg_rate = cash.loc[idx].sum() / (len(idx) / dpy) * 100
    print(f"\n===== PART 1: {ticker} train / test =====")
    print(f"Data:  {close.index[0].date()} to {close.index[-1].date()}  ({len(close)} days)")
    print(f"TRAIN: {train_idx[0].date()} to {train_idx[-1].date()}  ({len(train_idx)} days, picks winner)")
    print(f"TEST:  {test_idx[0].date()} to {test_idx[-1].date()}  ({len(test_idx)} days, never seen)")
    print(f"Fee {FEE:.1%}/trade. Cash earns T-bill rate (avg ~{avg_rate:.1f}%/yr here). Combos: {len(res)}")
    print("Sharpe = return above cash, per unit of risk.")

    print("\nBuy & hold:")
    print(f"  train: {bh_train['ret']:9.1f}%   Sharpe {bh_train['sharpe']:5.2f}   worst drop {bh_train['dd']:6.1f}%")
    print(f"  test:  {bh_test['ret']:9.1f}%   Sharpe {bh_test['sharpe']:5.2f}   worst drop {bh_test['dd']:6.1f}%")

    print("\nTop 5 combos by TRAIN Sharpe, and how they did on TEST:")
    show = res.head(5)[["fast", "slow", "train_ret", "train_Sh", "test_ret", "test_Sh", "trades"]]
    print(show.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    w = res.iloc[0]
    wf, ws = int(w.fast), int(w.slow)
    print(f"\nChosen winner (best on train): MA{wf} / MA{ws}")
    print(f"  train: {w.train_ret:9.1f}%   Sharpe {w.train_Sh:5.2f}")
    print(f"  test:  {w.test_ret:9.1f}%   Sharpe {w.test_Sh:5.2f}   worst drop {w.test_dd:6.1f}%")

    rank_corr = res["train_Sh"].rank().corr(res["test_Sh"].rank())
    print(f"\nAll {len(res)} combos on TEST: beat buy & hold on return: "
          f"{int((res['test_ret'] > bh_test['ret']).sum())} | on Sharpe: "
          f"{int((res['test_Sh'] > bh_test['sharpe']).sum())}")
    print(f"Train-vs-test rank correlation: {rank_corr:.2f}  (near 0 = train doesn't predict test)")

    edge_train = w.train_Sh - bh_train["sharpe"]
    edge_test = w.test_Sh - bh_test["sharpe"]
    print(f"\nSharpe edge over buy & hold:  train {edge_train:+.2f}   test {edge_test:+.2f}")
    if edge_train - edge_test > 0.25:
        print(f"  Most of the train edge disappeared on test: overfitting likely.")
    else:
        print(f"  Edge held about steady from train to test: no strong sign of overfitting.")

    # ---- is the timing real, or just less exposure? (BTC test period) ----
    win_ret, win_trades, win_pos = strats[(wf, ws)]
    exposure = win_pos.loc[test_idx].mean()
    strat_test = metrics(win_ret.loc[test_idx], cash.loc[test_idx], win_trades.loc[test_idx], dpy)
    mix_test = metrics(fixed_mix(market, cash, exposure).loc[test_idx], cash.loc[test_idx], days_per_year=dpy)

    print(f"\n--- BTC TEST: is the timing real, or just less exposure? ---")
    print(f"Strategy was in BTC {exposure:.0%} of the time on test.")
    print_compare(strat_test, bh_test, mix_test, exposure, "BTC")
    results_summary = []
    real, dd_win, lines = timing_verdict(strat_test, mix_test)
    results_summary.append(("BTC test", real, dd_win))
    print("Verdict:")
    print("  " + bh_verdict(strat_test, bh_test))
    for ln in lines:
        print("  " + ln)

    # ================= PART 2: same settings on other assets =================
    print(f"\n===== PART 2: MA{wf} / MA{ws} on assets it was NEVER tuned on =====")
    for i, (tkr, days) in enumerate(OTHER_ASSETS, start=1):
        c = get_prices(tkr, seed=42 + i)
        if c is None or len(c) < max(SLOW_OPTIONS) + 250:
            print(f"\n--- {tkr}: skipped (no data or too little history) ---")
            continue
        csh = cash_returns(c.index, rates)
        s_all, mkt, _ = strategy_returns(c, csh)
        ix = c.index[max(SLOW_OPTIONS):]
        bh = metrics(mkt.loc[ix], csh.loc[ix], days_per_year=days)
        allm = {k: metrics(r.loc[ix], csh.loc[ix], t.loc[ix], days) for k, (r, t, p) in s_all.items()}
        win = allm[(wf, ws)]
        exp_i = s_all[(wf, ws)][2].loc[ix].mean()
        mix = metrics(fixed_mix(mkt, csh, exp_i).loc[ix], csh.loc[ix], days_per_year=days)
        rank = 1 + sum(m["sharpe"] > win["sharpe"] for m in allm.values())
        n_beat = sum(m["sharpe"] > bh["sharpe"] for m in allm.values())
        short = tkr.split("-")[0]

        print(f"\n--- {tkr}  ({ix[0].date()} to {ix[-1].date()}, {len(ix) / days:.1f} years) ---")
        print(f"Strategy was in {short} {exp_i:.0%} of the time.")
        print_compare(win, bh, mix, exp_i, short)
        print(f"  BTC winner's Sharpe rank here: {rank}/{len(allm)}")
        print(f"  Combos beating buy & hold Sharpe here: {n_beat}/{len(allm)}")
        real, dd_win, lines = timing_verdict(win, mix)
        results_summary.append((tkr, real, dd_win))
        print("Verdict:")
        print("  " + bh_verdict(win, bh))
        for ln in lines:
            print("  " + ln)

    # ================= SUMMARY =================
    print("\n===== SUMMARY =====")
    print("Strategy vs a fixed mix with the same average exposure:")
    print(f"  {'test':<10}{'better Sharpe?':>16}{'smaller worst drop?':>22}")
    for name, r, d in results_summary:
        print(f"  {name:<10}{'YES' if r else 'no':>16}{'YES' if d else 'no':>22}")
    n_sh = sum(r for _, r, _ in results_summary)
    n_dd = sum(d for _, _, d in results_summary)
    n = len(results_summary)
    if n_sh == n:
        print("The signals beat a fixed mix every time. Encouraging, but one strategy")
        print("family, related tests, idealized costs. Not proof.")
    elif n_sh == 0 and n_dd == 0:
        print("The signals added nothing a fixed mix couldn't. The 'safer' feel came")
        print("from holding less, not from smart timing.")
    elif n_sh == 0:
        print("No risk-adjusted edge over a fixed mix. Some smaller worst drops,")
        print("but not a consistent reason to use signals.")
    else:
        print("Mixed: timing helped in some tests, not others. Not a reliable edge.")
    print("Not modeled: taxes, slippage beyond the fee, exchange risk.")
    print("ETH moves with BTC and overlaps BTC's train years: weaker evidence.")
    print("SPY is the real independent check.")
    print("Don't keep re-tuning against these results, or they stop being unseen.")

    last_signal = ma[wf].iloc[-1] > ma[ws].iloc[-1]
    print(f"\nBTC winner's current signal: {'IN BTC (fast above slow)' if last_signal else 'CASH (fast below slow)'}")
    print("This is a backtest tool, not financial advice.")


if __name__ == "__main__":
    main()
