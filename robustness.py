"""
Robustness checks for the MA backtest (backtest only, no real money).

backtest.py gives ONE train/test split and point estimates. This script asks
how much those numbers can be trusted:

  A. BOOTSTRAP: how uncertain is the test-period Sharpe difference between the
     strategy and its benchmarks? Resamples the test days in ~20-day blocks
     (keeps BTC's calm/wild stretches together) thousands of times and reports
     a 95% interval. If the interval includes 0, the gap is not distinguishable
     from luck.

  B. WALK-FORWARD: instead of one 70/30 split, step through history. Each year:
     pick the best of the 34 combos using ONLY earlier data, then test it on the
     next year. Repeat. Also measures selection bias: does the train winner
     actually rank high among the 34 combos in the year that follows?

  C. FEE SWEEP: reruns A's split and B at fees of 0.05%, 0.1%, 0.25%, 0.5%.
     Both the original winner (kept fixed) and a winner re-picked at each fee.
     The fixed mix pays the same fee.

Strategy rules, combos, data, and benchmarks all come from backtest.py,
unchanged. backtest.py itself is not modified.

Run on your PC:   python robustness.py            (saves to results/)
Check code only:  python robustness.py --demo     (FAKE prices, saves to results_demo/)
"""
import argparse
import os
from contextlib import contextmanager

import numpy as np
import pandas as pd

import backtest as bt

# ---- settings ----
FEES = [0.0005, 0.001, 0.0025, 0.005]   # 0.05%, 0.1% (original), 0.25%, 0.5%
BOOT_N = 5000                            # bootstrap resamples
BLOCK = 20                               # bootstrap block length in days
BLOCK_CHECK = [5, 20, 60]                # also try these, to show the answer isn't block-size luck
MIN_TRAIN_YEARS = 3                      # walk-forward: first pick uses at least 3 years
SEED = 0                                 # fixed so results are reproducible

LINES = []


def log(text=""):
    print(text)
    LINES.append(text)


@contextmanager
def fee_level(fee):
    """Temporarily set backtest.py's fee (strategy_returns and fixed_mix read it)."""
    old = bt.FEE
    bt.FEE = fee
    try:
        yield
    finally:
        bt.FEE = old


# ---------------------------------------------------------------- core pieces
def pick_winner(strats, cash, dates, dpy):
    """Combo with the best Sharpe on `dates` (same rule as backtest.py)."""
    sh = {k: bt.metrics(r.loc[dates], cash.loc[dates], days_per_year=dpy)["sharpe"]
          for k, (r, t, p) in strats.items()}
    return max(sh, key=sh.get)


def evaluate(strats, market, cash, dates, combo):
    """Daily returns on `dates` for the combo, buy & hold, and a fixed mix with
    the combo's average exposure on those dates (same control as backtest.py)."""
    ret, trades, pos = strats[combo]
    exposure = pos.loc[dates].mean()
    mix = bt.fixed_mix(market, cash, exposure)
    return pd.DataFrame({"strategy": ret.loc[dates], "buyhold": market.loc[dates],
                         "mix": mix.loc[dates], "cash": cash.loc[dates],
                         "trades": trades.loc[dates]}), exposure


def scores(df, dpy):
    out = {}
    for col in ["strategy", "buyhold", "mix"]:
        trades = df["trades"] if col == "strategy" else None
        out[col] = bt.metrics(df[col], df["cash"], trades, dpy)
    return out


def block_bootstrap_sharpes(df, dpy, n_boot=BOOT_N, block=BLOCK, seed=SEED):
    """Circular block bootstrap. Every resample uses the SAME days for strategy,
    buy & hold and mix (paired), so the Sharpe differences are fair.
    Returns array (n_boot, 3): Sharpe of strategy, buy & hold, mix."""
    X = np.column_stack([(df[c] - df["cash"]).values for c in ["strategy", "buyhold", "mix"]])
    n = len(X)
    rng = np.random.default_rng(seed)
    n_blocks = -(-n // block)
    offsets = np.arange(block)
    out = np.empty((n_boot, 3))
    for i in range(0, n_boot, 500):
        b = min(500, n_boot - i)
        starts = rng.integers(0, n, size=(b, n_blocks))
        idx = ((starts[:, :, None] + offsets) % n).reshape(b, -1)[:, :n]
        S = X[idx]                                   # (b, n, 3)
        mean = S.mean(axis=1)
        sd = S.std(axis=1, ddof=1)
        out[i:i + b] = np.where(sd > 0, mean / np.where(sd > 0, sd, 1) * np.sqrt(dpy), 0.0)
    return out


def diff_summary(diffs, observed):
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"observed": observed, "ci_low": lo, "ci_high": hi,
            "share_ahead": (diffs > 0).mean()}


def ci_words(s):
    if s["ci_low"] > 0:
        return "interval is above 0: strategy ahead beyond the noise in this period."
    if s["ci_high"] < 0:
        return "interval is below 0: strategy behind beyond the noise in this period."
    return "interval includes 0: can't tell apart from luck."


def walk_forward(strats, market, cash, close, dpy):
    """Expanding window: train on everything before each test year, test on that year."""
    idx = close.index[max(bt.SLOW_OPTIONS):]
    start = int(MIN_TRAIN_YEARS * dpy)
    folds, parts = [], []
    while len(idx) - start >= dpy // 4:              # keep a final partial year if >= 3 months
        end = min(start + dpy, len(idx))
        tr, te = idx[:start], idx[start:end]
        combo = pick_winner(strats, cash, tr, dpy)
        df, exposure = evaluate(strats, market, cash, te, combo)
        sc = scores(df, dpy)
        test_sh = {k: bt.metrics(r.loc[te], cash.loc[te], days_per_year=dpy)["sharpe"]
                   for k, (r, t, p) in strats.items()}
        mine = test_sh[combo]
        others = [v for k, v in test_sh.items() if k != combo]
        pct = (sum(v < mine for v in others) + 0.5 * sum(v == mine for v in others)) / len(others)
        folds.append({
            "test_start": te[0].date(), "test_end": te[-1].date(), "days": len(te),
            "partial": len(te) < dpy, "fast": combo[0], "slow": combo[1],
            "exposure": exposure, "trades": sc["strategy"]["trades"],
            "strat_ret": sc["strategy"]["ret"], "strat_sharpe": sc["strategy"]["sharpe"],
            "bh_ret": sc["buyhold"]["ret"], "bh_sharpe": sc["buyhold"]["sharpe"],
            "mix_ret": sc["mix"]["ret"], "mix_sharpe": sc["mix"]["sharpe"],
            "winner_percentile_among_combos": pct,
        })
        parts.append(df)
        start = end
    return pd.DataFrame(folds), pd.concat(parts)


# ---------------------------------------------------------------- charts
COLORS = {"strategy": "#2a78d6", "buyhold": "#eb6834", "mix": "#1baf7a",
          "surface": "#fcfcfb", "text": "#0b0b0b", "text2": "#52514e",
          "grid": "#e6e5e1", "zero": "#8a8984"}


def setup_axes(ax):
    ax.set_facecolor(COLORS["surface"])
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(COLORS["zero"])
    ax.tick_params(colors=COLORS["text2"], labelsize=9, length=0)


def new_fig(w, h, ncols=1, nrows=1, **kw):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "text.color": COLORS["text"],
                         "axes.labelcolor": COLORS["text2"], "axes.titlecolor": COLORS["text"]})
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h), facecolor=COLORS["surface"], **kw)
    for ax in np.atleast_1d(axes).ravel():
        setup_axes(ax)
    return fig, axes


def chart_bootstrap(panels, path, demo):
    """panels: list of (title, diffs array, summary, color)."""
    fig, axes = new_fig(10, 6.4, ncols=2, nrows=2, sharex=True)
    for ax, (title, diffs, s, color) in zip(axes.ravel(), panels):
        ax.hist(diffs, bins=60, color=color, alpha=0.85, edgecolor=COLORS["surface"], linewidth=0.6)
        ax.axvline(0, color=COLORS["text2"], linewidth=1.2, linestyle=(0, (4, 3)))
        for v in (s["ci_low"], s["ci_high"]):
            ax.axvline(v, color=COLORS["text"], linewidth=0.9)
        ax.set_title(title, fontsize=10, loc="left", pad=22)
        ax.text(0, 1.02, f"observed {s['observed']:+.2f}   95% interval [{s['ci_low']:+.2f}, "
                f"{s['ci_high']:+.2f}]   strategy ahead in {s['share_ahead']:.0%}",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=8.5, color=COLORS["text2"])
        ax.set_yticks([])
    for ax in axes[1]:
        ax.set_xlabel("Sharpe difference (strategy minus benchmark)")
    fig.suptitle(("DEMO (fake prices): " if demo else "")
                 + "How sure are the Sharpe gaps? Block bootstrap (solid lines = 95% interval, dashed = 0)",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=COLORS["surface"])


def chart_walk_forward(wf, path, demo):
    fig, ax = new_fig(10, 4.8)
    x = np.arange(len(wf))
    w = 0.27
    series = [("strat_sharpe", "MA strategy (re-picked yearly)", COLORS["strategy"], -w),
              ("bh_sharpe", "Buy & hold", COLORS["buyhold"], 0),
              ("mix_sharpe", "Fixed mix, same exposure", COLORS["mix"], w)]
    for col, label, color, off in series:
        ax.bar(x + off, wf[col], width=w, color=color, label=label,
               edgecolor=COLORS["surface"], linewidth=1.5)
    for i, r in wf.iterrows():
        y = r["strat_sharpe"]
        ax.text(i - w, y + (0.05 if y >= 0 else -0.05), f"{r.fast}/{r.slow}", ha="center",
                va="bottom" if y >= 0 else "top", fontsize=7.5, color=COLORS["text2"])
    ax.axhline(0, color=COLORS["zero"], linewidth=1)
    labels = [f"{pd.Timestamp(d):%b %Y}" + ("*" if p else "") for d, p in zip(wf.test_start, wf.partial)]
    ax.set_xticks(x, labels)
    ax.set_ylabel("Sharpe in that test year")
    ax.legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0, 1.13), fontsize=9)
    ax.set_title(("DEMO (fake prices): " if demo else "")
                 + "Walk-forward: winner picked on past data only, tested on the next year",
                 loc="left", fontsize=12, pad=30)
    note = "Labels = chosen fast/slow MA. Test year starts on the date shown."
    if wf.partial.any():
        note += " * = partial year."
    fig.text(0.01, 0.01, note, fontsize=8, color=COLORS["text2"])
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150, facecolor=COLORS["surface"])


def chart_fees(fs, path, demo):
    fig, axes = new_fig(10, 4.4, ncols=2, sharey=True)
    xs = np.arange(len(fs))
    xl = [f"{f:.2%}" for f in fs.fee]
    panels = [
        (axes[0], "70/30 test period", [
            ("split_fixed_sharpe", "Original winner (fixed)", COLORS["strategy"], "-", "o"),
            ("split_repick_sharpe", "Re-picked at each fee", COLORS["strategy"], "--", "o"),
            ("split_mix_sharpe", "Fixed mix", COLORS["mix"], "-", "s"),
            ("split_bh_sharpe", "Buy & hold", COLORS["buyhold"], "-", "D")]),
        (axes[1], "Walk-forward, all test years joined", [
            ("wf_strat_sharpe", "MA strategy (re-picked yearly)", COLORS["strategy"], "--", "o"),
            ("wf_mix_sharpe", "Fixed mix", COLORS["mix"], "-", "s"),
            ("wf_bh_sharpe", "Buy & hold", COLORS["buyhold"], "-", "D")]),
    ]
    for ax, title, lines in panels:
        for col, label, color, ls, mk in lines:
            hollow = ls == "--"
            ax.plot(xs, fs[col], color=color, linestyle=ls, linewidth=2, marker=mk, markersize=7,
                    markerfacecolor=COLORS["surface"] if hollow else color,
                    markeredgecolor=color, markeredgewidth=2, label=label)
        if title.startswith("70/30"):
            for i, r in fs.iterrows():
                ax.annotate(f"{int(r.repick_fast)}/{int(r.repick_slow)}", (i, r.split_repick_sharpe),
                            textcoords="offset points", xytext=(0, -16), ha="center",
                            fontsize=7.5, color=COLORS["text2"])
        ax.set_xticks(xs, xl)
        ax.set_xlabel("fee per trade")
        ax.set_title(title, loc="left", fontsize=10)
        ax.legend(frameon=False, fontsize=8.5, loc="best")
    axes[0].set_ylabel("Sharpe (above cash)")
    fig.suptitle(("DEMO (fake prices): " if demo else "") + "Does the result survive higher trading costs?",
                 x=0.01, ha="left", fontsize=12)
    fig.text(0.01, 0.01, "Labels = combo re-picked on train at that fee. When it equals the original "
             "winner, the solid and dashed lines overlap.", fontsize=8, color=COLORS["text2"])
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=150, facecolor=COLORS["surface"])


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true", help="fake prices, for checking the code")
    ap.add_argument("--no-charts", action="store_true", help="skip PNG charts")
    args = ap.parse_args()
    demo = bt.DEMO
    outdir = "results_demo" if demo else "results"
    os.makedirs(outdir, exist_ok=True)
    if demo:
        log("*** DEMO MODE: FAKE random prices. Results are meaningless. ***")

    rates = bt.get_rates()
    if rates is None:
        log(f"WARNING: T-bill rates not downloaded. Cash assumed to earn {bt.FALLBACK_CASH_RATE}%/yr.")
    ticker, dpy = bt.MAIN_ASSET
    close = bt.get_prices(ticker, seed=42)          # same seed as backtest.py demo
    if close is None:
        raise SystemExit("No BTC data downloaded. Check your internet connection.")
    cash = bt.cash_returns(close.index, rates)
    idx = close.index[max(bt.SLOW_OPTIONS):]
    split = int(len(idx) * bt.TRAIN_FRACTION)
    train_idx, test_idx = idx[:split], idx[split:]
    base_fee = bt.FEE
    assert base_fee in FEES, "original fee must be in the sweep"

    log(f"\nData: {ticker} {close.index[0].date()} to {close.index[-1].date()}. "
        f"Test period (same as backtest.py): {test_idx[0].date()} to {test_idx[-1].date()}.")

    # ===== A. bootstrap on the 70/30 test period, original fee =====
    strats, market, _ = bt.strategy_returns(close, cash)
    base_combo = pick_winner(strats, cash, train_idx, dpy)
    test_df, exposure = evaluate(strats, market, cash, test_idx, base_combo)
    sc = scores(test_df, dpy)
    boot = block_bootstrap_sharpes(test_df, dpy)
    a_mix = diff_summary(boot[:, 0] - boot[:, 2], sc["strategy"]["sharpe"] - sc["mix"]["sharpe"])
    a_bh = diff_summary(boot[:, 0] - boot[:, 1], sc["strategy"]["sharpe"] - sc["buyhold"]["sharpe"])
    s_lo, s_hi = np.percentile(boot[:, 0], [2.5, 97.5])

    log(f"\n===== A. BOOTSTRAP: 70/30 test period, MA{base_combo[0]}/MA{base_combo[1]}, "
        f"fee {base_fee:.2%} =====")
    log(f"(should match backtest.py) Sharpe: strategy {sc['strategy']['sharpe']:.2f}, "
        f"buy & hold {sc['buyhold']['sharpe']:.2f}, fixed {exposure:.0%} mix {sc['mix']['sharpe']:.2f}")
    log(f"{BOOT_N} resamples, {BLOCK}-day blocks. 95% intervals:")
    log(f"  Strategy Sharpe alone:     {sc['strategy']['sharpe']:.2f}  [{s_lo:.2f}, {s_hi:.2f}]")
    for name, s in [("fixed mix", a_mix), ("buy & hold", a_bh)]:
        log(f"  Strategy minus {name + ':':<15}{s['observed']:+.2f}  [{s['ci_low']:+.2f}, {s['ci_high']:+.2f}]"
            f"  ahead in {s['share_ahead']:.0%} of resamples")
        log(f"    -> {ci_words(s)}")
    log("  Block-size check (strategy minus fixed mix):")
    block_rows = []
    for L in BLOCK_CHECK:
        b = block_bootstrap_sharpes(test_df, dpy, block=L)
        s = diff_summary(b[:, 0] - b[:, 2], a_mix["observed"])
        block_rows.append({"block": L, **s})
        log(f"    {L:>3}-day blocks: [{s['ci_low']:+.2f}, {s['ci_high']:+.2f}]")

    # ===== B + C. walk-forward and split results at every fee =====
    fee_rows, wf_base, wf_base_df = [], None, None
    for fee in FEES:
        with fee_level(fee):
            st, mk, _ = bt.strategy_returns(close, cash)
            fixed_df, _ = evaluate(st, mk, cash, test_idx, base_combo)
            repick = pick_winner(st, cash, train_idx, dpy)
            repick_df, _ = evaluate(st, mk, cash, test_idx, repick)
            sf, sr = scores(fixed_df, dpy), scores(repick_df, dpy)
            wf, wf_df = walk_forward(st, mk, cash, close, dpy)
        wsc = scores(wf_df, dpy)
        fee_rows.append({
            "fee": fee,
            "split_fixed_sharpe": sf["strategy"]["sharpe"], "split_fixed_ret": sf["strategy"]["ret"],
            "split_fixed_trades": sf["strategy"]["trades"],
            "split_mix_sharpe": sf["mix"]["sharpe"], "split_bh_sharpe": sf["buyhold"]["sharpe"],
            "repick_fast": repick[0], "repick_slow": repick[1],
            "split_repick_sharpe": sr["strategy"]["sharpe"], "split_repick_mix_sharpe": sr["mix"]["sharpe"],
            "wf_strat_sharpe": wsc["strategy"]["sharpe"], "wf_mix_sharpe": wsc["mix"]["sharpe"],
            "wf_bh_sharpe": wsc["buyhold"]["sharpe"],
            "wf_strat_ret": wsc["strategy"]["ret"], "wf_bh_ret": wsc["buyhold"]["ret"],
            "wf_mix_ret": wsc["mix"]["ret"],
        })
        if fee == base_fee:
            wf_base, wf_base_df = wf, wf_df
    fs = pd.DataFrame(fee_rows)

    # ----- B. print walk-forward at original fee -----
    wf = wf_base
    log(f"\n===== B. WALK-FORWARD at fee {base_fee:.2%} "
        f"(expanding train, min {MIN_TRAIN_YEARS} yrs, 1-yr test steps) =====")
    log(f"  {'test year':<24}{'pick':>8}{'in':>6}{'strat Sh':>10}{'B&H Sh':>8}{'mix Sh':>8}{'rank pct':>10}")
    for _, r in wf.iterrows():
        tag = " (partial)" if r.partial else ""
        log(f"  {str(r.test_start) + ' to ' + str(r.test_end):<24}{f'{r.fast}/{r.slow}':>8}"
            f"{r.exposure:6.0%}{r.strat_sharpe:10.2f}{r.bh_sharpe:8.2f}{r.mix_sharpe:8.2f}"
            f"{r.winner_percentile_among_combos:10.0%}{tag}")
    n = len(wf)
    log(f"  Test years where the pick beat buy & hold Sharpe: {int((wf.strat_sharpe > wf.bh_sharpe).sum())}/{n}, "
        f"beat fixed mix Sharpe: {int((wf.strat_sharpe > wf.mix_sharpe).sum())}/{n}")
    log(f"  Distinct winners picked: {wf[['fast', 'slow']].drop_duplicates().shape[0]} "
        f"(stable picks = more trustworthy rule)")
    log(f"  Average rank of the pick among the 34 combos in its test year: "
        f"{wf.winner_percentile_among_combos.mean():.0%} (50% = no better than a random combo)")
    wsc = scores(wf_base_df, dpy)
    bw = block_bootstrap_sharpes(wf_base_df, dpy)
    b_mix = diff_summary(bw[:, 0] - bw[:, 2], wsc["strategy"]["sharpe"] - wsc["mix"]["sharpe"])
    b_bh = diff_summary(bw[:, 0] - bw[:, 1], wsc["strategy"]["sharpe"] - wsc["buyhold"]["sharpe"])
    years = len(wf_base_df) / dpy
    log(f"  All test years joined ({years:.1f} years, every day out-of-sample):")
    log(f"    return: strategy {wsc['strategy']['ret']:.0f}%, buy & hold {wsc['buyhold']['ret']:.0f}%, "
        f"mix {wsc['mix']['ret']:.0f}%")
    log(f"    worst drop: strategy {wsc['strategy']['dd']:.1f}%, buy & hold {wsc['buyhold']['dd']:.1f}%, "
        f"mix {wsc['mix']['dd']:.1f}%")
    log(f"    Sharpe: strategy {wsc['strategy']['sharpe']:.2f}, buy & hold {wsc['buyhold']['sharpe']:.2f}, "
        f"mix {wsc['mix']['sharpe']:.2f}")
    for name, s in [("fixed mix", b_mix), ("buy & hold", b_bh)]:
        log(f"    Strategy minus {name + ':':<15}{s['observed']:+.2f}  [{s['ci_low']:+.2f}, {s['ci_high']:+.2f}]"
            f"  ahead in {s['share_ahead']:.0%}")
        log(f"      -> {ci_words(s)}")
    log("  Note: the fixed mix uses each year's actual exposure, so it is a same-exposure control,")
    log("  not a strategy you could have run in advance.")

    # ----- C. print fee sweep -----
    log(f"\n===== C. FEE SWEEP =====")
    log(f"  {'fee':>6}{'MA' + str(base_combo[0]) + '/' + str(base_combo[1]) + ' Sh':>13}{'trades':>8}"
        f"{'re-pick':>9}{'its Sh':>8}{'mix Sh':>8}{'B&H Sh':>8}{'WF strat':>10}{'WF mix':>8}")
    for _, r in fs.iterrows():
        log(f"  {r.fee:6.2%}{r.split_fixed_sharpe:13.2f}{int(r.split_fixed_trades):8d}"
            f"{f'{int(r.repick_fast)}/{int(r.repick_slow)}':>9}{r.split_repick_sharpe:8.2f}"
            f"{r.split_mix_sharpe:8.2f}{r.split_bh_sharpe:8.2f}{r.wf_strat_sharpe:10.2f}{r.wf_mix_sharpe:8.2f}")
    log("  Sh = Sharpe on the 70/30 test period. WF = walk-forward, all test years joined.")
    log("  'mix Sh' is matched to the original winner's exposure. Re-picking at each fee is")
    log("  fair: the fee is known before choosing. Buy & hold pays no fee in this model, so its line is flat.")

    log("\nReminders: one strategy family, one asset, daily data, idealized fills.")
    log("A 95% interval that includes 0 means 'no evidence', not 'proof of no edge'.")
    log("Not financial advice.")

    # ---- save ----
    pd.DataFrame([{"test": "70/30 test, vs fixed mix", **a_mix}, {"test": "70/30 test, vs buy & hold", **a_bh},
                  {"test": "walk-forward joined, vs fixed mix", **b_mix},
                  {"test": "walk-forward joined, vs buy & hold", **b_bh}]
                 + [{"test": f"70/30 vs fixed mix, {r['block']}-day blocks",
                     **{k: v for k, v in r.items() if k != "block"}} for r in block_rows]
                 ).to_csv(f"{outdir}/bootstrap_summary.csv", index=False, float_format="%.4f")
    wf.to_csv(f"{outdir}/walk_forward.csv", index=False, float_format="%.4f")
    fs.to_csv(f"{outdir}/fee_sweep.csv", index=False, float_format="%.4f")
    with open(f"{outdir}/summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(LINES) + "\n")

    if not args.no_charts:
        try:
            chart_bootstrap([
                ("70/30 test: strategy minus fixed mix", boot[:, 0] - boot[:, 2], a_mix, COLORS["mix"]),
                ("70/30 test: strategy minus buy & hold", boot[:, 0] - boot[:, 1], a_bh, COLORS["buyhold"]),
                ("Walk-forward: strategy minus fixed mix", bw[:, 0] - bw[:, 2], b_mix, COLORS["mix"]),
                ("Walk-forward: strategy minus buy & hold", bw[:, 0] - bw[:, 1], b_bh, COLORS["buyhold"]),
            ], f"{outdir}/bootstrap_sharpe_diff.png", demo)
            chart_walk_forward(wf, f"{outdir}/walk_forward.png", demo)
            chart_fees(fs, f"{outdir}/fee_sensitivity.png", demo)
        except ImportError:
            print("matplotlib not installed: skipped charts (pip install matplotlib).")
    print(f"\nSaved tables, summary.txt and charts to {outdir}/")


if __name__ == "__main__":
    main()
