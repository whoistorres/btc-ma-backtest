# Does a Moving-Average Trading Strategy Actually Work on Bitcoin?

A backtesting project that tests a classic moving-average crossover strategy on
Bitcoin, using train/test splits, realistic cash interest, a same-exposure benchmark,
and out-of-sample checks on Ethereum and the S&P 500.

**Short answer: no reliable edge.** The strategy did not beat buy & hold on return,
the "best settings" found in the past did not predict the future, and on stocks the
signals added nothing beyond simply holding less. There is a weak hint that it helps
in crypto by avoiding crashes, but the evidence is not strong enough to trade on.

> This is an educational project, not financial advice. No real money was traded.

---

## The strategy

- Compute a **fast** and a **slow** moving average of the daily closing price.
- **Fast above slow** → hold the asset. **Fast below slow** → hold cash.
- Trades act on the *previous* day's signal (no look-ahead).
- **0.1% fee** on every buy and sell.
- While in cash, the money earns the **real 13-week US T-bill rate** of that time
  (Yahoo Finance `^IRX`).

## How it's tested

Most trading-bot tutorials stop at "my backtest made money." This project tries to
answer harder questions:

| Question | How it's tested |
|---|---|
| Are the chosen settings just lucky? | 34 fast/slow combinations. The winner is picked on the **first 70%** of BTC history (train) and judged on the **last 30%** it never saw (test). |
| Did train performance predict test performance? | Rank correlation of every combo's train Sharpe vs test Sharpe. |
| Is the edge real or did the market just change? | Compare the strategy's Sharpe **edge over buy & hold** in train vs test, not raw Sharpe. |
| Is the strategy smart, or just holding less? | Compare against a **fixed mix** (e.g. always 59% BTC / 41% cash, no signals) with the **same average exposure**. |
| Do the settings work elsewhere? | Run the BTC winner, unchanged, on **ETH** and **SPY** (S&P 500 ETF). |

Sharpe ratio here = return **above cash**, per unit of risk, annualized
(365 days for crypto, 252 for stocks).

---

## Results (run on 2026-10-18)

### Bitcoin: train vs test

- Data: 2014-09-17 to 2026-10-18
- Train: 2015-04-05 to 2023-05-01 (picks the winner)
- Test: 2023-05-02 to 2026-10-18 (never seen)
- Winner on train: **MA5 / MA100**

| | Train return | Train Sharpe | Test return | Test Sharpe | Test worst drop |
|---|---|---|---|---|---|
| MA5 / MA100 | 34,413% | 1.54 | 158.7% | 0.86 | -27.5% |
| Buy & hold | 10,973% | 1.16 | 190.6% | 0.80 | -53.1% |

- Sharpe edge over buy & hold fell from **+0.38 (train)** to **+0.05 (test)**: most of the edge disappeared.
- Train-vs-test rank correlation across all 34 combos: **0.09** (train results barely predicted test results).
- Only 1 of 34 combos beat buy & hold on test return.

### Is the timing real, or just less exposure?

The strategy was in BTC 59% of the time during the test period, so it's compared
against always holding 59% BTC and 41% cash.

| BTC test period | Return | Sharpe | Worst drop |
|---|---|---|---|
| MA strategy | 158.7% | 0.86 | -27.5% |
| Buy & hold | 190.6% | 0.80 | -53.1% |
| Fixed 59% BTC | 117.3% | 0.80 | -34.1% |

The strategy edged out the fixed mix on all three measures, but the Sharpe gap (+0.06)
is far inside the noise. A Sharpe ratio measured over 3.5 years has an uncertainty of
roughly ±0.5 (standard error ≈ 1/√years).

### Same settings on assets they were never tuned on

| Asset | Period | | Return | Sharpe | Worst drop |
|---|---|---|---|---|---|
| ETH | 2018–2026 (8.4 yrs) | MA strategy | 2,481% | 0.91 | -66.6% |
| | | Buy & hold | 321% | 0.59 | -86.4% |
| | | Fixed 53% ETH | 371% | 0.58 | -63.0% |
| SPY | 1993–2026 (32.9 yrs) | MA strategy | 1,092% | 0.49 | -39.4% |
| | | Buy & hold | 2,853% | 0.51 | -55.2% |
| | | Fixed 74% SPY | 1,561% | 0.51 | -43.3% |

- **ETH:** strong result, but 33 of 34 combos beat buy & hold, so it reflects trend-following
  on ETH in general, not these specific settings. ETH also moves with BTC and its history
  overlaps BTC's training years, so it's weak evidence.
- **SPY:** the real independent check. No edge over a fixed mix. BTC's winning settings
  ranked 16th of 34, exactly in the middle.

> Results change slightly every day as new price data arrives.

---

## What I learned

1. **The strategy doesn't beat buy & hold on return.** Lower risk came with lower return.
2. **Optimizing settings on past data doesn't work here.** The train winner was not the
   test winner, and train ranking barely predicted test ranking.
3. **"Safer" can just mean "holding less."** On stocks, a fixed mix with the same exposure
   did as well as the signals.
4. **In crypto, there's a weak hint that timing helps by dodging crashes**, but the clean
   test (unseen BTC data) shows only a tiny, statistically meaningless edge.
5. **Benchmarks matter more than returns.** Every version of this strategy "made money."
   That alone meant nothing until it was compared against the right alternatives.

## Limitations

- One strategy family, one train/test split, and three related assets.
- Not modeled: taxes, slippage beyond the 0.1% fee, exchange risk, and the fact that early
  Bitcoin (2015–2017) was thin and illiquid, so train-period returns are unrealistic.
- The fixed-mix benchmark rebalances daily; it pays the fee on rebalancing but is still idealized.
- The verdict thresholds (e.g. 0.1 Sharpe) are rules of thumb, not statistical tests.

---

## How to run it

Requires Python 3.10+ (tested on 3.13).

```bash
pip install -r requirements.txt
python backtest.py
```

On Windows with multiple Python versions installed, use `py -3.13` instead of `python`.

To check the code works without internet, run it on **fake random prices**
(results are meaningless):

```bash
python backtest.py --demo
```

Settings such as the fee, the moving-average options, the train/test split and the
assets tested are at the top of `backtest.py`.

## Project structure

```
backtest.py        # the full backtest: data, strategy, train/test, benchmarks, report
requirements.txt   # Python packages
README.md          # this file
LICENSE            # MIT
```

## Data

Daily prices and T-bill rates come from Yahoo Finance via the
[`yfinance`](https://github.com/ranaroussi/yfinance) package. Yahoo data can have gaps
or occasional errors.

## License

MIT. See [LICENSE](LICENSE).
