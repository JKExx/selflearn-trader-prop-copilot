import argparse
from app.dataio.yf import get_ohlcv
from app.backtest import run_backtest
import matplotlib.pyplot as plt

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--symbol', type=str, default='EURUSD=X')
    p.add_argument('--interval', type=str, default='1h')
    p.add_argument('--start', type=str, default='2023-01-01')
    p.add_argument('--end', type=str, default='')
    p.add_argument('--threshold', type=float, default=0.55)
    p.add_argument('--epsilon', type=float, default=0.1)
    p.add_argument('--warmup', type=int, default=200)
    p.add_argument('--cash', type=float, default=10000.0)
    p.add_argument('--kill_dd', type=float, default=0.10)
    p.add_argument('--risk', type=float, default=0.005, help='Risk per trade fraction (0.005=0.5%)')
    p.add_argument('--atr_mult', type=float, default=1.0)
    p.add_argument('--spread_pips', type=float, default=0.2)
    p.add_argument('--slippage_pips', type=float, default=0.1)
    p.add_argument('--commission', type=float, default=0.0)
    p.add_argument('--gate_kz', action='store_true')
    p.add_argument('--no_persist', action='store_true')
    args = p.parse_args()

    df = get_ohlcv(args.symbol, args.interval, start=args.start, end=args.end if args.end else None)
    res = run_backtest(
        df,
        symbol=args.symbol,
        threshold=args.threshold,
        epsilon=args.epsilon,
        min_samples_before_trade=args.warmup,
        starting_cash=args.cash,
        kill_switch_dd=args.kill_dd,
        risk_per_trade=args.risk,
        atr_multiple=args.atr_mult,
        spread_pips=args.spread_pips,
        slippage_pips=args.slippage_pips,
        commission_per_trade=args.commission,
        gate_killzones=args.gate_kz,
        persist_model=not args.no_persist
    )

    print("Metrics:", res.metrics)

    fig = plt.figure()
    plt.plot(res.equity_curve)
    plt.xlabel("Trade # (approx)")
    plt.ylabel("Equity")
    plt.title("Equity Curve")
    plt.show()

if __name__ == '__main__':
    main()
