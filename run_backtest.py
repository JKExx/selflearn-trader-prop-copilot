import matplotlib.pyplot as plt
from app.backtest import run_backtest
from app.dataio.yf import get_ohlcv

def main():
    df = get_ohlcv("XAUUSD=X", "1h", "2025-01-01", None)
    res = run_backtest(df, symbol="XAU_USD", interval="1h")
    print("Metrics:", res.metrics)
    plt.figure()
    plt.plot(res.equity_curve)
    plt.xlabel("Trade # (approx)")
    plt.ylabel("Equity")
    plt.show()

if __name__ == "__main__":
    main()
