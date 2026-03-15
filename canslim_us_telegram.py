import math
import os
import time
import requests
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None

warnings.filterwarnings("ignore")


# =========================
# CONFIG
# =========================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

MIN_PASS_COUNT = int(os.getenv("MIN_PASS_COUNT", "5"))

# CANSLIM-like thresholds
MIN_Q_GROWTH = float(os.getenv("MIN_Q_GROWTH", "0.20"))                 # C
MIN_3Y_CAGR = float(os.getenv("MIN_3Y_CAGR", "0.15"))                   # A
NEAR_HIGH_RATIO = float(os.getenv("NEAR_HIGH_RATIO", "0.95"))           # N
MIN_RS_PERCENTILE = float(os.getenv("MIN_RS_PERCENTILE", "80"))         # L
MIN_INST_OWNERSHIP = float(os.getenv("MIN_INST_OWNERSHIP", "0.35"))     # I
MIN_VOLUME_SURGE = float(os.getenv("MIN_VOLUME_SURGE", "1.30"))         # S
MIN_DOLLAR_VOLUME_20D = float(os.getenv("MIN_DOLLAR_VOLUME_20D", "20000000"))

BENCHMARK = os.getenv("BENCHMARK", "SPY")
MARKET_ETFS = ["SPY", "QQQ"]

REQUEST_PAUSE = float(os.getenv("REQUEST_PAUSE", "0.2"))
MAX_MESSAGE_STOCKS = int(os.getenv("MAX_MESSAGE_STOCKS", "20"))

SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


# =========================
# HELPERS
# =========================
def require_env() -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        raise ValueError("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID in environment variables.")


def send_telegram_message(token: str, chat_id: str, text: str) -> None:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }
    r = requests.post(url, json=payload, timeout=20)
    r.raise_for_status()


def get_sp500_tickers() -> List[str]:
    """
    Read current S&P 500 constituents from Wikipedia.
    Notes:
    - Wikipedia table usually contains ~503 symbols because some companies have multiple share classes.
    - Yahoo Finance tickers use '-' instead of '.' for share-class symbols, e.g. BRK.B -> BRK-B
    """
    tables = pd.read_html(SP500_WIKI_URL)
    if not tables:
        raise ValueError("Failed to read tables from S&P 500 source page.")

    df = tables[0]
    if "Symbol" not in df.columns:
        raise ValueError("S&P 500 table missing 'Symbol' column.")

    tickers = (
        df["Symbol"]
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(".", "-", regex=False)   # Yahoo format: BRK.B -> BRK-B
        .tolist()
    )

    tickers = sorted(list(set(tickers)))
    if not tickers:
        raise ValueError("No S&P 500 tickers parsed from source.")

    return tickers


def safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        val = float(x)
        if math.isnan(val):
            return None
        return val
    except Exception:
        return None


def pct_change(new: Optional[float], old: Optional[float]) -> Optional[float]:
    if new is None or old is None or old == 0:
        return None
    return (new - old) / abs(old)


def calc_cagr(values: List[float]) -> Optional[float]:
    vals = [v for v in values if v is not None and pd.notna(v)]
    if len(vals) < 2:
        return None

    first, last = vals[0], vals[-1]
    n = len(vals) - 1
    if first <= 0 or last <= 0 or n <= 0:
        return None

    return (last / first) ** (1 / n) - 1


def compute_ma(series: pd.Series, window: int) -> Optional[float]:
    if series is None or len(series.dropna()) < window:
        return None
    return float(series.rolling(window).mean().iloc[-1])


def get_history(symbol: str, period: str = "2y", interval: str = "1d") -> Optional[pd.DataFrame]:
    try:
        df = yf.download(
            symbol,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if df is None or df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        return df.dropna(how="all")
    except Exception:
        return None


def get_info(ticker: yf.Ticker) -> Dict:
    try:
        return ticker.info or {}
    except Exception:
        return {}


def get_quarterly_net_income_or_eps(ticker: yf.Ticker) -> Tuple[Optional[float], Optional[float], str]:
    try:
        qearn = getattr(ticker, "quarterly_earnings", None)
        if qearn is not None and not qearn.empty:
            cols = [str(c).lower() for c in qearn.columns]
            if "earnings" in cols:
                col = qearn.columns[cols.index("earnings")]
                vals = qearn[col].dropna().tolist()
                if len(vals) >= 5:
                    return safe_float(vals[-1]), safe_float(vals[-5]), "quarterly_earnings"
    except Exception:
        pass

    try:
        q_is = ticker.quarterly_income_stmt
        if q_is is not None and not q_is.empty:
            idx_lower = [str(i).lower() for i in q_is.index]
            candidates = ["net income", "netincome", "normalized income"]
            row_name = None
            for c in candidates:
                if c in idx_lower:
                    row_name = q_is.index[idx_lower.index(c)]
                    break
            if row_name is not None:
                row = q_is.loc[row_name]
                vals = [safe_float(v) for v in row.tolist()]
                vals = [v for v in vals if v is not None]
                if len(vals) >= 5:
                    return vals[0], vals[4], "quarterly_net_income"
    except Exception:
        pass

    return None, None, "missing"


def get_annual_eps_or_net_income_series(ticker: yf.Ticker) -> Tuple[List[float], str]:
    try:
        earnings = getattr(ticker, "earnings", None)
        if earnings is not None and not earnings.empty:
            cols = [str(c).lower() for c in earnings.columns]
            if "earnings" in cols:
                col = earnings.columns[cols.index("earnings")]
                vals = [safe_float(v) for v in earnings[col].dropna().tolist()]
                if len(vals) >= 3:
                    return vals, "annual_earnings"
    except Exception:
        pass

    try:
        a_is = ticker.income_stmt
        if a_is is not None and not a_is.empty:
            idx_lower = [str(i).lower() for i in a_is.index]
            candidates = ["net income", "netincome", "normalized income"]
            row_name = None
            for c in candidates:
                if c in idx_lower:
                    row_name = a_is.index[idx_lower.index(c)]
                    break
            if row_name is not None:
                row = a_is.loc[row_name]
                vals = [safe_float(v) for v in row.tolist()]
                vals = [v for v in vals if v is not None]
                if len(vals) >= 3:
                    vals = list(reversed(vals))
                    return vals, "annual_net_income"
    except Exception:
        pass

    return [], "missing"


@dataclass
class ScreenResult:
    symbol: str
    pass_count: int
    c_current_growth: Optional[bool]
    a_cagr_3y: Optional[bool]
    a_recent_positive: Optional[bool]
    n_near_high: Optional[bool]
    n_trend: Optional[bool]
    l_rs: Optional[bool]
    i_inst: Optional[bool]
    s_volume: Optional[bool]
    market_ok: Optional[bool]

    current_growth_value: Optional[float] = None
    cagr_3y_value: Optional[float] = None
    rs_percentile: Optional[float] = None
    inst_ownership: Optional[float] = None
    price: Optional[float] = None
    high_52w: Optional[float] = None
    volume_ratio: Optional[float] = None
    avg_dollar_volume_20d: Optional[float] = None
    notes: str = ""


# =========================
# MARKET TREND (M)
# =========================
def market_trend_is_bullish() -> Tuple[bool, str]:
    notes = []
    ok = True

    for symbol in MARKET_ETFS:
        hist = get_history(symbol, period="1y")
        if hist is None or hist.empty:
            return False, f"{symbol}: no history"

        close = hist["Close"].dropna()
        if len(close) < 200:
            return False, f"{symbol}: insufficient history"

        last = float(close.iloc[-1])
        ma50 = compute_ma(close, 50)
        ma200 = compute_ma(close, 200)

        if symbol == "SPY":
            cond = (last > ma50) and (ma50 > ma200)
        else:
            cond = (last > ma200)

        notes.append(f"{symbol} last={last:.2f}, ma50={ma50:.2f}, ma200={ma200:.2f}, bull={cond}")
        ok = ok and cond

    return ok, " | ".join(notes)


# =========================
# RELATIVE STRENGTH
# =========================
def compute_relative_strength_percentiles(
    tickers: List[str],
    benchmark_symbol: str = BENCHMARK
) -> Dict[str, Dict[str, float]]:
    benchmark_hist = get_history(benchmark_symbol, period="1y")
    benchmark_12m = None
    if benchmark_hist is not None and len(benchmark_hist) > 230:
        bclose = benchmark_hist["Close"].dropna()
        benchmark_12m = float(bclose.iloc[-1] / bclose.iloc[0] - 1)

    rets_6m = {}
    rets_12m = {}

    for symbol in tickers:
        hist = get_history(symbol, period="1y")
        time.sleep(REQUEST_PAUSE)
        if hist is None or hist.empty:
            continue

        close = hist["Close"].dropna()
        if len(close) < 126:
            continue

        try:
            ret_6m = float(close.iloc[-1] / close.iloc[-126] - 1)
            ret_12m = float(close.iloc[-1] / close.iloc[0] - 1)
            rets_6m[symbol] = ret_6m
            rets_12m[symbol] = ret_12m
        except Exception:
            continue

    if not rets_6m:
        return {}

    series_6m = pd.Series(rets_6m)
    percentiles = series_6m.rank(pct=True) * 100

    result = {}
    for symbol in series_6m.index:
        result[symbol] = {
            "rs_percentile": float(percentiles.loc[symbol]),
            "ret_12m": float(rets_12m.get(symbol, np.nan)),
            "benchmark_12m": benchmark_12m if benchmark_12m is not None else np.nan,
        }

    return result


# =========================
# SINGLE STOCK SCREEN
# =========================
def screen_stock(symbol: str, rs_map: Dict[str, Dict[str, float]], market_ok: bool) -> ScreenResult:
    notes = []
    ticker = yf.Ticker(symbol)
    info = get_info(ticker)

    hist = get_history(symbol, period="1y")
    if hist is None or hist.empty:
        return ScreenResult(
            symbol=symbol,
            pass_count=0,
            c_current_growth=None,
            a_cagr_3y=None,
            a_recent_positive=None,
            n_near_high=None,
            n_trend=None,
            l_rs=None,
            i_inst=None,
            s_volume=None,
            market_ok=market_ok,
            notes="No price history",
        )

    close = hist["Close"].dropna()
    volume = hist["Volume"].dropna()

    last_price = float(close.iloc[-1])
    high_52w = float(close.max())
    ma50 = compute_ma(close, 50)
    ma200 = compute_ma(close, 200)

    avg_vol_50 = float(volume.tail(50).mean()) if len(volume) >= 50 else None
    latest_vol = float(volume.iloc[-1]) if len(volume) > 0 else None

    avg_dollar_volume_20d = None
    try:
        dv20 = (hist["Close"] * hist["Volume"]).tail(20)
        if len(dv20) >= 20:
            avg_dollar_volume_20d = float(dv20.mean())
    except Exception:
        pass

    latest_q, year_ago_q, c_field = get_quarterly_net_income_or_eps(ticker)
    current_growth_value = pct_change(latest_q, year_ago_q)
    c_current_growth = None if current_growth_value is None else (current_growth_value >= MIN_Q_GROWTH)
    notes.append(f"C source={c_field}")

    annual_series, a_field = get_annual_eps_or_net_income_series(ticker)
    cagr_3y_value = None
    a_cagr_3y = None
    a_recent_positive = None

    if len(annual_series) >= 4:
        last4 = annual_series[-4:]
        cagr_3y_value = calc_cagr(last4)
        a_cagr_3y = None if cagr_3y_value is None else (cagr_3y_value >= MIN_3Y_CAGR)
    elif len(annual_series) >= 3:
        cagr_3y_value = calc_cagr(annual_series)
        a_cagr_3y = None if cagr_3y_value is None else (cagr_3y_value >= MIN_3Y_CAGR)

    if len(annual_series) >= 3:
        a_recent_positive = all(v > 0 for v in annual_series[-3:])

    notes.append(f"A source={a_field}")

    n_near_high = (last_price >= NEAR_HIGH_RATIO * high_52w) if high_52w else None
    n_trend = None
    if ma50 is not None and ma200 is not None:
        n_trend = (last_price > ma50) and (ma50 > ma200)

    rs_percentile = None
    l_rs = None
    if symbol in rs_map:
        rs_percentile = safe_float(rs_map[symbol].get("rs_percentile"))
        stock_12m = safe_float(rs_map[symbol].get("ret_12m"))
        bench_12m = safe_float(rs_map[symbol].get("benchmark_12m"))
        if rs_percentile is not None and stock_12m is not None and bench_12m is not None:
            l_rs = (rs_percentile >= MIN_RS_PERCENTILE) and (stock_12m > bench_12m)

    inst = safe_float(info.get("heldPercentInstitutions"))
    i_inst = None if inst is None else (inst >= MIN_INST_OWNERSHIP)

    volume_ratio = None
    s_volume = None
    if latest_vol is not None and avg_vol_50 is not None and avg_vol_50 > 0:
        volume_ratio = latest_vol / avg_vol_50
        liquid_enough = (avg_dollar_volume_20d is not None and avg_dollar_volume_20d >= MIN_DOLLAR_VOLUME_20D)
        s_volume = (volume_ratio >= MIN_VOLUME_SURGE) and liquid_enough

    checks = [
        c_current_growth,
        a_cagr_3y,
        a_recent_positive,
        n_near_high,
        n_trend,
        l_rs,
        i_inst,
        s_volume,
    ]
    pass_count = sum(1 for x in checks if x is True)

    return ScreenResult(
        symbol=symbol,
        pass_count=pass_count,
        c_current_growth=c_current_growth,
        a_cagr_3y=a_cagr_3y,
        a_recent_positive=a_recent_positive,
        n_near_high=n_near_high,
        n_trend=n_trend,
        l_rs=l_rs,
        i_inst=i_inst,
        s_volume=s_volume,
        market_ok=market_ok,
        current_growth_value=current_growth_value,
        cagr_3y_value=cagr_3y_value,
        rs_percentile=rs_percentile,
        inst_ownership=inst,
        price=last_price,
        high_52w=high_52w,
        volume_ratio=volume_ratio,
        avg_dollar_volume_20d=avg_dollar_volume_20d,
        notes="; ".join(notes),
    )


# =========================
# FORMATTERS
# =========================
def bool_icon(v: Optional[bool]) -> str:
    if v is True:
        return "✅"
    if v is False:
        return "❌"
    return "➖"


def pct_fmt(v: Optional[float]) -> str:
    if v is None or pd.isna(v):
        return "N/A"
    return f"{v * 100:.1f}%"


def money_fmt(v: Optional[float]) -> str:
    if v is None or pd.isna(v):
        return "N/A"
    if v >= 1_000_000_000:
        return f"${v / 1_000_000_000:.2f}B"
    if v >= 1_000_000:
        return f"${v / 1_000_000:.2f}M"
    return f"${v:,.0f}"


def fmt_num(v: Optional[float], ndigits: int = 2) -> str:
    if v is None or pd.isna(v):
        return "N/A"
    return f"{v:.{ndigits}f}"


def build_message(results: List[ScreenResult], market_note: str, run_label: str, universe_size: int) -> str:
    if not results:
        return (
            f"*Modern CANSLIM Screen*\n\n"
            f"Run time: `{run_label}`\n"
            f"Universe: S&P 500 constituents ({universe_size} tickers)\n"
            f"Market trend: ✅\n"
            f"{market_note}\n\n"
            f"No stocks passed this run."
        )

    lines = [
        "*Modern CANSLIM Screen*",
        "",
        f"Run time: `{run_label}`",
        f"Universe: S&P 500 constituents ({universe_size} tickers)",
        "Market trend: ✅",
        market_note,
        "",
        f"Passed (>= {MIN_PASS_COUNT} rules): *{len(results)}*",
        "",
    ]

    for r in results[:MAX_MESSAGE_STOCKS]:
        lines.extend([
            f"*{r.symbol}* | score: *{r.pass_count}*",
            f"C {bool_icon(r.c_current_growth)} {pct_fmt(r.current_growth_value)} | "
            f"A1 {bool_icon(r.a_cagr_3y)} {pct_fmt(r.cagr_3y_value)} | "
            f"A2 {bool_icon(r.a_recent_positive)}",
            f"N1 {bool_icon(r.n_near_high)} price {fmt_num(r.price)} / 52wH {fmt_num(r.high_52w)} | "
            f"N2 {bool_icon(r.n_trend)}",
            f"L {bool_icon(r.l_rs)} RS {fmt_num(r.rs_percentile, 1)} | "
            f"I {bool_icon(r.i_inst)} {pct_fmt(r.inst_ownership)} | "
            f"S {bool_icon(r.s_volume)} vol ratio {fmt_num(r.volume_ratio)}",
            f"20d $vol: {money_fmt(r.avg_dollar_volume_20d)}",
            "",
        ])

    return "\n".join(lines)[:3900]


# =========================
# MAIN
# =========================
def main():
    require_env()

    now_utc = datetime.now(timezone.utc)
    if ZoneInfo is not None:
        now_bj = now_utc.astimezone(ZoneInfo("Asia/Shanghai"))
        run_label = now_bj.strftime("%Y-%m-%d %H:%M:%S CST")
    else:
        run_label = now_utc.strftime("%Y-%m-%d %H:%M:%S UTC")

    print(f"Run time UTC: {now_utc.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"Run label: {run_label}")

    tickers = get_sp500_tickers()
    print(f"Loaded {len(tickers)} S&P 500 tickers")

    market_ok, market_note = market_trend_is_bullish()
    print(f"Market trend ok: {market_ok}")
    print(market_note)

    if not market_ok:
        msg = (
            f"*Modern CANSLIM Screen*\n\n"
            f"Run time: `{run_label}`\n"
            f"Universe: S&P 500 constituents ({len(tickers)} tickers)\n"
            f"Market trend: ❌\n"
            f"{market_note}\n\n"
            f"No signals sent because market filter is not bullish."
        )
        print(msg)
        send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, msg)
        pd.DataFrame(columns=["symbol"]).to_csv("canslim_results.csv", index=False)
        return

    print("Computing relative strength map...")
    rs_map = compute_relative_strength_percentiles(tickers, BENCHMARK)

    results: List[ScreenResult] = []
    failed: List[Tuple[str, str]] = []

    for i, symbol in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] Screening {symbol}")
        try:
            result = screen_stock(symbol, rs_map, market_ok)
            if result.pass_count >= MIN_PASS_COUNT:
                results.append(result)
        except Exception as e:
            failed.append((symbol, str(e)))
            print(f"{symbol} failed: {e}")
        time.sleep(REQUEST_PAUSE)

    results.sort(
        key=lambda x: (
            x.pass_count,
            x.rs_percentile if x.rs_percentile is not None else -1,
            x.current_growth_value if x.current_growth_value is not None else -999,
        ),
        reverse=True,
    )

    msg = build_message(results, market_note, run_label, len(tickers))
    print(msg)
    send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, msg)

    out = pd.DataFrame([asdict(r) for r in results])
    out.to_csv("canslim_results.csv", index=False)
    print("Saved to canslim_results.csv")

    if failed:
        fail_df = pd.DataFrame(failed, columns=["symbol", "error"])
        fail_df.to_csv("canslim_failures.csv", index=False)
        print("Saved to canslim_failures.csv")


if __name__ == "__main__":
    main()
