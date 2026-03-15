import json, math, os, time, warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None

warnings.filterwarnings("ignore")
BASE_DIR = Path(__file__).resolve().parent
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "arcee-ai/trinity-large-preview:free")
OPENROUTER_API_URL = os.getenv("OPENROUTER_API_URL", "https://openrouter.ai/api/v1/chat/completions")
OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL")
OPENROUTER_APP_NAME = os.getenv("OPENROUTER_APP_NAME", "Monthly CANSLIM AI Screen")
AI_ENABLED = os.getenv("AI_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
AI_REVIEW_MAX_STOCKS = int(os.getenv("AI_REVIEW_MAX_STOCKS", "12"))
AI_TIMEOUT = int(os.getenv("AI_TIMEOUT", "60"))
AI_TEMPERATURE = float(os.getenv("AI_TEMPERATURE", "0.2"))
AI_MAX_OUTPUT_TOKENS = min(int(os.getenv("AI_MAX_OUTPUT_TOKENS", "2000")), 2000)
MIN_PASS_COUNT = int(os.getenv("MIN_PASS_COUNT", "5"))
MIN_Q_GROWTH = float(os.getenv("MIN_Q_GROWTH", "0.20"))
MIN_3Y_CAGR = float(os.getenv("MIN_3Y_CAGR", "0.15"))
NEAR_HIGH_RATIO = float(os.getenv("NEAR_HIGH_RATIO", "0.95"))
MIN_RS_PERCENTILE = float(os.getenv("MIN_RS_PERCENTILE", "80"))
MIN_INST_OWNERSHIP = float(os.getenv("MIN_INST_OWNERSHIP", "0.35"))
MIN_VOLUME_SURGE = float(os.getenv("MIN_VOLUME_SURGE", "1.30"))
MIN_DOLLAR_VOLUME_20D = float(os.getenv("MIN_DOLLAR_VOLUME_20D", "20000000"))
PRE_NEAR_HIGH_RATIO = float(os.getenv("PRE_NEAR_HIGH_RATIO", "0.90"))
PRE_MIN_RS_PERCENTILE = float(os.getenv("PRE_MIN_RS_PERCENTILE", "60"))
PRE_MIN_DOLLAR_VOLUME_20D = float(os.getenv("PRE_MIN_DOLLAR_VOLUME_20D", "10000000"))
BENCHMARK = os.getenv("BENCHMARK", "SPY")
MARKET_ETFS = ["SPY", "QQQ"]
REQUEST_PAUSE = float(os.getenv("REQUEST_PAUSE", "0.05"))
MAX_MESSAGE_STOCKS = int(os.getenv("MAX_MESSAGE_STOCKS", "20"))
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "30"))
SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SP500_FALLBACK_CSV_URL = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept-Language": "en-US,en;q=0.9"}


def require_env():
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        raise ValueError("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")


def tg(text: str, parse_mode: Optional[str] = "Markdown"):
    body = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "disable_web_page_preview": True}
    if parse_mode:
        body["parse_mode"] = parse_mode
    r = requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage", json=body, timeout=HTTP_TIMEOUT)
    r.raise_for_status()


def safe_get(url: str):
    r = requests.get(url, headers=HEADERS, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r


def norm_tickers(items: List[str]) -> List[str]:
    return sorted({str(x).strip().upper().replace('.', '-') for x in items if str(x).strip()})


def sp500_wiki() -> List[str]:
    tables = pd.read_html(StringIO(safe_get(SP500_WIKI_URL).text))
    if not tables or "Symbol" not in tables[0].columns:
        raise ValueError("Wikipedia table missing Symbol")
    return norm_tickers(tables[0]["Symbol"].tolist())


def sp500_csv() -> List[str]:
    df = pd.read_csv(StringIO(safe_get(SP500_FALLBACK_CSV_URL).text))
    if "Symbol" not in df.columns:
        raise ValueError("Fallback CSV missing Symbol")
    return norm_tickers(df["Symbol"].tolist())


def get_sp500() -> Tuple[List[str], str]:
    errs = []
    for fn, name in ((sp500_wiki, "wikipedia"), (sp500_csv, "datahub_csv")):
        try:
            tickers = fn()
            if tickers:
                return tickers, name
        except Exception as exc:
            errs.append(f"{name} failed: {exc}")
    raise RuntimeError("Unable to load S&P 500 tickers. " + " | ".join(errs))


def sf(x: Any) -> Optional[float]:
    try:
        v = float(x)
        return None if math.isnan(v) else v
    except Exception:
        return None


def pct_change(new: Optional[float], old: Optional[float]) -> Optional[float]:
    return None if new is None or old in (None, 0) else (new - old) / abs(old)


def calc_cagr(values: List[float]) -> Optional[float]:
    vals = [v for v in values if v is not None and pd.notna(v)]
    if len(vals) < 2 or vals[0] <= 0 or vals[-1] <= 0:
        return None
    return (vals[-1] / vals[0]) ** (1 / (len(vals) - 1)) - 1


def ma(series: pd.Series, window: int) -> Optional[float]:
    return None if series is None or len(series.dropna()) < window else float(series.rolling(window).mean().iloc[-1])


def hist(symbol: str, period: str = "1y", interval: str = "1d", retries: int = 2):
    last_error = None
    for _ in range(retries + 1):
        try:
            df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False, threads=False)
            if df is None or df.empty:
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna(how="all")
            return None if df.empty else df
        except Exception as exc:
            last_error = exc
            time.sleep(0.5)
    if last_error:
        print(f"{symbol} history failed: {last_error}")
    return None


def ticker_info(t):
    try:
        info = t.info
        return info if isinstance(info, dict) else {}
    except Exception:
        return {}


def quarterly_values(t) -> Tuple[Optional[float], Optional[float], str]:
    try:
        q = getattr(t, "quarterly_earnings", None)
        if q is not None and not q.empty:
            cols = [str(c).lower() for c in q.columns]
            if "earnings" in cols:
                vals = [sf(v) for v in q[q.columns[cols.index("earnings")]].dropna().tolist()]
                vals = [v for v in vals if v is not None]
                if len(vals) >= 5:
                    return vals[-1], vals[-5], "quarterly_earnings"
    except Exception:
        pass
    try:
        q = t.quarterly_income_stmt
        if q is not None and not q.empty:
            idx = [str(i).lower() for i in q.index]
            for key in ["net income", "netincome", "normalized income"]:
                if key in idx:
                    vals = [sf(v) for v in q.loc[q.index[idx.index(key)]].tolist()]
                    vals = [v for v in vals if v is not None]
                    if len(vals) >= 5:
                        return vals[0], vals[4], "quarterly_net_income"
    except Exception:
        pass
    return None, None, "missing"


def annual_values(t) -> Tuple[List[float], str]:
    try:
        e = getattr(t, "earnings", None)
        if e is not None and not e.empty:
            cols = [str(c).lower() for c in e.columns]
            if "earnings" in cols:
                vals = [sf(v) for v in e[e.columns[cols.index("earnings")]].dropna().tolist()]
                vals = [v for v in vals if v is not None]
                if len(vals) >= 3:
                    return vals, "annual_earnings"
    except Exception:
        pass
    try:
        q = t.income_stmt
        if q is not None and not q.empty:
            idx = [str(i).lower() for i in q.index]
            for key in ["net income", "netincome", "normalized income"]:
                if key in idx:
                    vals = [sf(v) for v in q.loc[q.index[idx.index(key)]].tolist()]
                    vals = [v for v in vals if v is not None]
                    if len(vals) >= 3:
                        return list(reversed(vals)), "annual_net_income"
    except Exception:
        pass
    return [], "missing"


@dataclass
class Result:
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


def market_filter() -> Tuple[bool, str]:
    notes, ok = [], True
    for symbol in MARKET_ETFS:
        df = hist(symbol, period="1y")
        if df is None or df.empty:
            return False, f"{symbol}: no history"
        close = df["Close"].dropna()
        if len(close) < 200:
            return False, f"{symbol}: insufficient history"
        last, ma50, ma200 = float(close.iloc[-1]), ma(close, 50), ma(close, 200)
        if ma50 is None or ma200 is None:
            return False, f"{symbol}: moving averages unavailable"
        cond = (last > ma50 and ma50 > ma200) if symbol == "SPY" else (last > ma200)
        notes.append(f"{symbol} last={last:.2f}, ma50={ma50:.2f}, ma200={ma200:.2f}, bull={cond}")
        ok = ok and cond
    return ok, " | ".join(notes)


def build_rs_map(tickers: List[str]) -> Dict[str, Dict[str, float]]:
    bench = hist(BENCHMARK, period="1y")
    bench_12m = None
    if bench is not None and len(bench) > 230:
        c = bench["Close"].dropna()
        bench_12m = float(c.iloc[-1] / c.iloc[0] - 1)
    r6, r12 = {}, {}
    for symbol in tickers:
        df = hist(symbol, period="1y")
        time.sleep(REQUEST_PAUSE)
        if df is None or df.empty:
            continue
        close = df["Close"].dropna()
        if len(close) < 126:
            continue
        try:
            r6[symbol] = float(close.iloc[-1] / close.iloc[-126] - 1)
            r12[symbol] = float(close.iloc[-1] / close.iloc[0] - 1)
        except Exception:
            continue
    if not r6:
        return {}
    p = pd.Series(r6).rank(pct=True) * 100
    return {s: {"rs_percentile": float(p.loc[s]), "ret_12m": float(r12.get(s, np.nan)), "benchmark_12m": bench_12m if bench_12m is not None else np.nan} for s in p.index}


def pre_screen(tickers: List[str], rs_map: Dict[str, Dict[str, float]]) -> List[str]:
    out = []
    for symbol in tickers:
        df = hist(symbol, period="1y")
        time.sleep(REQUEST_PAUSE)
        if df is None or df.empty:
            continue
        close, volume = df["Close"].dropna(), df["Volume"].dropna()
        if len(close) < 200 or len(volume) < 20:
            continue
        last, high, ma200 = float(close.iloc[-1]), float(close.max()), ma(close, 200)
        dv20 = (df["Close"] * df["Volume"]).tail(20)
        avg_dv = float(dv20.mean()) if len(dv20) >= 20 else None
        rs = sf(rs_map.get(symbol, {}).get("rs_percentile"))
        score = sum([
            last >= PRE_NEAR_HIGH_RATIO * high if high else False,
            (ma200 is not None) and (last > ma200),
            (avg_dv is not None) and (avg_dv >= PRE_MIN_DOLLAR_VOLUME_20D),
            (rs is not None) and (rs >= PRE_MIN_RS_PERCENTILE),
        ])
        if score >= 3:
            out.append(symbol)
    return out


def screen(symbol: str, rs_map: Dict[str, Dict[str, float]], market_ok: bool) -> Result:
    notes, t = [], yf.Ticker(symbol)
    info = ticker_info(t)
    df = hist(symbol, period="1y")
    if df is None or df.empty:
        return Result(symbol, 0, None, None, None, None, None, None, None, None, market_ok, notes="No price history")
    close, volume = df["Close"].dropna(), df["Volume"].dropna()
    last, high = float(close.iloc[-1]), float(close.max())
    ma50, ma200 = ma(close, 50), ma(close, 200)
    avg_vol_50 = float(volume.tail(50).mean()) if len(volume) >= 50 else None
    latest_vol = float(volume.iloc[-1]) if len(volume) > 0 else None
    dv20 = (df["Close"] * df["Volume"]).tail(20)
    avg_dv = float(dv20.mean()) if len(dv20) >= 20 else None
    latest_q, year_q, csrc = quarterly_values(t)
    current_growth = pct_change(latest_q, year_q)
    c_ok = None if current_growth is None else current_growth >= MIN_Q_GROWTH
    notes.append(f"C source={csrc}")
    annual, asrc = annual_values(t)
    cagr_3y = calc_cagr(annual[-4:] if len(annual) >= 4 else annual)
    a_ok = None if cagr_3y is None else cagr_3y >= MIN_3Y_CAGR
    a_recent = all(v > 0 for v in annual[-3:]) if len(annual) >= 3 else None
    notes.append(f"A source={asrc}")
    n1 = (last >= NEAR_HIGH_RATIO * high) if high else None
    n2 = None if (ma50 is None or ma200 is None) else (last > ma50 and ma50 > ma200)
    rs = sf(rs_map.get(symbol, {}).get("rs_percentile")) if symbol in rs_map else None
    s12 = sf(rs_map.get(symbol, {}).get("ret_12m")) if symbol in rs_map else None
    b12 = sf(rs_map.get(symbol, {}).get("benchmark_12m")) if symbol in rs_map else None
    l_ok = (rs is not None and s12 is not None and b12 is not None and rs >= MIN_RS_PERCENTILE and s12 > b12)
    inst = sf(info.get("heldPercentInstitutions"))
    i_ok = None if inst is None else inst >= MIN_INST_OWNERSHIP
    vr, s_ok = None, None
    if latest_vol is not None and avg_vol_50 is not None and avg_vol_50 > 0:
        vr = latest_vol / avg_vol_50
        liquid = (avg_dv is not None) and (avg_dv >= MIN_DOLLAR_VOLUME_20D)
        s_ok = (vr >= MIN_VOLUME_SURGE) and liquid
    checks = [c_ok, a_ok, a_recent, n1, n2, l_ok, i_ok, s_ok]
    return Result(symbol, sum(1 for x in checks if x is True), c_ok, a_ok, a_recent, n1, n2, l_ok, i_ok, s_ok, market_ok, current_growth, cagr_3y, rs, inst, last, high, vr, avg_dv, "; ".join(notes))


def rv(x: Optional[float], n: int = 4):
    return None if x is None or pd.isna(x) else round(float(x), n)


def ai_payload(r: Result) -> Dict[str, Any]:
    return {
        "symbol": r.symbol,
        "pass_count": r.pass_count,
        "checks": {
            "c_current_growth": r.c_current_growth, "a_cagr_3y": r.a_cagr_3y, "a_recent_positive": r.a_recent_positive,
            "n_near_high": r.n_near_high, "n_trend": r.n_trend, "l_rs": r.l_rs, "i_inst": r.i_inst, "s_volume": r.s_volume,
        },
        "metrics": {
            "current_growth_value": rv(r.current_growth_value), "cagr_3y_value": rv(r.cagr_3y_value), "rs_percentile": rv(r.rs_percentile, 2),
            "inst_ownership": rv(r.inst_ownership), "price": rv(r.price, 2), "high_52w": rv(r.high_52w, 2),
            "volume_ratio": rv(r.volume_ratio), "avg_dollar_volume_20d": rv(r.avg_dollar_volume_20d, 2),
        },
        "notes": r.notes,
    }


def parse_ai(raw: str, allowed_symbols: List[str]) -> Dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("No JSON object found in AI response")
    payload = json.loads(raw[start:end + 1])
    allowed = set(allowed_symbols)
    seen, reviews, top = set(), [], []
    for item in payload.get("reviews", []):
        symbol = str(item.get("symbol", "")).strip().upper()
        if symbol not in allowed or symbol in seen:
            continue
        seen.add(symbol)
        reviews.append({
            "symbol": symbol,
            "verdict": {"buy": "focus", "strong_buy": "focus", "hold": "watch", "skip": "pass"}.get(str(item.get("verdict", "")).strip().lower(), str(item.get("verdict", "watch")).strip().lower()),
            "confidence": sf(item.get("confidence")),
            "reason": str(item.get("reason", "")).strip(),
            "risk": str(item.get("risk", "")).strip(),
        })
    for item in payload.get("top_picks", []):
        symbol = str(item).strip().upper()
        if symbol in allowed and symbol not in top:
            top.append(symbol)
    return {"summary": str(payload.get("summary", "")).strip(), "top_picks": top[:5], "reviews": reviews, "missing_symbols": [s for s in allowed_symbols if s not in seen]}


def review(results: List[Result], market_note: str, run_label: str) -> Dict[str, Any]:
    if not results:
        return {"status": "no_results", "summary": "机械筛选本轮没有产出候选标的，因此跳过 AI 复核。", "top_picks": [], "reviews": [], "missing_symbols": []}
    if not AI_ENABLED:
        return {"status": "disabled", "summary": "AI_ENABLED=0，已跳过 AI 复核。", "top_picks": [], "reviews": [], "missing_symbols": []}
    if not OPENROUTER_API_KEY:
        return {"status": "missing_api_key", "summary": "缺少 OPENROUTER_API_KEY，已跳过 AI 复核。", "top_picks": [], "reviews": [], "missing_symbols": []}
    shortlist = results[:AI_REVIEW_MAX_STOCKS]
    allowed = [r.symbol for r in shortlist]
    payload = {
        "run_label": run_label,
        "market_note": market_note,
        "mechanical_rule_thresholds": {
            "min_pass_count": MIN_PASS_COUNT, "min_q_growth": MIN_Q_GROWTH, "min_3y_cagr": MIN_3Y_CAGR,
            "near_high_ratio": NEAR_HIGH_RATIO, "min_rs_percentile": MIN_RS_PERCENTILE,
            "min_inst_ownership": MIN_INST_OWNERSHIP, "min_volume_surge": MIN_VOLUME_SURGE,
            "min_dollar_volume_20d": MIN_DOLLAR_VOLUME_20D,
        },
        "stocks": [ai_payload(r) for r in shortlist],
    }
    prompt = (
        "你是月度选股流程中的第二层 AI 复核助手。下面这些股票已经通过第一层机械 CANSLIM-like 筛选，"
        "你不能推翻机械筛选结果，只能在机械筛选结果之上给出第二层主观判断。\n\n"
        "你的任务：\n"
        "1. 只能基于我提供的字段进行判断，不能补充任何外部信息。\n"
        "2. 对每只股票给出一个 verdict，取值只能是 focus、watch、pass。\n"
        "3. 更偏好增长更强、RS 更高、趋势更完整、更接近 52 周高点、机构持股更合理、量能确认更充分的标的。\n"
        "4. 如果关键字段缺失、量价配合不完整、或者只是勉强通过机械阈值，要更谨慎。\n"
        "5. 输出必须是严格 JSON，不要 Markdown，不要代码块，不要解释性前言。\n\n"
        "输出格式：\n"
        "{\n"
        "  \"summary\": \"一句话总结本轮候选的整体质量\",\n"
        "  \"top_picks\": [\"示例代码1\", \"示例代码2\"],\n"
        "  \"reviews\": [\n"
        "    {\n"
        "      \"symbol\": \"输入中的股票代码\",\n"
        "      \"verdict\": \"focus\",\n"
        "      \"confidence\": 0.82,\n"
        "      \"reason\": \"一句话说明给出该评级的核心原因\",\n"
        "      \"risk\": \"一句话说明最主要的风险点\"\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "额外规则：\n"
        "- top_picks 只能来自 verdict=focus 的股票。\n"
        "- top_picks 最多 5 个。\n"
        "- reviews 必须覆盖每一只输入股票。\n"
        "- reviews 中的 symbol 必须严格来自输入数据，不能虚构。\n"
        "- top_picks 中的股票代码必须严格来自输入数据，不能虚构。\n"
        "- 如果没有任何股票达到 focus，则 top_picks 返回空数组 []。\n"
        "- confidence 必须在 0 到 1 之间。\n"
        "- summary、reason、risk 必须全部使用简体中文。\n"
        "- 语气直接、专业、简洁，不要空话。\n\n"
        f"输入数据如下：\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    if OPENROUTER_SITE_URL:
        headers["HTTP-Referer"] = OPENROUTER_SITE_URL
    if OPENROUTER_APP_NAME:
        headers["X-Title"] = OPENROUTER_APP_NAME
    body = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": "你是一个严格的中文选股复核助手。只能使用提供的数据，且必须返回合法 JSON。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": AI_TEMPERATURE,
        "max_tokens": AI_MAX_OUTPUT_TOKENS,
    }
    try:
        r = requests.post(OPENROUTER_API_URL, headers=headers, json=body, timeout=AI_TIMEOUT)
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"].strip()
        out = parse_ai(raw, allowed)
        out.update({"status": "ok", "model": OPENROUTER_MODEL, "raw": raw})
        return out
    except Exception as exc:
        return {"status": "error", "summary": f"AI 复核失败：{exc}", "top_picks": [], "reviews": [], "missing_symbols": allowed, "raw": ""}


def bi(v):
    return "Y" if v is True else ("N" if v is False else "-")


def pf(v):
    return "N/A" if v is None or pd.isna(v) else f"{v * 100:.1f}%"


def mf(v):
    if v is None or pd.isna(v):
        return "N/A"
    if v >= 1_000_000_000:
        return f"${v / 1_000_000_000:.2f}B"
    if v >= 1_000_000:
        return f"${v / 1_000_000:.2f}M"
    return f"${v:,.0f}"


def nf(v, d=2):
    return "N/A" if v is None or pd.isna(v) else f"{v:.{d}f}"


def cf(v):
    return "N/A" if v is None or pd.isna(v) else f"{v * 100:.0f}%"


def build_message(results: List[Result], market_note: str, run_label: str, universe: int, candidates: int, source: str) -> str:
    if not results:
        return f"*Monthly CANSLIM Screen*\n\nRun time: `{run_label}`\nUniverse: S&P 500 ({universe} tickers, source: {source})\nPre-screen candidates: {candidates}\nMarket trend: YES\n{market_note}\n\nNo stocks passed this run."
    lines = ["*Monthly CANSLIM Screen*", "", f"Run time: `{run_label}`", f"Universe: S&P 500 ({universe} tickers, source: {source})", f"Pre-screen candidates: {candidates}", "Market trend: YES", market_note, "", f"Passed (>= {MIN_PASS_COUNT} rules): *{len(results)}*", ""]
    for r in results[:MAX_MESSAGE_STOCKS]:
        lines += [
            f"*{r.symbol}* | score: *{r.pass_count}*",
            f"C {bi(r.c_current_growth)} {pf(r.current_growth_value)} | A1 {bi(r.a_cagr_3y)} {pf(r.cagr_3y_value)} | A2 {bi(r.a_recent_positive)}",
            f"N1 {bi(r.n_near_high)} price {nf(r.price)} / 52wH {nf(r.high_52w)} | N2 {bi(r.n_trend)}",
            f"L {bi(r.l_rs)} RS {nf(r.rs_percentile, 1)} | I {bi(r.i_inst)} {pf(r.inst_ownership)} | S {bi(r.s_volume)} vol ratio {nf(r.volume_ratio)}",
            f"20d $vol: {mf(r.avg_dollar_volume_20d)}",
            "",
        ]
    return "\n".join(lines)[:3900]


def build_ai_message(data: Dict[str, Any], run_label: str) -> str:
    if data.get("status") == "no_results":
        return ""
    lines = [f"AI 复核（OpenRouter / {OPENROUTER_MODEL}）", f"运行时间: {run_label}", f"状态: {data.get('status')}"]
    if data.get("summary"):
        lines += ["", f"总评: {data['summary']}"]
    if data.get("top_picks"):
        lines.append(f"优先关注: {', '.join(data['top_picks'])}")
    if data.get("reviews"):
        lines += ["", "个股复核:"]
        order = {"focus": 0, "watch": 1, "pass": 2}
        reviews = sorted(data["reviews"], key=lambda x: (order.get(x.get("verdict", "watch"), 9), -(x.get("confidence") or 0), x.get("symbol", "")))
        for r in reviews[:AI_REVIEW_MAX_STOCKS]:
            lines.append(f"- {r['symbol']} | {r['verdict'].title()} | 置信度 {cf(r.get('confidence'))}")
            if r.get("reason"):
                lines.append(f"  原因: {r['reason']}")
            if r.get("risk"):
                lines.append(f"  风险: {r['risk']}")
    if data.get("missing_symbols"):
        lines += ["", f"AI 返回中缺失的股票: {', '.join(data['missing_symbols'])}"]
    return "\n".join(lines)[:3900]


def save_outputs(results: List[Result], failed: List[Tuple[str, str]], ai_data: Dict[str, Any]):
    pd.DataFrame([asdict(r) for r in results]).to_csv(BASE_DIR / "canslim_results.csv", index=False)
    pd.DataFrame(failed, columns=["symbol", "error"]).to_csv(BASE_DIR / "canslim_failures.csv", index=False)
    (BASE_DIR / "canslim_ai_review.json").write_text(json.dumps(ai_data, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(ai_data.get("reviews", [])).to_csv(BASE_DIR / "canslim_ai_reviews.csv", index=False)


def main():
    require_env()
    now_utc = datetime.now(timezone.utc)
    run_label = now_utc.strftime("%Y-%m-%d %H:%M:%S UTC") if ZoneInfo is None else now_utc.astimezone(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S CST")
    print(f"Run label: {run_label}")
    tickers, source = get_sp500()
    market_ok, market_note = market_filter()
    print(f"Loaded {len(tickers)} S&P 500 tickers from {source}")
    print(market_note)
    if not market_ok:
        msg = f"*Monthly CANSLIM Screen*\n\nRun time: `{run_label}`\nUniverse: S&P 500 ({len(tickers)} tickers, source: {source})\nMarket trend: NO\n{market_note}\n\nNo signals sent because market filter is not bullish."
        print(msg)
        tg(msg)
        save_outputs([], [], {"status": "market_blocked", "summary": "Market filter is not bullish.", "top_picks": [], "reviews": [], "missing_symbols": []})
        return
    print("Computing RS map...")
    rs = build_rs_map(tickers)
    print("Running pre-screen...")
    candidates = pre_screen(tickers, rs)
    print(f"Pre-screen kept {len(candidates)} / {len(tickers)}")
    results, failed = [], []
    for i, symbol in enumerate(candidates, 1):
        print(f"[{i}/{len(candidates)}] Screening {symbol}")
        try:
            r = screen(symbol, rs, market_ok)
            if r.pass_count >= MIN_PASS_COUNT:
                results.append(r)
        except Exception as exc:
            failed.append((symbol, str(exc)))
        time.sleep(REQUEST_PAUSE)
    results.sort(key=lambda x: (x.pass_count, x.rs_percentile if x.rs_percentile is not None else -1, x.current_growth_value if x.current_growth_value is not None else -999), reverse=True)
    msg = build_message(results, market_note, run_label, len(tickers), len(candidates), source)
    print(msg)
    tg(msg)
    ai = review(results, market_note, run_label)
    aimsg = build_ai_message(ai, run_label)
    if aimsg:
        print(aimsg)
        tg(aimsg, parse_mode=None)
    save_outputs(results, failed, ai)


if __name__ == "__main__":
    main()
