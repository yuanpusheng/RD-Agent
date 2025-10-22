from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from loguru import logger


def get_csi300_symbols() -> List[str]:
    try:
        import akshare as ak

        df = ak.index_stock_cons(symbol="000300")
        # Normalize to ticker format with exchange suffix? keep numeric code as string
        codes = df["品种代码"].astype(str).str.zfill(6).tolist() if "品种代码" in df.columns else df["code"].astype(str).tolist()
        return codes
    except Exception as e:
        logger.warning(f"Failed to load CSI300 from akshare: {e}; using sample list")
        return ["000001", "000002", "000333", "600000", "600519", "000651", "600036", "600104", "600703", "601318"]


def get_zz500_symbols() -> List[str]:
    try:
        import akshare as ak

        df = ak.index_stock_cons(symbol="000905")
        codes = df["品种代码"].astype(str).str.zfill(6).tolist() if "品种代码" in df.columns else df["code"].astype(str).tolist()
        return codes
    except Exception as e:
        logger.warning(f"Failed to load ZZ500 from akshare: {e}; fallback to CSI300 sample")
        return get_csi300_symbols()


def get_all_a_stock_symbols(exclude_st: bool = True, exclude_suspended: bool = True, limit: Optional[int] = None) -> List[str]:
    codes: List[str] = []
    try:
        import akshare as ak

        base = ak.stock_info_a_code_name()
        base["code"] = base["code"].astype(str).str.zfill(6)
        if exclude_st:
            base = base[~base["name"].str.contains("ST|\*ST|退", regex=True)]
        codes = base["code"].tolist()
        if exclude_suspended:
            try:
                spot = ak.stock_zh_a_spot_em()
                spot["代码"] = spot["代码"].astype(str).str.zfill(6)
                trading = spot[spot["成交量"] > 0]["代码"].tolist()
                codes = [c for c in codes if c in trading]
            except Exception as e2:
                logger.warning(f"Failed to filter suspended stocks: {e2}")
    except Exception as e:
        logger.warning(f"Failed to load full A-share universe from akshare: {e}; using CSI300 fallback")
        codes = get_csi300_symbols()

    if limit:
        codes = codes[:limit]
    return codes


def resolve_universe(universe: Sequence[str] | str) -> List[str]:
    if isinstance(universe, str):
        u = universe.upper()
        if u in ("CSI300", "HS300", "沪深300", "000300"):
            return get_csi300_symbols()
        if u in ("ZZ500", "中证500", "000905"):
            return get_zz500_symbols()
        # comma-separated custom list
        if "," in universe:
            return [x.strip() for x in universe.split(",") if x.strip()]
        # fallback single symbol
        return [universe]
    return list(universe)


def get_suspensions(date: Optional[str] = None) -> List[str]:
    """Return list of suspended symbols for given date using Akshare spot data as heuristic.
    If Akshare is unavailable, return empty list.
    """
    try:
        import akshare as ak

        spot = ak.stock_zh_a_spot_em()
        spot["代码"] = spot["代码"].astype(str).str.zfill(6)
        if "成交量" in spot.columns:
            return spot[spot["成交量"] == 0]["代码"].tolist()
        return []
    except Exception as e:
        logger.warning(f"get_suspensions fallback: {e}")
        return []


def get_limit_flags(d: Optional[str] = None) -> pd.DataFrame:
    """Best-effort limit up/down flags. Returns DataFrame [symbol, limit_up, limit_down].
    When data source unavailable, returns empty frame.
    """
    try:
        import akshare as ak

        # Use daily_board_limit_up/down as approximations if available
        df_up = ak.stock_zt_pool_em(date=d) if hasattr(ak, "stock_zt_pool_em") else pd.DataFrame(columns=["代码"])
        df_down = ak.stock_zt_pool_dtgc_em(date=d) if hasattr(ak, "stock_zt_pool_dtgc_em") else pd.DataFrame(columns=["代码"])
        up = set(df_up.get("代码", pd.Series(dtype=str)).astype(str).str.zfill(6).tolist())
        down = set(df_down.get("代码", pd.Series(dtype=str)).astype(str).str.zfill(6).tolist())
        syms = sorted(up | down)
        return pd.DataFrame({
            "symbol": syms,
            "limit_up": [s in up for s in syms],
            "limit_down": [s in down for s in syms],
        })
    except Exception as e:
        logger.warning(f"get_limit_flags fallback: {e}")
        return pd.DataFrame(columns=["symbol", "limit_up", "limit_down"])
