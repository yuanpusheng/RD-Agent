from __future__ import annotations

from typing import List, Optional

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
