r"""
utils.py — single source of truth for config, tickers, and trusted sources.
All scripts import from here instead of maintaining their own hardcoded dicts.
"""
import os

import yaml

CONFIG_PATH = "gdelt_config.yaml"
_cfg = None


def get_config():
    global _cfg
    if _cfg is None:
        config_path = os.path.join(os.path.dirname(__file__), CONFIG_PATH)
        with open(config_path, 'r', encoding='utf-8') as fh:
            _cfg = yaml.safe_load(fh) or {}
    return _cfg


def get_tickers():
    """Returns merged dict of all gdelt_query -> ticker mappings."""
    t = get_config().get('tickers', {})
    company = {k: v for k, v in (t.get('company_queries', {}) or {}).items() if v}
    plain   = {k: v for k, v in (t.get('plain_names',    {}) or {}).items() if v}
    sector  = {k: v for k, v in (t.get('sector_queries', {}) or {}).items() if v}
    return {**company, **plain, **sector}


def get_benchmarks():
    """Returns dict of ticker -> benchmark index ticker."""
    return get_config().get('tickers', {}).get('benchmarks', {})


def get_trusted_sources():
    """Returns set of trusted domain names."""
    return set(get_config().get('trusted_sources', []))


def get_watchlist():
    """Returns (companies, sectors, macro) lists."""
    wl = get_config().get('watchlist', {})
    return wl.get('companies', []), wl.get('sectors', []), wl.get('macro', [])


def get_private_companies():
    """Returns set of known private company query strings — no ticker, skip silently."""
    return set(get_config().get('tickers', {}).get('private_companies', []))
