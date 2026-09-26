"""
normalize.py — Name/address normalization and multi-key blocking.

Blocking strategy (multi-pass, high recall):
  NP  country + sorted-pair of two name content tokens
  NT  country + single long name token (>= 7 chars)
  ND  country + first 12 chars of normalized name
  Z   country + postal code
  AP  country + sorted-pair of two address content tokens
"""

import re
import unicodedata

# -----------------------------------------------------------------------
# Stop words
# -----------------------------------------------------------------------

NAME_STOP = frozenset([
    "the", "a", "an", "and", "of", "for", "in", "at", "by", "co",
    "llc", "ltd", "inc", "corp", "pvt", "private", "limited",
    "incorporated", "corporation", "group", "holdings", "services",
    "solutions", "enterprises", "associates", "international",
    "national", "global", "india", "us", "new", "old",
    "sri", "shri", "formerly", "fka", "dba", "company",
])

ADDR_STOP = frozenset([
    "the", "a", "an", "and", "of", "for", "in", "at", "by",
    "road", "street", "avenue", "ave", "rd", "st", "blvd", "lane",
    "ln", "suite", "ste", "apt", "floor", "fl", "unit", "no", "po",
    "box", "north", "south", "east", "west", "ne", "nw", "se", "sw",
    "india", "us", "usa",
])

# -----------------------------------------------------------------------
# Core normalisation helpers
# -----------------------------------------------------------------------

_DOMAIN_TLD_RE = re.compile(
    r"\.(com|org|net|in|co|io|biz|info)\b", re.IGNORECASE
)
_NON_WORD_RE   = re.compile(r"[^\w\s]")
_SPACE_RE      = re.compile(r"\s+")


def ascii_normalize(text: str) -> str:
    """
    Lowercase, transliterate non-ASCII to closest ASCII equivalent,
    strip domain TLDs, remove punctuation, collapse whitespace.
    """
    if not isinstance(text, str):
        return ""
    # Hard cap — corrupted rows can be megabytes long
    if len(text) > 500:
        text = text[:500]
    try:
        s = unicodedata.normalize("NFKD", text)
        s = s.encode("ascii", "ignore").decode("ascii")
    except Exception:
        s = text[:500]
    s = s.lower().strip()
    s = _DOMAIN_TLD_RE.sub(" ", s)
    s = _NON_WORD_RE.sub(" ", s)
    s = _SPACE_RE.sub(" ", s).strip()
    return s


def name_content_tokens(name: str, min_len: int = 3):
    """Meaningful tokens from a business name, stop-words removed."""
    return [t for t in ascii_normalize(name).split()
            if len(t) >= min_len and t not in NAME_STOP]


def addr_content_tokens(addr: str, min_len: int = 4):
    """Meaningful tokens from an address, stop-words + pure digits removed."""
    return [t for t in ascii_normalize(addr).split()
            if len(t) >= min_len and t not in ADDR_STOP and not t.isdigit()]


def build_block_keys(country: str, name: str, address: str) -> set:
    """
    Return a set of integer-hashed blocking keys for one record.
    Using hash(tuple) collapses each key to one Python int — ~8 bytes vs ~200.
    """
    keys: set = set()
    c = str(country).strip()

    ntoks = name_content_tokens(name)
    norm_name = ascii_normalize(name)

    # NP: all pairs from first 5 content tokens
    seen = ntoks[:5]
    for i in range(len(seen)):
        for j in range(i + 1, len(seen)):
            a, b = sorted([seen[i][:6], seen[j][:6]])
            keys.add(hash(("NP", c, a, b)))

    # NT: individual tokens >= 7 chars
    for t in ntoks[:6]:
        if len(t) >= 7:
            keys.add(hash(("NT", c, t[:9])))

    # ND: normalized name prefix
    if norm_name:
        keys.add(hash(("ND", c, norm_name[:12])))

    # Z: postal code
    m = re.search(r"\b(\d{4,6})\b", str(address))
    if m:
        keys.add(hash(("Z", c, m.group(1))))

    # AP: all pairs from first 4 address content tokens
    atoks = addr_content_tokens(address)[:4]
    for i in range(len(atoks)):
        for j in range(i + 1, len(atoks)):
            a, b = sorted([atoks[i][:6], atoks[j][:6]])
            keys.add(hash(("AP", c, a, b)))

    return keys


def build_block_keys_bulk(country_col, name_col, addr_col, cap_per_key=None):
    """
    Build a {key -> [entity_id, ...]} index from three parallel arrays/lists.
    Returns dict.  Used for building the pool lookup.
    cap_per_key: if set, each bucket is capped at this many entries.
    """
    index: dict = {}
    for country, name, addr in zip(country_col, name_col, addr_col):
        for k in build_block_keys(country, name, addr):
            if k not in index:
                index[k] = []
            index[k].append(None)  # placeholder; caller must pass entity_ids

    # This function is here for reference; actual index building uses
    # build_pool_index() below which passes entity_ids.
    return index


def build_pool_index(df, cap_per_key: int = 200) -> dict:
    """
    Given a DataFrame with [entity_id, business_name, business_address, country],
    build a multi-key blocking index: {key -> [entity_id, ...]}.

    cap_per_key: maximum number of entities stored per key bucket.
                 High-frequency generic keys (e.g. 'services') are capped to
                 avoid blowing up the candidate set.
    """
    index: dict = {}
    for row in df.itertuples(index=False):
        eid = row.entity_id
        for k in build_block_keys(row.country, row.business_name, row.business_address):
            bucket = index.setdefault(k, [])
            if cap_per_key is None or len(bucket) < cap_per_key:
                bucket.append(eid)
    return index


def lookup_candidates(s1_country, s1_name, s1_addr, index: dict) -> set:
    """
    Return the set of candidate entity IDs for one S1 record.
    """
    cands: set = set()
    for k in build_block_keys(s1_country, s1_name, s1_addr):
        bucket = index.get(k)
        if bucket:
            cands.update(bucket)
    return cands


# -----------------------------------------------------------------------
# Legacy helpers kept for backward compatibility
# -----------------------------------------------------------------------

LEGAL_SUFFIXES = {
    r"\bpvt\.?\b": "private",
    r"\bltd\.?\b": "limited",
    r"\bcorp\.?\b": "corporation",
    r"\binc\.?\b": "incorporated",
    r"\bllc\b": "llc",
    r"\bllp\b": "llp",
}

ADDRESS_ABBREVS = {
    r"\brd\.?\b": "road",
    r"\bst\.?\b": "street",
    r"\bave\.?\b": "avenue",
    r"\bapt\.?\b": "apartment",
    r"\bblvd\.?\b": "boulevard",
}


def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    return ascii_normalize(name)


def normalize_address(address: str) -> str:
    if not isinstance(address, str):
        return ""
    return ascii_normalize(address)


def extract_postal_code(address: str) -> str:
    if not isinstance(address, str):
        return ""
    match = re.search(r"\b\d{5,6}\b", address)
    return match.group(0) if match else ""


# Legacy block-key used by old blocking.py — kept so old scripts still run.
_FILLER_WORDS = NAME_STOP

def make_block_key(country: str, name: str) -> str:
    norm = normalize_name(name)
    tokens = [t for t in norm.split() if t not in _FILLER_WORDS]
    key_str = " ".join(sorted(tokens)) if tokens else norm
    return str(country).strip() + "_" + key_str[:8]


def make_block_keys_series(country_series, name_series):
    raw_names   = name_series.fillna("").to_numpy(dtype=str)
    raw_country = country_series.fillna("").astype(str).to_numpy(dtype=str)
    keys = []
    for country, name in zip(raw_country, raw_names):
        keys.append(make_block_key(country, name))
    return keys
