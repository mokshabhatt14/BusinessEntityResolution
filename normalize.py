import re

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
    s = name.lower().strip()
    s = s.replace("&", " and ")
    for pattern, repl in LEGAL_SUFFIXES.items():
        s = re.sub(pattern, repl, s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def extract_postal_code(address: str) -> str:
    if not isinstance(address, str):
        return ""
    match = re.search(r"\b\d{5,6}\b", address)
    return match.group(0) if match else ""

def normalize_address(address: str) -> str:
    if not isinstance(address, str):
        return ""
    s = address.lower().strip()
    for pattern, repl in ADDRESS_ABBREVS.items():
        s = re.sub(pattern, repl, s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s