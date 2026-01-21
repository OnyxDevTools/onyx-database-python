"""Aggregate and string helpers for select() expressions."""

def avg(attribute: str) -> str:
    return f"avg({attribute})"


def sum(attribute: str) -> str:  # noqa: A001 - mirrors TypeScript helper name
    return f"sum({attribute})"


def count(attribute: str) -> str:
    return f"count({attribute})"


def min(attribute: str) -> str:
    return f"min({attribute})"


def max(attribute: str) -> str:
    return f"max({attribute})"


def std(attribute: str) -> str:
    return f"std({attribute})"


def variance(attribute: str) -> str:
    return f"variance({attribute})"


def median(attribute: str) -> str:
    return f"median({attribute})"


def upper(attribute: str) -> str:
    return f"upper({attribute})"


def lower(attribute: str) -> str:
    return f"lower({attribute})"


def substring(attribute: str, start: int, length: int) -> str:
    return f"substring({attribute},{start},{length})"


def replace(attribute: str, pattern: str, repl: str) -> str:
    pat = pattern.replace("'", "\\'")
    rep = repl.replace("'", "\\'")
    return f"replace({attribute}, '{pat}', '{rep}')"


def format(attribute: str, formatter: str) -> str:  # noqa: A001 - mirrors backend helper name
    fmt = formatter.replace("'", "\\'")
    return f"format({attribute}, '{fmt}')"


def percentile(attribute: str, p: float) -> str:
    return f"percentile({attribute}, {p})"
