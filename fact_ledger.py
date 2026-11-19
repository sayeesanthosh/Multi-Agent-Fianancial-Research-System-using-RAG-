
"""
Shared financial fact ledger + ledger-based calculation tools.

Facts are extracted deterministically from the financial statement pages
(never by the LLM). Calculation tools take a fiscal year or metric name and
read the ledger, so an agent cannot feed a wrong number into a calculation.
"""

import json
import re
from pathlib import Path

from financial_tools import (
    revenue_growth,
    calculate_cagr,
    profit_margin,
    operating_margin,
    debt_to_equity,
    current_ratio,
    return_on_equity,
    year_on_year_growth,
)
from evidence_tools import assess_source_quality

DEFAULT_PDF = "data/documents/apple_annual_report.pdf"
DEFAULT_LABEL = "Apple 2025 Form 10-K"
DEFAULT_SOURCE_META = {
    "source_type": "regulatory_filing",
    "publication_year": 2025,
    "has_url": False,
    "is_primary_source": True,
    "currency": "USD",
}


class LedgerError(ValueError):
    pass


# ==========================================
# METRIC DEFINITIONS
# ==========================================

STATEMENT_TITLES = {
    "operations": r"consolidated\s+statements?\s+of\s+operations",
    "balance_sheet": r"consolidated\s+balance\s+sheets?",
    "cash_flow": r"consolidated\s+statements?\s+of\s+cash\s+flows?",
}

METRIC_SPECS = [
    {"metric": "net_sales", "statement": "operations",
     "label": r"total\s+net\s+sales"},
    {"metric": "operating_income", "statement": "operations",
     "label": r"operating\s+income"},
    {"metric": "net_income", "statement": "operations",
     "label": r"net\s+income"},
    {"metric": "total_current_assets", "statement": "balance_sheet",
     "label": r"total\s+current\s+assets"},
    {"metric": "total_assets", "statement": "balance_sheet",
     "label": r"total\s+assets"},
    {"metric": "total_current_liabilities", "statement": "balance_sheet",
     "label": r"total\s+current\s+liabilities"},
    {"metric": "total_liabilities", "statement": "balance_sheet",
     "label": r"total\s+liabilities"},
    {"metric": "total_equity", "statement": "balance_sheet",
     "label": r"total\s+shareholders.{0,3}\s*equity"},
    {"metric": "total_liabilities_and_equity", "statement": "balance_sheet",
     "label": r"total\s+liabilities\s+and\s+shareholders.{0,3}\s*equity"},
    {"metric": "operating_cash_flow", "statement": "cash_flow",
     "label": r"cash\s+generated\s+by\s+operating\s+activities"},
]

# total_debt is deliberately NOT extracted: it is not a single statement line.
KNOWN_METRICS = {spec["metric"] for spec in METRIC_SPECS} | {"total_debt"}

NUM_TOKEN = r"\(?\d[\d,]*(?:\.\d+)?\)?"


# ==========================================
# EXTRACTION
# ==========================================

def _to_number(token):
    negative = token.startswith("(") and token.endswith(")")
    value = float(token.strip("()").replace(",", ""))
    return -value if negative else value


def _find_row(text, label_regex):
    pattern = re.compile(
        rf"^[ \t]*(?:{label_regex})\b[ \t\$\u00a0]*"
        rf"((?:{NUM_TOKEN})(?:[ \t\$\u00a0]+{NUM_TOKEN})*)[ \t]*$",
        re.IGNORECASE | re.MULTILINE,
    )

    match = pattern.search(text)

    if match is None:
        return None

    return match.group(0).strip(), re.findall(NUM_TOKEN, match.group(1))



def detect_year_columns(text):
    """
    Detect fiscal years even when PDF extraction splits dates and years
    across multiple lines.

    Examples:
        2025 2024 2023 -> [2025, 2024, 2023]
        2025, 2024     -> [2025, 2024]

        Also supports:
        September 27,
        2025
        September 28,
        2024
        September 30,
        2023
    """

    years = [
        int(y)
        for y in re.findall(r"\b(20\d{2})\b", text)
    ]

    best_run = []
    current_run = []

    for year in years:

        if current_run and current_run[-1] - 1 == year:
            current_run.append(year)

        else:
            current_run = [year]

        if len(current_run) > len(best_run):
            best_run = current_run.copy()

    if len(best_run) >= 2:
        return best_run

    return None



def _printed_page(text):
    match = re.search(r"Form\s+10-K\s*\|\s*(\d{1,3})", text)
    return int(match.group(1)) if match else None


class FactLedger:

    def __init__(self, source_label, source_path="", source_meta=None):
        self.source_label = source_label
        self.source_path = source_path
        self.source_meta = source_meta or {}
        self.facts = {}
        self.warnings = []

    def get(self, metric, year):
        return self.facts.get(f"{metric}_{year}")

    def as_table(self):
        if not self.facts:
            return "(ledger is empty)"

        rows = []

        for fact in sorted(
            self.facts.values(), key=lambda f: (f["metric"], -f["year"])
        ):
            page = f"pdf p.{fact['pdf_page']}"

            if fact["printed_page"]:
                page += f", printed p.{fact['printed_page']}"

            value = f"{fact['value']:,.2f}".rstrip("0").rstrip(".")

            rows.append(
                f"- {fact['fact_id']}: {value} "
                f"{fact['unit'] or 'UNIT UNKNOWN'} "
                f"[{fact['verification_status']}] "
                f"({fact['source']}, {page})"
            )

        return "\n".join(rows)

    def to_dict(self):
        return {
            "source": self.source_label,
            "source_path": self.source_path,
            "source_meta": self.source_meta,
            "facts": self.facts,
            "warnings": self.warnings,
        }

    def save(self, path):
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )


def extract_facts(pages, source_label=DEFAULT_LABEL,
                  source_meta=None, source_path=""):

    meta = dict(DEFAULT_SOURCE_META)
    meta.update(source_meta or {})

    ledger = FactLedger(source_label, source_path, meta)
    candidates = {}

    for page in pages:

        text = page.get("text") or ""
        pdf_page = page.get("page")

        if not text:
            continue

        printed = _printed_page(text)

        for statement, title in STATEMENT_TITLES.items():

            if not re.search(title, text, re.IGNORECASE):
                continue

            years = detect_year_columns(text)
            unit = (
                "USD million"
                if re.search(r"in\s+millions", text, re.IGNORECASE)
                else None
            )
            currency = "USD" if "$" in text else meta.get("currency")

            for spec in METRIC_SPECS:

                if spec["statement"] != statement:
                    continue

                row = _find_row(text, spec["label"])

                if row is None:
                    continue

                line, tokens = row

                if years is None:
                    ledger.warnings.append(
                        f"pdf p.{pdf_page}: found '{spec['metric']}' but could "
                        f"not detect year columns; fact not recorded."
                    )
                    continue

                values = [_to_number(t) for t in tokens]

                notes = []
                status = "verified"

                if unit is None:
                    status = "needs_verification"
                    notes.append("Unit ('in millions') not found on page.")

                if len(values) != len(years):
                    status = "needs_verification"
                    notes.append(
                        f"{len(values)} values but {len(years)} year columns."
                    )

                for index, value in enumerate(values):

                    if index >= len(years):
                        continue

                    year = years[index]
                    fact_id = f"{spec['metric']}_{year}"

                    candidates.setdefault(fact_id, []).append({
                        "fact_id": fact_id,
                        "metric": spec["metric"],
                        "value": value,
                        "year": year,
                        "unit": unit,
                        "currency": currency,
                        "source": source_label,
                        "pdf_page": pdf_page,
                        "printed_page": printed,
                        "evidence": line,
                        "extraction_method": "statement_line_regex",
                        "verification_status": status,
                        "notes": list(notes),
                    })

    for fact_id, items in candidates.items():

        first = items[0]
        distinct = {item["value"] for item in items}

        if len(distinct) > 1:
            first["verification_status"] = "needs_verification"
            first["notes"].append(
                f"Conflicting values across pages: {sorted(distinct)}"
            )
            ledger.warnings.append(f"{fact_id}: conflicting values {sorted(distinct)}")

        elif len(items) > 1:
            first["notes"].append(
                "Same value also on pdf pages "
                f"{[i['pdf_page'] for i in items[1:]]}"
            )

        ledger.facts[fact_id] = first

    _cross_check(ledger)

    found = {f["metric"] for f in ledger.facts.values()}

    for spec in METRIC_SPECS:
        if spec["metric"] not in found:
            ledger.warnings.append(
                f"No extraction for metric '{spec['metric']}'. "
                f"Treat it as missing."
            )

    ledger.warnings.append(
        "'total_debt' is not extracted. Do not use total_liabilities or "
        "total_current_liabilities as debt."
    )

    return ledger


def _mark(fact, note):
    fact["verification_status"] = "needs_verification"
    fact["notes"].append(note)


def _cross_check(ledger):

    years = sorted({f["year"] for f in ledger.facts.values()})

    for year in years:

        assets = ledger.get("total_assets", year)
        liabilities = ledger.get("total_liabilities", year)
        equity = ledger.get("total_equity", year)
        both = ledger.get("total_liabilities_and_equity", year)

        if assets and liabilities and equity:

            if abs(assets["value"] - (liabilities["value"] + equity["value"])) > 0.5:

                for fact in (assets, liabilities, equity):
                    _mark(fact, "Balance identity failed: assets != liabilities + equity.")

                ledger.warnings.append(
                    f"FY{year}: balance sheet identity failed; facts need verification."
                )

            else:
                for fact in (assets, liabilities, equity):
                    fact["notes"].append("Balance identity check passed.")

        if assets and both and abs(assets["value"] - both["value"]) > 0.5:
            _mark(assets, "total_assets != total_liabilities_and_equity.")
            _mark(both, "total_assets != total_liabilities_and_equity.")
            ledger.warnings.append(
                f"FY{year}: total assets differ from total liabilities and equity."
            )


def build_ledger(pdf_path=DEFAULT_PDF, source_label=DEFAULT_LABEL,
                 source_meta=None):

    from rag.document_loader import load_pdf

    pages = load_pdf(pdf_path)

    return extract_facts(pages, source_label, source_meta, pdf_path)


# ==========================================
# ACTIVE LEDGER (shared by all agents)
# ==========================================

_ACTIVE = {"ledger": None}


def set_active_ledger(ledger):
    _ACTIVE["ledger"] = ledger


def get_active_ledger():
    return _ACTIVE["ledger"]


def ensure_ledger(pdf_path=DEFAULT_PDF, source_label=DEFAULT_LABEL,
                  source_meta=None):

    if _ACTIVE["ledger"] is None:
        try:
            _ACTIVE["ledger"] = build_ledger(pdf_path, source_label, source_meta)
        except Exception as error:
            ledger = FactLedger(source_label, pdf_path, source_meta or {})
            ledger.warnings.append(f"Ledger build failed: {error}")
            _ACTIVE["ledger"] = ledger

    return _ACTIVE["ledger"]


# ==========================================
# LEDGER ACCESS HELPERS
# ==========================================

def _year(value):
    try:
        return int(float(str(value).strip()))
    except ValueError:
        raise LedgerError(
            f"year must be a fiscal year such as 2025, got {value!r}."
        )


def _metric(value):
    name = str(value).strip().lower().replace(" ", "_")

    if name not in KNOWN_METRICS:
        raise LedgerError(
            f"Unknown metric {value!r}. Valid metrics: {sorted(KNOWN_METRICS)}."
        )

    return name


def _require(metric, year):

    ledger = get_active_ledger()

    if ledger is None:
        raise LedgerError("No fact ledger is loaded.")

    metric = _metric(metric)
    year = _year(year)
    fact = ledger.get(metric, year)

    if fact is None:
        extra = ""

        if metric == "total_debt":
            extra = (
                " Do NOT use total_liabilities or total_current_liabilities "
                "as total debt."
            )

        raise LedgerError(
            f"Required fact '{metric}' for FY{year} is not in the verified "
            f"ledger. Report it as missing; do not substitute another "
            f"metric.{extra}"
        )

    if fact["verification_status"] != "verified":
        raise LedgerError(
            f"Fact '{fact['fact_id']}' has status "
            f"'{fact['verification_status']}' and cannot be used in "
            f"calculations. Notes: {fact['notes']}"
        )

    return fact


def _check_units(facts):
    units = {f["unit"] for f in facts}

    if len(units) != 1 or None in units:
        raise LedgerError(f"Inconsistent or unknown units: {units}.")


def _inputs(facts):
    return [
        {
            "fact_id": f["fact_id"],
            "value": f["value"],
            "unit": f["unit"],
            "pdf_page": f["pdf_page"],
            "printed_page": f["printed_page"],
        }
        for f in facts
    ]


def _ok(tool, value, unit, formula, facts):
    return {
        "status": "ok",
        "tool": tool,
        "value": value,
        "unit": unit,
        "formula": formula,
        "inputs": _inputs(facts),
        "note": "Historical calculation from reported figures. Not a forecast.",
    }


# ==========================================
# TOOLS EXPOSED TO AGENTS
# (type hints + Args docstrings drive the Ollama tool schema)
# ==========================================

def list_facts():
    """
    List every fact in the verified financial fact ledger with its source page.
    """
    ledger = get_active_ledger()

    if ledger is None:
        raise LedgerError("No fact ledger is loaded.")

    return {
        "status": "ok",
        "facts": [
            {
                "fact_id": f["fact_id"],
                "value": f["value"],
                "unit": f["unit"],
                "year": f["year"],
                "verification_status": f["verification_status"],
                "pdf_page": f["pdf_page"],
                "printed_page": f["printed_page"],
            }
            for f in ledger.facts.values()
        ],
        "warnings": ledger.warnings,
    }


def get_fact(metric: str, year: int):
    """
    Get one fact from the ledger, including its evidence text and page.

    Args:
        metric: Ledger metric name, for example net_sales, operating_income, net_income, total_current_assets, total_current_liabilities, total_liabilities, total_equity, total_assets, operating_cash_flow.
        year: Fiscal year, for example 2025.
    """
    fact = _require(metric, year)

    return {"status": "ok", "fact": fact}


def calc_revenue_growth(year: int):
    """
    Year-on-year net sales (revenue) growth in percent, from ledger facts.

    Args:
        year: Fiscal year to measure, for example 2025. The prior year is used automatically.
    """
    year = _year(year)
    current = _require("net_sales", year)
    previous = _require("net_sales", year - 1)
    _check_units([current, previous])

    value = revenue_growth(previous["value"], current["value"])

    return _ok("calc_revenue_growth", value, "percent",
               f"(net_sales_{year} - net_sales_{year-1}) / net_sales_{year-1} * 100",
               [previous, current])


def calc_operating_margin(year: int):
    """
    Operating margin in percent: operating income divided by net sales.

    Args:
        year: Fiscal year, for example 2025.
    """
    year = _year(year)
    income = _require("operating_income", year)
    sales = _require("net_sales", year)
    _check_units([income, sales])

    value = operating_margin(income["value"], sales["value"])

    return _ok("calc_operating_margin", value, "percent",
               f"operating_income_{year} / net_sales_{year} * 100",
               [income, sales])


def calc_net_margin(year: int):
    """
    Net profit margin in percent: net income divided by net sales.

    Args:
        year: Fiscal year, for example 2025.
    """
    year = _year(year)
    income = _require("net_income", year)
    sales = _require("net_sales", year)
    _check_units([income, sales])

    value = profit_margin(income["value"], sales["value"])

    return _ok("calc_net_margin", value, "percent",
               f"net_income_{year} / net_sales_{year} * 100",
               [income, sales])


def calc_growth(metric: str, year: int):
    """
    Year-on-year growth in percent for any ledger metric.

    Args:
        metric: Ledger metric name, for example net_income or operating_cash_flow.
        year: Fiscal year to measure. The prior year is used automatically.
    """
    year = _year(year)
    metric = _metric(metric)
    current = _require(metric, year)
    previous = _require(metric, year - 1)
    _check_units([current, previous])

    value = year_on_year_growth(previous["value"], current["value"])

    return _ok("calc_growth", value, "percent",
               f"({metric}_{year} - {metric}_{year-1}) / |{metric}_{year-1}| * 100",
               [previous, current])


def calc_cagr(metric: str, start_year: int, end_year: int):
    """
    Compound annual growth rate in percent for a ledger metric between two fiscal years.

    Args:
        metric: Ledger metric name, for example net_sales.
        start_year: First fiscal year.
        end_year: Last fiscal year.
    """
    start_year = _year(start_year)
    end_year = _year(end_year)
    metric = _metric(metric)

    if end_year <= start_year:
        raise LedgerError("end_year must be greater than start_year.")

    start = _require(metric, start_year)
    end = _require(metric, end_year)
    _check_units([start, end])

    value = calculate_cagr(start["value"], end["value"], end_year - start_year)

    return _ok("calc_cagr", value, "percent",
               f"({metric}_{end_year} / {metric}_{start_year}) ** (1/{end_year - start_year}) - 1",
               [start, end])


def calc_current_ratio(year: int):
    """
    Current ratio: total current assets divided by total current liabilities.

    Args:
        year: Fiscal year, for example 2025.
    """
    year = _year(year)
    assets = _require("total_current_assets", year)
    liabilities = _require("total_current_liabilities", year)
    _check_units([assets, liabilities])

    value = current_ratio(assets["value"], liabilities["value"])

    return _ok("calc_current_ratio", value, "ratio",
               f"total_current_assets_{year} / total_current_liabilities_{year}",
               [assets, liabilities])


def calc_liabilities_to_equity(year: int):
    """
    Total liabilities divided by total equity. This is NOT debt-to-equity.

    Args:
        year: Fiscal year, for example 2025.
    """
    year = _year(year)
    liabilities = _require("total_liabilities", year)
    equity = _require("total_equity", year)
    _check_units([liabilities, equity])

    value = debt_to_equity(liabilities["value"], equity["value"])

    return _ok("calc_liabilities_to_equity", value, "ratio",
               f"total_liabilities_{year} / total_equity_{year} "
               f"(liabilities-to-equity, not debt-to-equity)",
               [liabilities, equity])


def calc_debt_to_equity(year: int):
    """
    Debt-to-equity: total debt divided by total equity. Fails if total debt is not in the ledger.

    Args:
        year: Fiscal year, for example 2025.
    """
    year = _year(year)
    debt = _require("total_debt", year)
    equity = _require("total_equity", year)
    _check_units([debt, equity])

    value = debt_to_equity(debt["value"], equity["value"])

    return _ok("calc_debt_to_equity", value, "ratio",
               f"total_debt_{year} / total_equity_{year}",
               [debt, equity])


def calc_return_on_equity(year: int):
    """
    Return on equity in percent: net income divided by average of opening and closing total equity.

    Args:
        year: Fiscal year, for example 2025. Needs total equity for this year and the prior year.
    """
    year = _year(year)
    income = _require("net_income", year)
    closing = _require("total_equity", year)
    opening = _require("total_equity", year - 1)
    _check_units([income, closing, opening])

    average = (closing["value"] + opening["value"]) / 2

    value = return_on_equity(income["value"], average)

    return _ok("calc_return_on_equity", value, "percent",
               f"net_income_{year} / average(total_equity_{year}, "
               f"total_equity_{year-1}) * 100",
               [income, opening, closing])


def assess_ledger_source():
    """
    Assess the metadata of the document the fact ledger was extracted from.
    """
    ledger = get_active_ledger()

    if ledger is None:
        raise LedgerError("No fact ledger is loaded.")

    meta = ledger.source_meta

    result = assess_source_quality(
        meta.get("source_type", ""),
        meta.get("publication_year"),
        meta.get("has_url", False),
        meta.get("is_primary_source", False),
    )

    result["status"] = "ok"
    result["source"] = ledger.source_label

    return result


if __name__ == "__main__":

    built = build_ledger()

    print("FACT LEDGER\n")
    print(built.as_table())
    print("\nWARNINGS")

    for warning in built.warnings:
        print("-", warning)
