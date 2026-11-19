from datetime import datetime


def _to_year(value):
    if isinstance(value, bool):
        raise ValueError("publication_year must be a year, not a boolean.")

    try:
        year = int(float(str(value).strip()))
    except ValueError:
        raise ValueError(
            f"publication_year must be a 4-digit year, got {value!r}."
        )

    current_year = datetime.now().year

    if year < 1900 or year > current_year + 1:
        raise ValueError(
            f"publication_year {year} is outside 1900-{current_year + 1}."
        )

    return year


def _to_bool(value, name):
    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)

    text = str(value).strip().lower()

    if text in ("true", "yes", "y", "1"):
        return True

    if text in ("false", "no", "n", "0", "", "none", "null"):
        return False

    raise ValueError(f"{name} must be true or false, got {value!r}.")


def assess_source_quality(
    source_type: str,
    publication_year: int,
    has_url: bool,
    is_primary_source: bool
):
    """
    Assess source metadata only. This does NOT verify the source's claims.

    Args:
        source_type: One of regulatory_filing, audited_annual_report, earnings_release, company_presentation, earnings_transcript, news_article, analyst_report, research_article.
        publication_year: Four-digit publication year, for example 2025.
        has_url: True if the source has a URL, otherwise false.
        is_primary_source: True if the source is a primary source.
    """

    source_type = str(source_type).lower().strip()

    publication_year = _to_year(publication_year)
    has_url = _to_bool(has_url, "has_url")
    is_primary_source = _to_bool(is_primary_source, "is_primary_source")

    current_year = datetime.now().year
    source_age = current_year - publication_year

    primary_types = [
        "regulatory_filing",
        "audited_annual_report",
        "earnings_release",
        "company_presentation",
        "earnings_transcript"
    ]

    secondary_types = [
        "news_article",
        "analyst_report",
        "research_article"
    ]

    if source_type in primary_types:
        source_category = "Primary source"

    elif source_type in secondary_types:
        source_category = "Secondary source"

    else:
        source_category = "Unknown source type"

    if is_primary_source and source_category == "Secondary source":
        source_category = "Inconsistent metadata: check classification"

    if source_age < 0:
        freshness = "Invalid publication year"

    elif source_age <= 1:
        freshness = "Recent"

    elif source_age <= 3:
        freshness = "Moderately recent"

    else:
        freshness = "Potentially outdated"

    if source_category == "Primary source" and has_url and source_age <= 3:
        quality = "Strong source metadata"

    elif source_category == "Primary source":
        quality = "Potentially useful primary source"

    elif source_category == "Secondary source" and has_url:
        quality = "Traceable secondary source"

    elif source_category == "Secondary source":
        quality = "Secondary source without URL"

    else:
        quality = "Insufficient metadata"

    return {
        "source_category": source_category,
        "freshness": freshness,
        "source_age_years": source_age,
        "url_provided": has_url,
        "quality_assessment": quality,
        "warning": (
            "Metadata assessment only. "
            "The source content and claims have not been verified."
        )
    }