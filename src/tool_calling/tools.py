from datetime import date

from langchain_core.tools import tool

REVENUE_DATA = {
    "2024": {
        "data center": 47525,
        "gaming": 10447,
        "professional visualization": 1553,
        "automotive": 1091,
    },
    "2025": {
        "data center": 115186,
        "gaming": 11350,
        "compute & networking": 116193,
        "graphics": 14304,
    },
}


@tool
def calculate_growth_percent(old_value: float, new_value: float) -> float:
    """
    Use this tool to compute the percentage growth between two numeric values.

    The formula is:
        ((new_value - old_value) / old_value) * 100

    Call this whenever a user asks for growth rate, percentage increase,
    or percentage change between two numbers.
    """
    if old_value == 0:
        raise ValueError("old_value cannot be zero.")
    return round(((new_value - old_value) / old_value) * 100, 2)


@tool
def get_segment_revenue(segment_name: str, fiscal_year: str) -> str:
    """
    Use this tool to retrieve the revenue for a business segment in a given fiscal year.

    Inputs:
    - segment_name: business segment name (e.g. 'cloud', 'gaming')
    - fiscal_year: year such as '2023' or '2024'

    Returns the stored revenue if available.
    """
    year_data = REVENUE_DATA.get(fiscal_year)

    if year_data is None:
        return f"No revenue data available for fiscal year {fiscal_year}."

    revenue = year_data.get(segment_name.lower())

    if revenue is None:
        return (
            f"No revenue found for segment '{segment_name}' "
            f"in fiscal year {fiscal_year}."
        )

    return (
        f"{segment_name.title()} revenue in FY{fiscal_year}: {revenue}"
    )


@tool
def get_current_date() -> str:
    """
    Use this tool when the user asks for today's date or the current date.
    """
    return date.today().isoformat()


TOOLS = [
    calculate_growth_percent,
    get_segment_revenue,
    get_current_date,
]