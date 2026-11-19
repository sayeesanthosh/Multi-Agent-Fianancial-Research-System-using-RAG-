from financial_tools import operating_margin
from financial_tools import (
    revenue_growth,
    calculate_cagr,
    profit_margin,
    debt_to_equity,
    current_ratio
)

print("FINANCIAL TOOL TESTS")

print(
    "Revenue Growth:",
    revenue_growth("100 crore", "150crore"),
    "%"
)

print(
    "CAGR:",
    calculate_cagr(100, 150, 2),
    "%"
)

print(
    "Profit Margin:",
    profit_margin("20 crore", "100 crore"),
    "%"
)

print(
    "Debt-to-Equity:",
    debt_to_equity("200 crore", "100 crore")
)

print(
    "Current Ratio:",
    current_ratio(150, 75)
)
print("Operating Margin:", operating_margin("25 crore", "100 crore"), "%")
