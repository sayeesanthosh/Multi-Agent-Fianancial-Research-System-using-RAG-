
from financial_tools import (
    revenue_growth,
    current_ratio,
    operating_margin,
    profit_margin
)

# Apple FY2025 and FY2024 reported figures
revenue_2025 = 416161
revenue_2024 = 391035

net_income_2025 = 112010
operating_income_2025 = 133050

current_assets = 147957
current_liabilities = 165631

print("APPLE FY2025 CALCULATIONS")
print("=========================")

print(
    "Revenue Growth:",
    revenue_growth(revenue_2024, revenue_2025),
    "%"
)

print(
    "Current Ratio:",
    current_ratio(str(current_assets), str(current_liabilities))
)

print(
    "Operating Margin:",
    operating_margin(operating_income_2025, revenue_2025),
    "%"
)

print(
    "Net Profit Margin:",
    profit_margin(net_income_2025, revenue_2025),
    "%"
)
