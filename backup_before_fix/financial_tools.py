import math
# ==========================================
# FINANCIAL CALCULATION ENGINE
# ==========================================


def parse_financial_number(value):

    if isinstance(value, bool):
        raise ValueError("A boolean is not a financial number.")

    if isinstance(value, (int, float)):
        number = float(value)

    else:
        text = str(value).lower().strip()

        text = text.replace(",", "")
        text = text.replace("₹", "")
        text = text.replace("crores", "")
        text = text.replace("crore", "")
        text = text.replace(" ", "")

        try:
            number = float(text)

        except ValueError:
            raise ValueError(
                f"Invalid financial number: {value!r}"
            )

    if not math.isfinite(number):
        raise ValueError(
            f"Financial number must be finite: {value!r}"
        )

    return number


def revenue_growth(previous_revenue, current_revenue):

    previous_revenue = parse_financial_number(previous_revenue)
    current_revenue = parse_financial_number(current_revenue)

    if previous_revenue == 0:
        raise ValueError("Previous revenue cannot be zero.")

    growth = (
        (current_revenue - previous_revenue)
        / abs(previous_revenue)
    ) * 100

    return round(growth, 2)


def calculate_cagr(start_value, end_value, years):

    start_value = parse_financial_number(start_value)
    end_value = parse_financial_number(end_value)
    years = parse_financial_number(years)

    if start_value <= 0 or end_value <= 0:
        raise ValueError("CAGR requires positive values.")

    if years <= 0:
        raise ValueError("Years must be greater than zero.")

    cagr = (
        (end_value / start_value) ** (1 / years) - 1
    ) * 100

    return round(cagr, 2)



def profit_margin(net_profit, revenue):

    net_profit = parse_financial_number(net_profit)
    revenue = parse_financial_number(revenue)

    if revenue == 0:
        raise ValueError("Revenue cannot be zero.")

    margin = (net_profit / revenue) * 100

    return round(margin, 2)

def operating_margin(operating_profit, revenue):

    operating_profit = parse_financial_number(operating_profit)
    revenue = parse_financial_number(revenue)

    if revenue == 0:
        raise ValueError("Revenue cannot be zero.")

    margin = (operating_profit / revenue) * 100

    return round(margin, 2)



def debt_to_equity(total_debt, total_equity):

    total_debt = parse_financial_number(total_debt)
    total_equity = parse_financial_number(total_equity)

    if total_equity <= 0:
        raise ValueError("Equity must be positive.")

    ratio = total_debt / total_equity

    return round(ratio, 2)


def current_ratio(current_assets, current_liabilities):

    current_assets = parse_financial_number(current_assets)
    current_liabilities = parse_financial_number(current_liabilities)

    if current_liabilities <= 0:
        raise ValueError("Current liabilities must be positive.")

    ratio = current_assets / current_liabilities

    return round(ratio, 2)

def return_on_equity(net_income, average_equity):

    net_income = parse_financial_number(net_income)
    average_equity = parse_financial_number(average_equity)

    if average_equity <= 0:
        raise ValueError("Average equity must be positive.")

    roe = (net_income / average_equity) * 100

    return round(roe, 2)


def year_on_year_growth(previous_value, current_value):

    previous_value = parse_financial_number(previous_value)
    current_value = parse_financial_number(current_value)

    if previous_value == 0:
        raise ValueError("Previous value cannot be zero.")

    growth = (
        (current_value - previous_value)
        / abs(previous_value)
    ) * 100

    return round(growth, 2)

# ==========================================
# EXAMPLE EXECUTION
# ==========================================

if __name__ == "__main__":

    print("FINANCIAL CALCULATION ENGINE")
    print("============================")

    # Revenue figures in crore
    revenue_2023 = 100
    revenue_2024 = 120
    revenue_2025 = 150

    growth_2024 = revenue_growth(
        revenue_2023,
        revenue_2024
    )

    growth_2025 = revenue_growth(
        revenue_2024,
        revenue_2025
    )

    cagr = calculate_cagr(
        revenue_2023,
        revenue_2025,
        2
    )

    print("\nRevenue Analysis")

    print("2024 Growth:", growth_2024, "%")
    print("2025 Growth:", growth_2025, "%")
    print("Revenue CAGR:", cagr, "%")

    # Profitability
    net_profit = 20
    revenue = 150

    print("\nProfitability")

    print(
        "Net Profit Margin:",
        profit_margin(net_profit, revenue),
        "%"
    )

    # Financial ratios
    print("\nFinancial Ratios")

    print(
        "Debt-to-Equity:",
        debt_to_equity(200, 100)
    )

    print(
        "Current Ratio:",
        current_ratio(150, 75)
    )

    print(
        "Return on Equity:",
        return_on_equity(20, 90),
        "%"
    )

    print("\nCalculation completed.")
