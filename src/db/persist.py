from src.db.models import FiscalYear, SegmentRevenue
from src.extraction import extraction_chain
from src.pydantic_schemas import FiscalYearFinancials

from src.db.connections import get_session

# data passed into the function is an object of pydantic class FiscalYearFInancials

#look into extractions.py for reference. 

def save_financial(data: FiscalYearFinancials) -> int:
    fiscal_year_row = FiscalYear(
        year = int(data.fiscal_yr),
        total_revenue_millions = data.total_revenue_millions,
        gross_margin_percentage= data.gross_margin_percent,
    )

    for segment in data.segment_revenues:
        segment_row = SegmentRevenue(
            segment = segment.segment_name,
            amount = segment.revenue_millions,
        )
        fiscal_year_row.revenues.append(segment_row)

    #get_session func in connetions.py
    session = get_session()

    session.add(fiscal_year_row)

    #Commit and close
    session.commit()
    fiscal_year_id = fiscal_year_row.id   # read while still attached, before closing
    session.close()

    return fiscal_year_id


if __name__ == "__main__":
    data = extraction_chain.invoke(
        "NVIDIA fiscal year 2025 total revenue and segment breakdown"
    )
    print("Extracted from LLM:")
    print(data)

    fiscal_year_id = save_financial(data)
    print(f"\nSaved to Postgres with id={fiscal_year_id}")

    # Read it back through a fresh session, proving the round trip actually
    # worked, not just that commit() didn't raise.
    session = get_session()
    saved = session.get(FiscalYear, fiscal_year_id)

    print(f"\nRead back from database: year={saved.year}, "
          f"total_revenue_millions={saved.total_revenue_millions}")
    for segment in saved.revenues:
        print(f"  {segment.segment}: {segment.amount}")

    session.close()
