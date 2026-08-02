from pydantic import BaseModel, Field
from typing import List

class SegmentRevenue(BaseModel):
    segment_name : str
    revenue_millions : float

class FiscalYearFinancials(BaseModel):
    fiscal_yr : str
    total_revenue_millions: float
    segment_revenues: List[SegmentRevenue]
    gross_margin_percent: float | None=None

