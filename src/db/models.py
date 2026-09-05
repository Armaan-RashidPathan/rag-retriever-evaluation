from sqlalchemy import Column , Integer , String , ForeignKey, Float, Numeric
from sqlalchemy.orm import DeclarativeBase , Session , relationship

class Base(DeclarativeBase):
    pass

class FiscalYear(Base):
    __tablename__="fiscal_year"
    id = Column(Integer , primary_key=True)
    year = Column(Integer)
    total_revenue_millions = Column(Numeric)
    gross_margin_percentage = Column(Numeric)
    revenues = relationship("SegmentRevenue", back_populates="fiscal_year")

class SegmentRevenue(Base):
    __tablename__ = "segment_revenue"
    id = Column(Integer, primary_key=True)
    segment = Column(String, nullable=False)   # "Data Center", "Gaming", etc.
    amount = Column(Numeric , nullable=False)
    fiscal_year_id = Column(Integer, ForeignKey("fiscal_year.id"), nullable=False)
    fiscal_year = relationship("FiscalYear", back_populates="revenues")