import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from src.db.models import Base

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)


def get_session() -> Session:
    return Session(engine)


if __name__ == "__main__":
    Base.metadata.create_all(engine)
    print("Tables created (or already existed).")
