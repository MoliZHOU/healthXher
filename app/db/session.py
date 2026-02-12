from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from app.core.config import settings

# Note: sqlite+pysqlcipher requires pysqlcipher3 or sqlcipher3-binary
engine = create_engine(
    settings.DATABASE_URL.replace(":passphrase", f":{settings.DB_PASSPHRASE}"),
    connect_args={"check_same_thread": False},
)

@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    if "pysqlcipher" in settings.DATABASE_URL:
        cursor = dbapi_connection.cursor()
        cursor.execute(f"PRAGMA key = '{settings.DB_PASSPHRASE}';")
        cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
