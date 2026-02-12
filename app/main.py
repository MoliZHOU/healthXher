from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import auth, predict
from app.core.config import settings
from app.db.models import Base, User
from app.db.session import engine, SessionLocal
from app.core.security import get_password_hash

# Create tables
Base.metadata.create_all(bind=engine)

# Seed default user
def init_db():
    import os
    db = SessionLocal()
    try:
        username = "dadoudouoo"
        default_user = db.query(User).filter(User.username == username).first()
        admin_pw = os.environ.get("DEFAULT_ADMIN_PASSWORD")
        
        if not default_user and admin_pw:
            hashed_pw = get_password_hash(admin_pw)
            user = User(username=username, hashed_password=hashed_pw)
            db.add(user)
            db.commit()
            print(f"[Database] Default user '{username}' created from environment.")
        elif not default_user:
            print("[Database] Skip seeding: DEFAULT_ADMIN_PASSWORD not set.")
    finally:
        db.close()

init_db()

app = FastAPI(title=settings.PROJECT_NAME)

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix=f"{settings.API_V1_STR}/auth", tags=["auth"])
app.include_router(predict.router, prefix=f"{settings.API_V1_STR}/predict", tags=["predict"])

@app.get("/")
def read_root():
    return {"status": "online", "system": "healthXher Local-First RA Prediction"}
