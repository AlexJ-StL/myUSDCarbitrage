
from fastapi import FastAPI
from .routers import data, strategies, backtest, results
from .database import engine, Base

Base.metadata.create_all(bind=engine)

app = FastAPI()

app.include_router(data.router)
app.include_router(strategies.router)
app.include_router(backtest.router)
app.include_router(results.router)

@app.get("/")
def read_root():
    return {"message": "Welcome to the myUSDCarbitrage API"}
