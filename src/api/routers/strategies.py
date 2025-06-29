from fastapi import APIRouter, Depends, HTTPException, UploadFile
from ..dependencies import get_admin_user
import importlib
import inspect
import shutil
from pathlib import Path

router = APIRouter()

# Strategy loader functions
STRATEGY_DIR = Path("src/strategies")
strategies = {}

def load_strategies():
    """Load all strategies in strategies directory"""
    if not STRATEGY_DIR.exists():
        STRATEGY_DIR.mkdir()
    for file in STRATEGY_DIR.glob("*.py"):
        strategy_name = file.stem
        spec = importlib.util.spec_from_file_location(
            strategy_name,
            str(file)
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Find strategy functions
        for name, obj in inspect.getmembers(module):
            if inspect.isfunction(obj) and name.startswith("strategy_"):
                strategies[name.split("_", 1)[1]] = obj

@router.post("/register", summary="Upload a new strategy")
async def register_strategy(file: UploadFile, user=Depends(get_admin_user)):
    try:
        # Save to strategies directory
        file_path = STRATEGY_DIR / file.filename
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Reload strategies
        load_strategies()
        return {"status": "registered", "strategies": list(strategies.keys())}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/", summary="List available strategies")
async def list_strategies():
    return {"strategies": list(strategies.keys())}

def get_strategy_function(name: str):
    if not strategies:
        load_strategies()
    if name not in strategies:
        raise ValueError(f"Unknown strategy: {name}")
    return strategies[name]
