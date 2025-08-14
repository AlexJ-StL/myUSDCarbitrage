"""Strategy manager for handling strategy operations."""

from typing import Dict, List, Optional
from sqlalchemy.orm import Session

from ..models import Strategy


class StrategyManager:
    """Manager class for strategy operations."""

    def __init__(self, db: Session):
        self.db = db

    def get_strategy_by_id(self, strategy_id: int) -> Optional[Strategy]:
        """Get strategy by ID."""
        return self.db.query(Strategy).filter(Strategy.id == strategy_id).first()

    def get_strategies(self, skip: int = 0, limit: int = 100) -> List[Strategy]:
        """Get list of strategies."""
        return self.db.query(Strategy).offset(skip).limit(limit).all()

    def create_strategy(self, strategy_data: Dict) -> Strategy:
        """Create a new strategy."""
        strategy = Strategy(**strategy_data)
        self.db.add(strategy)
        self.db.commit()
        self.db.refresh(strategy)
        return strategy

    def update_strategy(
        self, strategy_id: int, strategy_data: Dict
    ) -> Optional[Strategy]:
        """Update an existing strategy."""
        strategy = self.get_strategy_by_id(strategy_id)
        if not strategy:
            return None

        for key, value in strategy_data.items():
            if hasattr(strategy, key):
                setattr(strategy, key, value)

        self.db.commit()
        self.db.refresh(strategy)
        return strategy

    def delete_strategy(self, strategy_id: int) -> bool:
        """Delete a strategy."""
        strategy = self.get_strategy_by_id(strategy_id)
        if not strategy:
            return False

        self.db.delete(strategy)
        self.db.commit()
        return True


def get_strategy_by_id(db: Session, strategy_id: int) -> Optional[Strategy]:
    """Helper function to get strategy by ID."""
    manager = StrategyManager(db)
    return manager.get_strategy_by_id(strategy_id)
