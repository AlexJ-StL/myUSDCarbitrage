"""WebSocket router for real-time backtest monitoring."""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .. import models
from ..database import get_db
from ..security import get_current_user_ws

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()


# Connection manager for WebSockets
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
        self.backtest_subscriptions: Dict[str, List[int]] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        if client_id not in self.active_connections:
            self.active_connections[client_id] = []
        self.active_connections[client_id].append(websocket)
        self.backtest_subscriptions[client_id] = []
        logger.info(f"Client {client_id} connected")

    def disconnect(self, websocket: WebSocket, client_id: str):
        if client_id in self.active_connections:
            if websocket in self.active_connections[client_id]:
                self.active_connections[client_id].remove(websocket)
            if not self.active_connections[client_id]:
                del self.active_connections[client_id]
                if client_id in self.backtest_subscriptions:
                    del self.backtest_subscriptions[client_id]
        logger.info(f"Client {client_id} disconnected")

    async def send_personal_message(self, message: Dict[str, Any], client_id: str):
        if client_id in self.active_connections:
            for connection in self.active_connections[client_id]:
                await connection.send_json(message)

    async def broadcast(self, message: Dict[str, Any]):
        for client_id in self.active_connections:
            await self.send_personal_message(message, client_id)

    def subscribe_to_backtest(self, client_id: str, backtest_id: int):
        if client_id in self.backtest_subscriptions:
            if backtest_id not in self.backtest_subscriptions[client_id]:
                self.backtest_subscriptions[client_id].append(backtest_id)
                logger.info(f"Client {client_id} subscribed to backtest {backtest_id}")
                return True
        return False

    def unsubscribe_from_backtest(self, client_id: str, backtest_id: int):
        if client_id in self.backtest_subscriptions:
            if backtest_id in self.backtest_subscriptions[client_id]:
                self.backtest_subscriptions[client_id].remove(backtest_id)
                logger.info(
                    f"Client {client_id} unsubscribed from backtest {backtest_id}"
                )
                return True
        return False

    def get_subscriptions(self, client_id: str) -> List[int]:
        return self.backtest_subscriptions.get(client_id, [])


# Create connection manager instance
manager = ConnectionManager()


# WebSocket endpoint for backtest monitoring
@router.websocket("/ws/backtest/{client_id}")
async def websocket_backtest(
    websocket: WebSocket, client_id: str, db: Session = Depends(get_db)
):
    user = None
    try:
        # Authenticate user
        user = await get_current_user_ws(websocket, db)
        if not user:
            await websocket.close(code=1008, reason="Authentication failed")
            return

        # Accept connection
        await manager.connect(websocket, client_id)

        # Send welcome message
        await websocket.send_json({
            "type": "connection_established",
            "message": "Connected to backtest monitoring service",
            "client_id": client_id,
            "timestamp": datetime.now().isoformat(),
        })

        # Handle messages
        while True:
            try:
                # Receive message
                data = await websocket.receive_text()
                message = json.loads(data)

                # Process message based on type
                if message.get("type") == "subscribe":
                    backtest_id = message.get("backtest_id")
                    if backtest_id:
                        # Check if backtest exists and user has access
                        backtest = (
                            db.query(models.BacktestResult)
                            .filter(models.BacktestResult.id == backtest_id)
                            .first()
                        )

                        if not backtest:
                            await websocket.send_json({
                                "type": "error",
                                "message": f"Backtest {backtest_id} not found",
                                "timestamp": datetime.now().isoformat(),
                            })
                            continue

                        # Subscribe to backtest updates
                        success = manager.subscribe_to_backtest(client_id, backtest_id)

                        if success:
                            # Send initial backtest state
                            await websocket.send_json({
                                "type": "backtest_state",
                                "backtest_id": backtest_id,
                                "status": backtest.status,
                                "progress": get_backtest_progress(backtest),
                                "metrics": backtest.metrics,
                                "timestamp": datetime.now().isoformat(),
                            })

                            # Send confirmation
                            await websocket.send_json({
                                "type": "subscription_success",
                                "backtest_id": backtest_id,
                                "message": f"Subscribed to backtest {backtest_id}",
                                "timestamp": datetime.now().isoformat(),
                            })
                        else:
                            await websocket.send_json({
                                "type": "error",
                                "message": f"Already subscribed to backtest {backtest_id}",
                                "timestamp": datetime.now().isoformat(),
                            })

                elif message.get("type") == "unsubscribe":
                    backtest_id = message.get("backtest_id")
                    if backtest_id:
                        success = manager.unsubscribe_from_backtest(
                            client_id, backtest_id
                        )

                        if success:
                            await websocket.send_json({
                                "type": "unsubscription_success",
                                "backtest_id": backtest_id,
                                "message": f"Unsubscribed from backtest {backtest_id}",
                                "timestamp": datetime.now().isoformat(),
                            })
                        else:
                            await websocket.send_json({
                                "type": "error",
                                "message": f"Not subscribed to backtest {backtest_id}",
                                "timestamp": datetime.now().isoformat(),
                            })

                elif message.get("type") == "get_subscriptions":
                    subscriptions = manager.get_subscriptions(client_id)
                    await websocket.send_json({
                        "type": "subscriptions",
                        "subscriptions": subscriptions,
                        "timestamp": datetime.now().isoformat(),
                    })

                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Unknown message type",
                        "timestamp": datetime.now().isoformat(),
                    })

            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON format",
                    "timestamp": datetime.now().isoformat(),
                })

    except WebSocketDisconnect:
        if user:
            manager.disconnect(websocket, client_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if user:
            manager.disconnect(websocket, client_id)


def get_backtest_progress(backtest: models.BacktestResult) -> float:
    """Calculate backtest progress percentage."""
    if backtest.status == "completed":
        return 100.0
    elif backtest.status == "failed":
        return 0.0
    elif backtest.status == "pending":
        return 0.0

    # For running backtests, calculate progress based on date range
    if backtest.results and "progress" in backtest.results:
        return backtest.results["progress"]

    # If no progress info, estimate based on start/end dates
    if backtest.start_date and backtest.end_date and backtest.created_at:
        total_days = (backtest.end_date - backtest.start_date).days
        if total_days <= 0:
            return 0.0

        elapsed_time = (datetime.now() - backtest.created_at).total_seconds()
        estimated_total_time = total_days * 10  # Rough estimate: 10 seconds per day

        progress = min(95.0, (elapsed_time / estimated_total_time) * 100)
        return max(0.0, progress)

    return 0.0


# Function to send backtest updates to subscribed clients
async def send_backtest_update(backtest_id: int, update_data: Dict[str, Any]):
    """Send backtest update to all subscribed clients."""
    for client_id, subscriptions in manager.backtest_subscriptions.items():
        if backtest_id in subscriptions:
            await manager.send_personal_message(
                {
                    "type": "backtest_update",
                    "backtest_id": backtest_id,
                    **update_data,
                    "timestamp": datetime.now().isoformat(),
                },
                client_id,
            )


# Background task to monitor backtests and send updates
async def backtest_monitor_task(db: Session):
    """Background task to monitor backtests and send updates."""
    while True:
        try:
            # Check for active subscriptions
            active_backtests = set()
            for subscriptions in manager.backtest_subscriptions.values():
                active_backtests.update(subscriptions)

            if active_backtests:
                # Query backtest status for subscribed backtests
                for backtest_id in active_backtests:
                    backtest = (
                        db.query(models.BacktestResult)
                        .filter(models.BacktestResult.id == backtest_id)
                        .first()
                    )

                    if backtest:
                        # Send update to subscribed clients
                        await send_backtest_update(
                            backtest_id,
                            {
                                "status": backtest.status,
                                "progress": get_backtest_progress(backtest),
                                "metrics": backtest.metrics,
                            },
                        )

            # Sleep before next update
            await asyncio.sleep(2)

        except Exception as e:
            logger.error(f"Error in backtest monitor task: {e}")
            await asyncio.sleep(5)  # Sleep longer on error


# Start background task on application startup
@router.on_event("startup")
async def startup_event():
    asyncio.create_task(backtest_monitor_task(next(get_db())))
