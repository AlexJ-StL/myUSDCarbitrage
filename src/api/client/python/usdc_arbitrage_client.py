"""Python client SDK for USDC Arbitrage API."""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import requests
from pydantic import BaseModel, Field
from requests.auth import AuthBase


class TokenAuth(AuthBase):
    """Token authentication for requests."""

    def __init__(self, token: str):
        self.token = token

    def __call__(self, r):
        r.headers["Authorization"] = f"Bearer {self.token}"
        return r


class BacktestRequest(BaseModel):
    """Backtest request model."""

    strategy_id: int = Field(..., description="Strategy ID to backtest")
    start_date: datetime = Field(..., description="Start date for backtest")
    end_date: datetime = Field(..., description="End date for backtest")
    exchanges: List[str] = Field(
        default=["coinbase", "kraken", "binance"],
        description="Exchanges to include in backtest",
    )
    symbols: List[str] = Field(
        default=["USDC/USD"],
        description="Symbols to include in backtest",
    )
    timeframe: str = Field(
        default="1h",
        description="Timeframe for backtest (e.g., 1m, 5m, 1h, 1d)",
    )
    initial_balance: float = Field(
        default=10000.0,
        description="Initial portfolio balance",
        gt=0,
    )
    position_sizing: str = Field(
        default="percent",
        description="Position sizing strategy (fixed, percent, kelly, volatility, risk_parity)",
    )
    position_size: float = Field(
        default=0.02,
        description="Position size (interpretation depends on position_sizing)",
        gt=0,
    )
    rebalance_frequency: str = Field(
        default="monthly",
        description="Portfolio rebalancing frequency (daily, weekly, monthly, quarterly, threshold)",
    )
    rebalance_threshold: float = Field(
        default=0.05,
        description="Threshold for threshold-based rebalancing",
        gt=0,
    )
    include_fees: bool = Field(
        default=True,
        description="Include exchange fees in backtest",
    )
    include_slippage: bool = Field(
        default=True,
        description="Include slippage in backtest",
    )
    strategy_params: Dict[str, Any] = Field(
        default={},
        description="Additional strategy parameters",
    )


class StrategyCreateRequest(BaseModel):
    """Strategy creation request model."""

    name: str = Field(..., description="Strategy name")
    description: str = Field(..., description="Strategy description")
    code: str = Field(..., description="Strategy code")
    parameters: Dict[str, Any] = Field(default={}, description="Strategy parameters")
    tags: Optional[List[str]] = Field(default=None, description="Strategy tags")
    strategy_type: str = Field(default="custom", description="Strategy type")
    is_active: bool = Field(default=True, description="Whether strategy is active")


class StrategyUpdateRequest(BaseModel):
    """Strategy update request model."""

    name: Optional[str] = Field(default=None, description="Strategy name")
    description: Optional[str] = Field(default=None, description="Strategy description")
    code: Optional[str] = Field(default=None, description="Strategy code")
    parameters: Optional[Dict[str, Any]] = Field(
        default=None, description="Strategy parameters"
    )
    tags: Optional[List[str]] = Field(default=None, description="Strategy tags")
    is_active: Optional[bool] = Field(
        default=None, description="Whether strategy is active"
    )
    commit_message: str = Field(
        default="", description="Commit message for version control"
    )


class USDCArbitrageClient:
    """Client for USDC Arbitrage API."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_version: str = "1.0",
        token: Optional[str] = None,
    ):
        """Initialize client."""
        self.base_url = base_url
        self.api_version = api_version
        self.token = token
        self.logger = logging.getLogger(__name__)

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Accept-Version": self.api_version,
        }

        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        return headers

    def _handle_response(self, response: requests.Response) -> Dict[str, Any]:
        """Handle API response."""
        try:
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            self.logger.error(f"HTTP error: {e}")
            error_detail = {}
            try:
                error_detail = response.json()
            except:
                error_detail = {"detail": response.text}

            raise Exception(f"API error: {response.status_code} - {error_detail}")
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request error: {e}")
            raise Exception(f"Request error: {e}")
        except json.JSONDecodeError:
            self.logger.error("Failed to decode JSON response")
            raise Exception("Failed to decode JSON response")

    def login(self, username: str, password: str) -> Dict[str, Any]:
        """Login to API and get access token."""
        url = f"{self.base_url}/auth/token"
        data = {
            "username": username,
            "password": password,
        }

        response = requests.post(url, json=data, headers=self._get_headers())
        result = self._handle_response(response)

        if "access_token" in result:
            self.token = result["access_token"]

        return result

    def refresh_token(self, refresh_token: str) -> Dict[str, Any]:
        """Refresh access token."""
        url = f"{self.base_url}/auth/refresh"
        data = {
            "refresh_token": refresh_token,
        }

        response = requests.post(url, json=data, headers=self._get_headers())
        result = self._handle_response(response)

        if "access_token" in result:
            self.token = result["access_token"]

        return result

    # Strategy endpoints
    def list_strategies(
        self,
        active_only: bool = False,
        tag: Optional[str] = None,
        search: Optional[str] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List strategies."""
        url = f"{self.base_url}/strategies/"
        params = {
            "active_only": active_only,
            "skip": skip,
            "limit": limit,
        }

        if tag:
            params["tag"] = tag

        if search:
            params["search"] = search

        response = requests.get(url, params=params, headers=self._get_headers())
        return self._handle_response(response)

    def get_strategy(
        self, strategy_id: int, include_code: bool = False
    ) -> Dict[str, Any]:
        """Get strategy by ID."""
        url = f"{self.base_url}/strategies/{strategy_id}"
        params = {
            "include_code": include_code,
        }

        response = requests.get(url, params=params, headers=self._get_headers())
        return self._handle_response(response)

    def create_strategy(
        self, request: Union[StrategyCreateRequest, Dict[str, Any]], author: str = "api"
    ) -> Dict[str, Any]:
        """Create a new strategy."""
        url = f"{self.base_url}/strategies/"
        params = {
            "author": author,
        }

        if isinstance(request, StrategyCreateRequest):
            data = request.model_dump()
        else:
            data = request

        response = requests.post(
            url, params=params, json=data, headers=self._get_headers()
        )
        return self._handle_response(response)

    def update_strategy(
        self,
        strategy_id: int,
        request: Union[StrategyUpdateRequest, Dict[str, Any]],
        author: str = "api",
    ) -> Dict[str, Any]:
        """Update an existing strategy."""
        url = f"{self.base_url}/strategies/{strategy_id}"
        params = {
            "author": author,
        }

        if isinstance(request, StrategyUpdateRequest):
            data = request.model_dump(exclude_unset=True)
        else:
            data = request

        response = requests.put(
            url, params=params, json=data, headers=self._get_headers()
        )
        return self._handle_response(response)

    def delete_strategy(self, strategy_id: int) -> Dict[str, Any]:
        """Delete a strategy."""
        url = f"{self.base_url}/strategies/{strategy_id}"

        response = requests.delete(url, headers=self._get_headers())
        return self._handle_response(response)

    # Backtest endpoints
    def run_backtest(
        self, request: Union[BacktestRequest, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Run a backtest."""
        url = f"{self.base_url}/backtest/"

        if isinstance(request, BacktestRequest):
            data = request.model_dump()
        else:
            data = request

        response = requests.post(url, json=data, headers=self._get_headers())
        return self._handle_response(response)

    def get_backtest(self, backtest_id: int) -> Dict[str, Any]:
        """Get backtest by ID."""
        url = f"{self.base_url}/backtest/{backtest_id}"

        response = requests.get(url, headers=self._get_headers())
        return self._handle_response(response)

    def list_backtests(
        self,
        strategy_id: Optional[int] = None,
        status: Optional[str] = None,
        limit: int = 10,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List backtests."""
        url = f"{self.base_url}/backtest/"
        params = {
            "limit": limit,
            "offset": offset,
        }

        if strategy_id is not None:
            params["strategy_id"] = strategy_id

        if status is not None:
            params["status"] = status

        response = requests.get(url, params=params, headers=self._get_headers())
        return self._handle_response(response)

    # Results endpoints
    def list_results(
        self,
        strategy_id: Optional[int] = None,
        status: Optional[str] = None,
        skip: int = 0,
        limit: int = 100,
        sort_by: str = "created_at",
        sort_order: str = "desc",
    ) -> List[Dict[str, Any]]:
        """List backtest results."""
        url = f"{self.base_url}/results/"
        params = {
            "skip": skip,
            "limit": limit,
            "sort_by": sort_by,
            "sort_order": sort_order,
        }

        if strategy_id is not None:
            params["strategy_id"] = strategy_id

        if status is not None:
            params["status"] = status

        response = requests.get(url, params=params, headers=self._get_headers())
        return self._handle_response(response)

    def get_result(self, result_id: int) -> Dict[str, Any]:
        """Get backtest result by ID."""
        url = f"{self.base_url}/results/{result_id}"

        response = requests.get(url, headers=self._get_headers())
        return self._handle_response(response)

    def get_result_transactions(
        self, result_id: int, skip: int = 0, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get transactions for a backtest result."""
        url = f"{self.base_url}/results/{result_id}/transactions"
        params = {
            "skip": skip,
            "limit": limit,
        }

        response = requests.get(url, params=params, headers=self._get_headers())
        return self._handle_response(response)

    def get_result_positions(
        self, result_id: int, skip: int = 0, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get position snapshots for a backtest result."""
        url = f"{self.base_url}/results/{result_id}/positions"
        params = {
            "skip": skip,
            "limit": limit,
        }

        response = requests.get(url, params=params, headers=self._get_headers())
        return self._handle_response(response)

    def get_result_equity_curve(self, result_id: int) -> List[Dict[str, Any]]:
        """Get equity curve for a backtest result."""
        url = f"{self.base_url}/results/{result_id}/equity_curve"

        response = requests.get(url, headers=self._get_headers())
        return self._handle_response(response)

    # Data export endpoints
    def export_data(
        self,
        data_type: str,
        format: str = "json",
        filters: Optional[Dict[str, Any]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        exchanges: Optional[List[str]] = None,
        symbols: Optional[List[str]] = None,
        strategy_id: Optional[int] = None,
        backtest_id: Optional[int] = None,
        limit: Optional[int] = 10000,
    ) -> bytes:
        """Export data in the requested format."""
        url = f"{self.base_url}/export/data"
        data = {
            "data_type": data_type,
            "format": format,
        }

        if filters:
            data["filters"] = filters

        if start_date:
            data["start_date"] = start_date.isoformat()

        if end_date:
            data["end_date"] = end_date.isoformat()

        if exchanges:
            data["exchanges"] = exchanges

        if symbols:
            data["symbols"] = symbols

        if strategy_id:
            data["strategy_id"] = strategy_id

        if backtest_id:
            data["backtest_id"] = backtest_id

        if limit:
            data["limit"] = limit

        response = requests.post(url, json=data, headers=self._get_headers())

        try:
            response.raise_for_status()
            return response.content
        except requests.exceptions.HTTPError as e:
            self.logger.error(f"HTTP error: {e}")
            error_detail = {}
            try:
                error_detail = response.json()
            except:
                error_detail = {"detail": response.text}

            raise Exception(f"API error: {response.status_code} - {error_detail}")
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request error: {e}")
            raise Exception(f"Request error: {e}")

    def export_backtest_results(
        self,
        backtest_id: int,
        format: str = "json",
        include_transactions: bool = False,
        include_positions: bool = False,
    ) -> bytes:
        """Export backtest results in the requested format."""
        url = f"{self.base_url}/export/backtest/{backtest_id}"
        params = {
            "format": format,
            "include_transactions": include_transactions,
            "include_positions": include_positions,
        }

        response = requests.get(url, params=params, headers=self._get_headers())

        try:
            response.raise_for_status()
            return response.content
        except requests.exceptions.HTTPError as e:
            self.logger.error(f"HTTP error: {e}")
            error_detail = {}
            try:
                error_detail = response.json()
            except:
                error_detail = {"detail": response.text}

            raise Exception(f"API error: {response.status_code} - {error_detail}")
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request error: {e}")
            raise Exception(f"Request error: {e}")

    def export_strategy(
        self,
        strategy_id: int,
        format: str = "json",
        include_code: bool = False,
        include_backtests: bool = False,
    ) -> bytes:
        """Export strategy details in the requested format."""
        url = f"{self.base_url}/export/strategy/{strategy_id}"
        params = {
            "format": format,
            "include_code": include_code,
            "include_backtests": include_backtests,
        }

        response = requests.get(url, params=params, headers=self._get_headers())

        try:
            response.raise_for_status()
            return response.content
        except requests.exceptions.HTTPError as e:
            self.logger.error(f"HTTP error: {e}")
            error_detail = {}
            try:
                error_detail = response.json()
            except:
                error_detail = {"detail": response.text}

            raise Exception(f"API error: {response.status_code} - {error_detail}")
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request error: {e}")
            raise Exception(f"Request error: {e}")
