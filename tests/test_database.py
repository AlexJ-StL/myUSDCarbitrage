import pytest
from unittest.mock import patch, MagicMock
from api.database import DBConnector, Database
import pandas as pd
from sqlalchemy import text

@pytest.fixture
def db_connector():
    with patch('src.api.database.create_engine') as mock_create_engine:
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        yield DBConnector('test_connection_string')

def test_get_ohlcv_data(db_connector):
    mock_connection = MagicMock()
    db_connector.engine.connect.return_value.__enter__.return_value = mock_connection
    
    mock_result = MagicMock()
    mock_connection.execute.return_value = mock_result
    mock_result.fetchall.return_value = [
        (pd.to_datetime('2023-01-01 00:00:00'), 1.0, 1.1, 0.9, 1.05, 100),
        (pd.to_datetime('2023-01-01 01:00:00'), 1.05, 1.15, 1.0, 1.1, 150)
    ]
    mock_result.keys.return_value = ['timestamp', 'open_price', 'high_price', 'low_price', 'close_price', 'volume']
    
    df = db_connector.get_ohlcv_data('test_exchange', 'test_symbol', '1h')
    
    assert not df.empty
    assert len(df) == 2
    assert list(df.columns) == ['timestamp', 'open_price', 'high_price', 'low_price', 'close_price', 'volume']
    db_connector.engine.connect.assert_called_once()
    mock_connection.execute.assert_called_once()

def test_get_user_roles(db_connector):
    mock_connection = MagicMock()
    db_connector.engine.connect.return_value.__enter__.return_value = mock_connection
    
    mock_result = MagicMock()
    mock_connection.execute.return_value = mock_result
    mock_result.fetchall.return_value = [{'role': 'admin'}, {'role': 'user'}]
    
    roles = db_connector.get_user_roles('test_user')
    
    assert roles == ['admin', 'user']
    db_connector.engine.connect.assert_called_once()
    mock_connection.execute.assert_called_once()

@patch('src.api.database.SessionLocal')
def test_insert_data(MockSessionLocal):
    mock_session = MagicMock()
    MockSessionLocal.return_value = mock_session
    
    db = Database()
    
    data_to_insert = [
        [1672531200000, 1.0, 1.1, 0.9, 1.05, 100],
        [1672534800000, 1.05, 1.15, 1.0, 1.1, 150]
    ]
    
    db.insert_data('test_exchange', 'test_symbol', '1h', data_to_insert)
    
    assert mock_session.add.call_count == 2
    mock_session.commit.assert_called_once()

@patch('src.api.database.SessionLocal')
def test_close_db_session(MockSessionLocal):
    mock_session = MagicMock()
    MockSessionLocal.return_value = mock_session
    
    db = Database()
    db.close()
    
    mock_session.close.assert_called_once()
