/**
 * JavaScript client SDK for USDC Arbitrage API.
 */

class USDCArbitrageClient {
  /**
   * Initialize client.
   * @param {string} baseUrl - Base URL for API
   * @param {string} apiVersion - API version
   * @param {string} token - Authentication token
   */
  constructor(baseUrl = 'http://localhost:8000', apiVersion = '1.0', token = null) {
    this.baseUrl = baseUrl;
    this.apiVersion = apiVersion;
    this.token = token;
  }

  /**
   * Get request headers.
   * @returns {Object} - Headers object
   */
  _getHeaders() {
    const headers = {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
      'Accept-Version': this.apiVersion,
    };

    if (this.token) {
      headers['Authorization'] = `Bearer ${this.token}`;
    }

    return headers;
  }

  /**
   * Handle API response.
   * @param {Response} response - Fetch response
   * @returns {Promise<Object>} - Response data
   */
  async _handleResponse(response) {
    if (!response.ok) {
      let errorDetail = {};
      try {
        errorDetail = await response.json();
      } catch (e) {
        errorDetail = { detail: await response.text() };
      }
      throw new Error(`API error: ${response.status} - ${JSON.stringify(errorDetail)}`);
    }

    return response.json();
  }

  /**
   * Login to API and get access token.
   * @param {string} username - Username
   * @param {string} password - Password
   * @returns {Promise<Object>} - Login response
   */
  async login(username, password) {
    const url = `${this.baseUrl}/auth/token`;
    const data = {
      username,
      password,
    };

    const response = await fetch(url, {
      method: 'POST',
      headers: this._getHeaders(),
      body: JSON.stringify(data),
    });

    const result = await this._handleResponse(response);

    if (result.access_token) {
      this.token = result.access_token;
    }

    return result;
  }

  /**
   * Refresh access token.
   * @param {string} refreshToken - Refresh token
   * @returns {Promise<Object>} - Refresh response
   */
  async refreshToken(refreshToken) {
    const url = `${this.baseUrl}/auth/refresh`;
    const data = {
      refresh_token: refreshToken,
    };

    const response = await fetch(url, {
      method: 'POST',
      headers: this._getHeaders(),
      body: JSON.stringify(data),
    });

    const result = await this._handleResponse(response);

    if (result.access_token) {
      this.token = result.access_token;
    }

    return result;
  }

  /**
   * List strategies.
   * @param {Object} options - Options
   * @param {boolean} options.activeOnly - Only include active strategies
   * @param {string} options.tag - Filter by tag
   * @param {string} options.search - Search in name and description
   * @param {number} options.skip - Number of records to skip
   * @param {number} options.limit - Maximum number of records
   * @returns {Promise<Array>} - List of strategies
   */
  async listStrategies({
    activeOnly = false,
    tag = null,
    search = null,
    skip = 0,
    limit = 100,
  } = {}) {
    const url = new URL(`${this.baseUrl}/strategies/`);
    url.searchParams.append('active_only', activeOnly);
    url.searchParams.append('skip', skip);
    url.searchParams.append('limit', limit);

    if (tag) {
      url.searchParams.append('tag', tag);
    }

    if (search) {
      url.searchParams.append('search', search);
    }

    const response = await fetch(url, {
      method: 'GET',
      headers: this._getHeaders(),
    });

    return this._handleResponse(response);
  }

  /**
   * Get strategy by ID.
   * @param {number} strategyId - Strategy ID
   * @param {boolean} includeCode - Include strategy code
   * @returns {Promise<Object>} - Strategy details
   */
  async getStrategy(strategyId, includeCode = false) {
    const url = new URL(`${this.baseUrl}/strategies/${strategyId}`);
    url.searchParams.append('include_code', includeCode);

    const response = await fetch(url, {
      method: 'GET',
      headers: this._getHeaders(),
    });

    return this._handleResponse(response);
  }

  /**
   * Create a new strategy.
   * @param {Object} request - Strategy creation request
   * @param {string} author - Author name
   * @returns {Promise<Object>} - Created strategy
   */
  async createStrategy(request, author = 'api') {
    const url = new URL(`${this.baseUrl}/strategies/`);
    url.searchParams.append('author', author);

    const response = await fetch(url, {
      method: 'POST',
      headers: this._getHeaders(),
      body: JSON.stringify(request),
    });

    return this._handleResponse(response);
  }

  /**
   * Update an existing strategy.
   * @param {number} strategyId - Strategy ID
   * @param {Object} request - Strategy update request
   * @param {string} author - Author name
   * @returns {Promise<Object>} - Updated strategy
   */
  async updateStrategy(strategyId, request, author = 'api') {
    const url = new URL(`${this.baseUrl}/strategies/${strategyId}`);
    url.searchParams.append('author', author);

    const response = await fetch(url, {
      method: 'PUT',
      headers: this._getHeaders(),
      body: JSON.stringify(request),
    });

    return this._handleResponse(response);
  }

  /**
   * Delete a strategy.
   * @param {number} strategyId - Strategy ID
   * @returns {Promise<Object>} - Deletion response
   */
  async deleteStrategy(strategyId) {
    const url = `${this.baseUrl}/strategies/${strategyId}`;

    const response = await fetch(url, {
      method: 'DELETE',
      headers: this._getHeaders(),
    });

    return this._handleResponse(response);
  }

  /**
   * Run a backtest.
   * @param {Object} request - Backtest request
   * @returns {Promise<Object>} - Backtest response
   */
  async runBacktest(request) {
    const url = `${this.baseUrl}/backtest/`;

    const response = await fetch(url, {
      method: 'POST',
      headers: this._getHeaders(),
      body: JSON.stringify(request),
    });

    return this._handleResponse(response);
  }

  /**
   * Get backtest by ID.
   * @param {number} backtestId - Backtest ID
   * @returns {Promise<Object>} - Backtest details
   */
  async getBacktest(backtestId) {
    const url = `${this.baseUrl}/backtest/${backtestId}`;

    const response = await fetch(url, {
      method: 'GET',
      headers: this._getHeaders(),
    });

    return this._handleResponse(response);
  }

  /**
   * List backtests.
   * @param {Object} options - Options
   * @param {number} options.strategyId - Filter by strategy ID
   * @param {string} options.status - Filter by status
   * @param {number} options.limit - Maximum number of records
   * @param {number} options.offset - Number of records to skip
   * @returns {Promise<Array>} - List of backtests
   */
  async listBacktests({
    strategyId = null,
    status = null,
    limit = 10,
    offset = 0,
  } = {}) {
    const url = new URL(`${this.baseUrl}/backtest/`);
    url.searchParams.append('limit', limit);
    url.searchParams.append('offset', offset);

    if (strategyId !== null) {
      url.searchParams.append('strategy_id', strategyId);
    }

    if (status !== null) {
      url.searchParams.append('status', status);
    }

    const response = await fetch(url, {
      method: 'GET',
      headers: this._getHeaders(),
    });

    return this._handleResponse(response);
  }

  /**
   * List backtest results.
   * @param {Object} options - Options
   * @param {number} options.strategyId - Filter by strategy ID
   * @param {string} options.status - Filter by status
   * @param {number} options.skip - Number of records to skip
   * @param {number} options.limit - Maximum number of records
   * @param {string} options.sortBy - Field to sort by
   * @param {string} options.sortOrder - Sort order (asc or desc)
   * @returns {Promise<Array>} - List of backtest results
   */
  async listResults({
    strategyId = null,
    status = null,
    skip = 0,
    limit = 100,
    sortBy = 'created_at',
    sortOrder = 'desc',
  } = {}) {
    const url = new URL(`${this.baseUrl}/results/`);
    url.searchParams.append('skip', skip);
    url.searchParams.append('limit', limit);
    url.searchParams.append('sort_by', sortBy);
    url.searchParams.append('sort_order', sortOrder);

    if (strategyId !== null) {
      url.searchParams.append('strategy_id', strategyId);
    }

    if (status !== null) {
      url.searchParams.append('status', status);
    }

    const response = await fetch(url, {
      method: 'GET',
      headers: this._getHeaders(),
    });

    return this._handleResponse(response);
  }

  /**
   * Get backtest result by ID.
   * @param {number} resultId - Result ID
   * @returns {Promise<Object>} - Result details
   */
  async getResult(resultId) {
    const url = `${this.baseUrl}/results/${resultId}`;

    const response = await fetch(url, {
      method: 'GET',
      headers: this._getHeaders(),
    });

    return this._handleResponse(response);
  }

  /**
   * Get transactions for a backtest result.
   * @param {number} resultId - Result ID
   * @param {number} skip - Number of records to skip
   * @param {number} limit - Maximum number of records
   * @returns {Promise<Array>} - List of transactions
   */
  async getResultTransactions(resultId, skip = 0, limit = 100) {
    const url = new URL(`${this.baseUrl}/results/${resultId}/transactions`);
    url.searchParams.append('skip', skip);
    url.searchParams.append('limit', limit);

    const response = await fetch(url, {
      method: 'GET',
      headers: this._getHeaders(),
    });

    return this._handleResponse(response);
  }

  /**
   * Get position snapshots for a backtest result.
   * @param {number} resultId - Result ID
   * @param {number} skip - Number of records to skip
   * @param {number} limit - Maximum number of records
   * @returns {Promise<Array>} - List of position snapshots
   */
  async getResultPositions(resultId, skip = 0, limit = 100) {
    const url = new URL(`${this.baseUrl}/results/${resultId}/positions`);
    url.searchParams.append('skip', skip);
    url.searchParams.append('limit', limit);

    const response = await fetch(url, {
      method: 'GET',
      headers: this._getHeaders(),
    });

    return this._handleResponse(response);
  }

  /**
   * Get equity curve for a backtest result.
   * @param {number} resultId - Result ID
   * @returns {Promise<Array>} - Equity curve data
   */
  async getResultEquityCurve(resultId) {
    const url = `${this.baseUrl}/results/${resultId}/equity_curve`;

    const response = await fetch(url, {
      method: 'GET',
      headers: this._getHeaders(),
    });

    return this._handleResponse(response);
  }

  /**
   * Export data in the requested format.
   * @param {Object} options - Export options
   * @param {string} options.dataType - Data type
   * @param {string} options.format - Export format
   * @param {Object} options.filters - Filters
   * @param {Date} options.startDate - Start date
   * @param {Date} options.endDate - End date
   * @param {Array<string>} options.exchanges - Exchanges
   * @param {Array<string>} options.symbols - Symbols
   * @param {number} options.strategyId - Strategy ID
   * @param {number} options.backtestId - Backtest ID
   * @param {number} options.limit - Maximum number of records
   * @returns {Promise<Blob>} - Exported data
   */
  async exportData({
    dataType,
    format = 'json',
    filters = null,
    startDate = null,
    endDate = null,
    exchanges = null,
    symbols = null,
    strategyId = null,
    backtestId = null,
    limit = 10000,
  }) {
    const url = `${this.baseUrl}/export/data`;
    const data = {
      data_type: dataType,
      format,
    };

    if (filters) {
      data.filters = filters;
    }

    if (startDate) {
      data.start_date = startDate.toISOString();
    }

    if (endDate) {
      data.end_date = endDate.toISOString();
    }

    if (exchanges) {
      data.exchanges = exchanges;
    }

    if (symbols) {
      data.symbols = symbols;
    }

    if (strategyId) {
      data.strategy_id = strategyId;
    }

    if (backtestId) {
      data.backtest_id = backtestId;
    }

    if (limit) {
      data.limit = limit;
    }

    const response = await fetch(url, {
      method: 'POST',
      headers: this._getHeaders(),
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      let errorDetail = {};
      try {
        errorDetail = await response.json();
      } catch (e) {
        errorDetail = { detail: await response.text() };
      }
      throw new Error(`API error: ${response.status} - ${JSON.stringify(errorDetail)}`);
    }

    return response.blob();
  }

  /**
   * Export backtest results in the requested format.
   * @param {number} backtestId - Backtest ID
   * @param {string} format - Export format
   * @param {boolean} includeTransactions - Include transaction details
   * @param {boolean} includePositions - Include position snapshots
   * @returns {Promise<Blob>} - Exported data
   */
  async exportBacktestResults(
    backtestId,
    format = 'json',
    includeTransactions = false,
    includePositions = false,
  ) {
    const url = new URL(`${this.baseUrl}/export/backtest/${backtestId}`);
    url.searchParams.append('format', format);
    url.searchParams.append('include_transactions', includeTransactions);
    url.searchParams.append('include_positions', includePositions);

    const response = await fetch(url, {
      method: 'GET',
      headers: this._getHeaders(),
    });

    if (!response.ok) {
      let errorDetail = {};
      try {
        errorDetail = await response.json();
      } catch (e) {
        errorDetail = { detail: await response.text() };
      }
      throw new Error(`API error: ${response.status} - ${JSON.stringify(errorDetail)}`);
    }

    return response.blob();
  }

  /**
   * Export strategy details in the requested format.
   * @param {number} strategyId - Strategy ID
   * @param {string} format - Export format
   * @param {boolean} includeCode - Include strategy code
   * @param {boolean} includeBacktests - Include backtest results
   * @returns {Promise<Blob>} - Exported data
   */
  async exportStrategy(
    strategyId,
    format = 'json',
    includeCode = false,
    includeBacktests = false,
  ) {
    const url = new URL(`${this.baseUrl}/export/strategy/${strategyId}`);
    url.searchParams.append('format', format);
    url.searchParams.append('include_code', includeCode);
    url.searchParams.append('include_backtests', includeBacktests);

    const response = await fetch(url, {
      method: 'GET',
      headers: this._getHeaders(),
    });

    if (!response.ok) {
      let errorDetail = {};
      try {
        errorDetail = await response.json();
      } catch (e) {
        errorDetail = { detail: await response.text() };
      }
      throw new Error(`API error: ${response.status} - ${JSON.stringify(errorDetail)}`);
    }

    return response.blob();
  }

  /**
   * Connect to WebSocket for backtest monitoring.
   * @param {string} clientId - Client ID
   * @param {Function} onMessage - Message handler
   * @param {Function} onError - Error handler
   * @param {Function} onClose - Close handler
   * @returns {WebSocket} - WebSocket connection
   */
  connectToBacktestWebSocket(clientId, onMessage, onError, onClose) {
    const url = new URL(`${this.baseUrl.replace('http', 'ws')}/ws/backtest/${clientId}`);
    
    if (this.token) {
      url.searchParams.append('token', this.token);
    }

    const ws = new WebSocket(url);

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onMessage(data);
      } catch (e) {
        console.error('Error parsing WebSocket message:', e);
      }
    };

    ws.onerror = (event) => {
      if (onError) {
        onError(event);
      }
    };

    ws.onclose = (event) => {
      if (onClose) {
        onClose(event);
      }
    };

    return ws;
  }

  /**
   * Subscribe to backtest updates.
   * @param {WebSocket} ws - WebSocket connection
   * @param {number} backtestId - Backtest ID
   */
  subscribeToBacktest(ws, backtestId) {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({
        type: 'subscribe',
        backtest_id: backtestId,
      }));
    }
  }

  /**
   * Unsubscribe from backtest updates.
   * @param {WebSocket} ws - WebSocket connection
   * @param {number} backtestId - Backtest ID
   */
  unsubscribeFromBacktest(ws, backtestId) {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({
        type: 'unsubscribe',
        backtest_id: backtestId,
      }));
    }
  }

  /**
   * Get current subscriptions.
   * @param {WebSocket} ws - WebSocket connection
   */
  getSubscriptions(ws) {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({
        type: 'get_subscriptions',
      }));
    }
  }
}

// Export for Node.js or browser
if (typeof module !== 'undefined' && module.exports) {
  module.exports = USDCArbitrageClient;
} else {
  window.USDCArbitrageClient = USDCArbitrageClient;
}