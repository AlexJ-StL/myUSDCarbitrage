# Database Setup Instructions

This document explains how to set up the PostgreSQL database for the USDC arbitrage project.

## Prerequisites

1. **PostgreSQL Installation**
   - Download and install PostgreSQL from [postgresql.org](https://www.postgresql.org/download/)
   - Make sure to add PostgreSQL bin directory to your PATH during installation
   - Default superuser is typically `postgres`

## Option 1: Using psql directly (recommended)

If PostgreSQL is properly installed and in your PATH:

```bash
# Windows
psql -U postgres -f .sql/database_setup.sql

# Linux/Mac
psql -U postgres -f .sql/database_setup.sql
```

## Option 2: Using the Python setup script

We've provided a Python script that can execute the SQL commands:

```bash
# Install required package
pip install psycopg2-binary

# Run the setup script
python src/setup_database.py
```

This script:
1. Connects to the default PostgreSQL database
2. Reads and executes the commands in `.sql/database_setup.sql`
3. Handles errors gracefully

## Option 3: Manual setup

If you prefer to set up the database manually:

1. Connect to PostgreSQL:
   ```
   psql -U postgres
   ```

2. Create the database:
   ```sql
   CREATE DATABASE usdc_arbitrage;
   ```

3. Create the user:
   ```sql
   CREATE USER arb_user WITH PASSWORD 'strongpassword';
   ```

4. Grant privileges:
   ```sql
   GRANT ALL PRIVILEGES ON DATABASE usdc_arbitrage TO arb_user;
   ```

5. Connect to the new database:
   ```sql
   \c usdc_arbitrage
   ```

6. Create the tables (copy from `.sql/database_setup.sql`)

## Troubleshooting

### "psql: command not found"
- PostgreSQL is not installed or not in your PATH
- Solution: Install PostgreSQL or add its bin directory to your PATH

### "Connection refused"
- PostgreSQL service is not running
- Solution: Start the PostgreSQL service

### "Password authentication failed"
- Incorrect password for the postgres user
- Solution: Use the correct password or reset it

### Python script errors
- Make sure psycopg2-binary is installed: `pip install psycopg2-binary`
- Check that PostgreSQL is running and accessible