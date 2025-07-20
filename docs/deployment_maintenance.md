# Deployment and Maintenance Procedures

## Deployment Guide

This document provides comprehensive instructions for deploying and maintaining the USDC Arbitrage Backtesting System in production environments.

### System Requirements

#### Hardware Requirements

- **Production Environment**:
  - CPU: 8+ cores (recommended: 16 cores)
  - RAM: 16+ GB (recommended: 32 GB)
  - Storage: 500+ GB SSD (recommended: 1 TB NVMe SSD)
  - Network: 1 Gbps (recommended: 10 Gbps)

#### Software Requirements

- Docker 20.10+
- Docker Compose 2.0+
- Kubernetes 1.24+ (for production deployment)
- Helm 3.8+
- PostgreSQL 14+ with TimescaleDB extension
- Redis 6.2+
- Python 3.11+

### Deployment Options

The system supports three deployment options:

1. **Docker Compose**: Suitable for development and small-scale deployments
2. **Kubernetes**: Recommended for production deployments
3. **Manual Installation**: For custom environments

### Docker Compose Deployment

#### Step 1: Clone the Repository

```bash
git clone https://github.com/myusdcarbitrage/usdc-arbitrage-backtesting.git
cd usdc-arbitrage-backtesting
```

#### Step 2: Configure Environment Variables

```bash
cp .env.example .env
```

Edit the `.env` file to set appropriate values for your environment:

```
# Database Configuration
DB_HOST=postgres
DB_PORT=5432
DB_NAME=usdc_arbitrage
DB_USER=postgres
DB_PASSWORD=your_secure_password

# Redis Configuration
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=your_secure_redis_password

# API Configuration
API_SECRET_KEY=your_secure_api_key
JWT_SECRET=your_secure_jwt_secret
JWT_EXPIRATION=3600
```

#### Step 3: Start the Services

```bash
docker-compose up -d
```

#### Step 4: Initialize the Database

```bash
docker-compose exec api python scripts/setup_database.py
```

#### Step 5: Create Initial Admin User

```bash
docker-compose exec api python scripts/init_auth.py --username admin --password secure_password --email admin@example.com
```

### Kubernetes Deployment

#### Step 1: Configure Kubernetes Manifests

```bash
cd k8s/base
```

Edit the configuration files in the `k8s/base` directory to match your environment.

#### Step 2: Create Kubernetes Secrets

```bash
kubectl create namespace usdc-arbitrage

kubectl create secret generic db-credentials \
  --namespace usdc-arbitrage \
  --from-literal=username=postgres \
  --from-literal=password=your_secure_password

kubectl create secret generic api-secrets \
  --namespace usdc-arbitrage \
  --from-literal=api-key=your_secure_api_key \
  --from-literal=jwt-secret=your_secure_jwt_secret
```

#### Step 3: Deploy with Helm

```bash
helm repo add timescale https://charts.timescale.com/
helm repo update

helm install timescaledb timescale/timescaledb-single \
  --namespace usdc-arbitrage \
  --values k8s/timescaledb-values.yaml

kubectl apply -k k8s/base
```

#### Step 4: Initialize the Database

```bash
kubectl exec -it deployment/api -n usdc-arbitrage -- python scripts/setup_database.py
```

#### Step 5: Create Initial Admin User

```bash
kubectl exec -it deployment/api -n usdc-arbitrage -- python scripts/init_auth.py --username admin --password secure_password --email admin@example.com
```

### Manual Installation

#### Step 1: Install Dependencies

```bash
# Install PostgreSQL with TimescaleDB
sudo apt-get update
sudo apt-get install -y postgresql postgresql-contrib

# Install TimescaleDB
sudo add-apt-repository ppa:timescale/timescaledb-ppa
sudo apt-get update
sudo apt-get install -y timescaledb-postgresql-14

# Configure TimescaleDB
sudo timescaledb-tune --yes

# Install Redis
sudo apt-get install -y redis-server

# Install Python dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### Step 2: Configure the Database

```bash
sudo -u postgres psql -c "CREATE DATABASE usdc_arbitrage;"
sudo -u postgres psql -c "CREATE USER usdc_user WITH ENCRYPTED PASSWORD 'your_secure_password';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE usdc_arbitrage TO usdc_user;"
sudo -u postgres psql -d usdc_arbitrage -c "CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"
```

#### Step 3: Initialize the Database Schema

```bash
source .venv/bin/activate
python scripts/setup_database.py
```

#### Step 4: Configure Environment Variables

```bash
cp .env.example .env
```

Edit the `.env` file with appropriate values.

#### Step 5: Start the Services

```bash
# Start API service
source .venv/bin/activate
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &

# Start worker processes
source .venv/bin/activate
celery -A src.api.worker worker --loglevel=info &
```

## Maintenance Procedures

### Database Maintenance

#### TimescaleDB Compression

TimescaleDB compression should be configured to optimize storage for historical data:

```sql
-- Connect to the database
psql -U usdc_user -d usdc_arbitrage

-- Set compression policy for market data
SELECT add_compression_policy('market_data', INTERVAL '7 days');
```

#### Database Backups

Implement regular database backups:

```bash
# Daily backup script
pg_dump -U usdc_user -d usdc_arbitrage -F c -f /backups/usdc_arbitrage_$(date +%Y%m%d).dump

# Restore from backup if needed
pg_restore -U usdc_user -d usdc_arbitrage -c /backups/usdc_arbitrage_20230101.dump
```

### System Updates

#### Updating the Application

```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt

# Apply database migrations
python scripts/migrate_database.py

# Restart services
docker-compose restart
# or
kubectl rollout restart deployment/api -n usdc-arbitrage
```

#### Updating Dependencies

Regularly update system dependencies:

```bash
# Update Docker images
docker-compose pull

# Update Kubernetes deployments
kubectl apply -k k8s/base
```

### Monitoring and Logging

#### Log Rotation

Configure log rotation to prevent disk space issues:

```bash
# Example logrotate configuration
cat > /etc/logrotate.d/usdc-arbitrage << EOF
/var/log/usdc-arbitrage/*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    create 0640 usdc-user usdc-user
}
EOF
```

#### Monitoring Setup

Set up monitoring with Prometheus and Grafana:

```bash
# Deploy monitoring stack
kubectl apply -f k8s/monitoring/prometheus.yaml
kubectl apply -f k8s/monitoring/grafana.yaml

# Access Grafana dashboard
kubectl port-forward svc/grafana 3000:3000 -n monitoring
```

### Scaling Procedures

#### Horizontal Scaling

To scale the API service horizontally:

```bash
# Docker Compose
docker-compose up -d --scale api=3

# Kubernetes
kubectl scale deployment api --replicas=3 -n usdc-arbitrage
```

#### Database Scaling

For database scaling:

1. **Read Replicas**: Set up PostgreSQL read replicas for query scaling
2. **Sharding**: Implement TimescaleDB multi-node for large datasets

```bash
# Example: Create a read replica in PostgreSQL
sudo -u postgres psql -c "SELECT pg_create_physical_replication_slot('replica_1');"
```

### Backup and Recovery

#### Full System Backup

Perform regular full system backups:

```bash
# Backup script
#!/bin/bash
BACKUP_DIR="/backups/$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR

# Backup database
pg_dump -U usdc_user -d usdc_arbitrage -F c -f $BACKUP_DIR/database.dump

# Backup configuration
cp .env $BACKUP_DIR/
cp -r config/ $BACKUP_DIR/

# Compress backup
tar -czf $BACKUP_DIR.tar.gz $BACKUP_DIR
rm -rf $BACKUP_DIR
```

#### Disaster Recovery

In case of system failure:

1. **Restore Database**: Restore from the latest backup
2. **Restore Configuration**: Deploy the backed-up configuration
3. **Verify Data Integrity**: Run validation scripts
4. **Restart Services**: Bring up all services in the correct order

```bash
# Restore database
pg_restore -U usdc_user -d usdc_arbitrage -c /backups/20230101/database.dump

# Restore configuration
cp /backups/20230101/.env .env
cp -r /backups/20230101/config/ ./config/

# Restart services
docker-compose down
docker-compose up -d
```

## Security Procedures

### SSL Certificate Management

Regularly update SSL certificates:

```bash
# Using certbot for Let's Encrypt certificates
certbot renew

# Copy certificates to the appropriate location
cp /etc/letsencrypt/live/api.myusdcarbitrage.com/fullchain.pem /path/to/app/certs/
cp /etc/letsencrypt/live/api.myusdcarbitrage.com/privkey.pem /path/to/app/certs/

# Restart web server to apply new certificates
docker-compose restart nginx
```

### Security Updates

Apply security updates regularly:

```bash
# Update system packages
sudo apt-get update
sudo apt-get upgrade -y

# Update Docker images
docker-compose pull
docker-compose up -d

# Update Python dependencies
pip install -r requirements.txt --upgrade
```

### User Management

Manage system users:

```bash
# Create a new user
python scripts/init_auth.py --username new_user --password secure_password --email user@example.com

# Assign roles
python scripts/assign_role.py --username new_user --role analyst

# Disable a user
python scripts/disable_user.py --username former_user
```

## Performance Tuning

### Database Optimization

Optimize database performance:

```sql
-- Analyze tables for query optimization
ANALYZE market_data;

-- Create additional indexes for common queries
CREATE INDEX idx_market_data_exchange_symbol_time ON market_data (exchange, symbol, timestamp);

-- Configure TimescaleDB chunk size
SELECT set_chunk_time_interval('market_data', INTERVAL '1 day');
```

### API Performance

Optimize API performance:

1. **Connection Pooling**: Configure database connection pooling
2. **Caching**: Implement Redis caching for frequent queries
3. **Asynchronous Processing**: Use background tasks for long-running operations

```python
# Example Redis caching configuration
REDIS_CACHE_CONFIG = {
    "host": "redis",
    "port": 6379,
    "db": 1,
    "expire": 300  # 5 minutes
}
```

## Scheduled Maintenance Tasks

### Daily Tasks

- Database backups
- Log rotation
- Data validation checks

### Weekly Tasks

- System updates
- Performance monitoring review
- Security audit

### Monthly Tasks

- Full system backup
- SSL certificate verification
- User access review

## Deployment Checklist

Use this checklist for new deployments:

- [ ] System requirements verified
- [ ] Environment variables configured
- [ ] Database initialized
- [ ] Initial admin user created
- [ ] SSL certificates installed
- [ ] Firewall rules configured
- [ ] Monitoring set up
- [ ] Backup procedures tested
- [ ] Security scan completed
- [ ] Load testing performed