provider "aws" {
  region = var.aws_region
}

# EKS Cluster
module "eks" {
  source          = "terraform-aws-modules/eks/aws"
  version         = "~> 19.0"
  cluster_name    = var.cluster_name
  cluster_version = "1.28"
  
  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets
  
  cluster_endpoint_public_access = true
  
  eks_managed_node_groups = {
    main = {
      min_size     = 2
      max_size     = 10
      desired_size = 3
      
      instance_types = ["t3.medium"]
      capacity_type  = "ON_DEMAND"
      
      labels = {
        Environment = var.environment
      }
    }
  }
  
  tags = {
    Environment = var.environment
    Project     = "usdc-arbitrage"
  }
}

# VPC
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"
  
  name = "${var.cluster_name}-vpc"
  cidr = "10.0.0.0/16"
  
  azs             = ["${var.aws_region}a", "${var.aws_region}b", "${var.aws_region}c"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
  
  enable_nat_gateway   = true
  single_nat_gateway   = true
  enable_dns_hostnames = true
  
  tags = {
    Environment = var.environment
    Project     = "usdc-arbitrage"
  }
}

# RDS PostgreSQL with TimescaleDB
resource "aws_db_subnet_group" "postgres" {
  name       = "${var.cluster_name}-postgres-subnet-group"
  subnet_ids = module.vpc.private_subnets
  
  tags = {
    Environment = var.environment
    Project     = "usdc-arbitrage"
  }
}

resource "aws_security_group" "postgres" {
  name        = "${var.cluster_name}-postgres-sg"
  description = "Allow PostgreSQL inbound traffic from EKS cluster"
  vpc_id      = module.vpc.vpc_id
  
  ingress {
    description     = "PostgreSQL from EKS"
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [module.eks.cluster_security_group_id]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Environment = var.environment
    Project     = "usdc-arbitrage"
  }
}

resource "aws_db_parameter_group" "timescaledb" {
  name   = "${var.cluster_name}-timescaledb-params"
  family = "postgres14"
  
  parameter {
    name  = "shared_preload_libraries"
    value = "timescaledb"
  }
  
  tags = {
    Environment = var.environment
    Project     = "usdc-arbitrage"
  }
}

resource "aws_db_instance" "postgres" {
  identifier             = "${var.cluster_name}-postgres"
  engine                 = "postgres"
  engine_version         = "14"
  instance_class         = "db.t3.medium"
  allocated_storage      = 20
  max_allocated_storage  = 100
  storage_type           = "gp2"
  storage_encrypted      = true
  
  db_name                = var.db_name
  username               = var.db_username
  password               = var.db_password
  
  parameter_group_name   = aws_db_parameter_group.timescaledb.name
  db_subnet_group_name   = aws_db_subnet_group.postgres.name
  vpc_security_group_ids = [aws_security_group.postgres.id]
  
  backup_retention_period = 7
  backup_window           = "03:00-04:00"
  maintenance_window      = "Mon:04:00-Mon:05:00"
  
  skip_final_snapshot     = var.environment != "production"
  deletion_protection     = var.environment == "production"
  
  tags = {
    Environment = var.environment
    Project     = "usdc-arbitrage"
  }
}

# ElastiCache Redis
resource "aws_elasticache_subnet_group" "redis" {
  name       = "${var.cluster_name}-redis-subnet-group"
  subnet_ids = module.vpc.private_subnets
}

resource "aws_security_group" "redis" {
  name        = "${var.cluster_name}-redis-sg"
  description = "Allow Redis inbound traffic from EKS cluster"
  vpc_id      = module.vpc.vpc_id
  
  ingress {
    description     = "Redis from EKS"
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [module.eks.cluster_security_group_id]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Environment = var.environment
    Project     = "usdc-arbitrage"
  }
}

resource "aws_elasticache_cluster" "redis" {
  cluster_id           = "${var.cluster_name}-redis"
  engine               = "redis"
  node_type            = "cache.t3.small"
  num_cache_nodes      = 1
  parameter_group_name = "default.redis7"
  engine_version       = "7.0"
  port                 = 6379
  
  subnet_group_name    = aws_elasticache_subnet_group.redis.name
  security_group_ids   = [aws_security_group.redis.id]
  
  tags = {
    Environment = var.environment
    Project     = "usdc-arbitrage"
  }
}

# S3 Bucket for Backups
resource "aws_s3_bucket" "backups" {
  bucket = "${var.cluster_name}-backups-${var.environment}"
  
  tags = {
    Environment = var.environment
    Project     = "usdc-arbitrage"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "backups_lifecycle" {
  bucket = aws_s3_bucket.backups.id
  
  rule {
    id     = "archive-old-backups"
    status = "Enabled"
    
    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }
    
    transition {
      days          = 90
      storage_class = "GLACIER"
    }
    
    expiration {
      days = 365
    }
  }
}

# IAM Role for EKS to access S3
resource "aws_iam_role_policy" "eks_s3_access" {
  name   = "eks-s3-access"
  role   = module.eks.cluster_iam_role_name
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket",
          "s3:DeleteObject"
        ]
        Effect   = "Allow"
        Resource = [
          aws_s3_bucket.backups.arn,
          "${aws_s3_bucket.backups.arn}/*"
        ]
      }
    ]
  })
}