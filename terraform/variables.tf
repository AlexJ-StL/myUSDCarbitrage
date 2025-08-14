variable "aws_region" {
  description = "AWS region to deploy resources"
  type        = string
  default     = "us-west-2"
}

variable "cluster_name" {
  description = "Name of the EKS cluster"
  type        = string
  default     = "usdc-arbitrage-cluster"
}

variable "environment" {
  description = "Environment (e.g., dev, staging, production)"
  type        = string
  default     = "production"
}

variable "db_name" {
  description = "Name of the database"
  type        = string
  default     = "usdc_arbitrage"
}

variable "db_username" {
  description = "Username for the database"
  type        = string
  default     = "arb_user"
}

variable "db_password" {
  description = "Password for the database"
  type        = string
  sensitive   = true
}