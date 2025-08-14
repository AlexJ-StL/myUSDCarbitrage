"""Automated service restart and recovery mechanisms."""

import asyncio
import logging
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import psutil
import redis

logger = logging.getLogger(__name__)


class ServiceRecoveryManager:
    """Manages automated service restart and recovery."""

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        """Initialize service recovery manager."""
        self.redis_url = redis_url
        self._redis_client: Optional[redis.Redis] = None
        self.recovery_attempts = {}
        self.max_recovery_attempts = 3
        self.recovery_cooldown = 300  # 5 minutes

    @property
    def redis_client(self) -> redis.Redis:
        """Get Redis client with lazy initialization."""
        if self._redis_client is None:
            self._redis_client = redis.from_url(self.redis_url, decode_responses=True)
        return self._redis_client

    def get_service_processes(self) -> Dict[str, List[psutil.Process]]:
        """Get all service-related processes."""
        services = {
            "uvicorn": [],
            "celery": [],
            "redis": [],
            "postgres": [],
        }

        try:
            for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                try:
                    name = proc.info["name"].lower()
                    cmdline = " ".join(proc.info["cmdline"] or []).lower()

                    if "uvicorn" in name or "uvicorn" in cmdline:
                        services["uvicorn"].append(proc)
                    elif "celery" in name or "celery" in cmdline:
                        services["celery"].append(proc)
                    elif "redis" in name or "redis-server" in name:
                        services["redis"].append(proc)
                    elif "postgres" in name or "postgresql" in name:
                        services["postgres"].append(proc)

                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

        except Exception as e:
            logger.error(f"Failed to get service processes: {e}")

        return services

    def check_service_health(self, service_name: str) -> Dict[str, Any]:
        """Check health of a specific service."""
        processes = self.get_service_processes().get(service_name, [])

        if not processes:
            return {
                "status": "not_running",
                "process_count": 0,
                "processes": [],
            }

        healthy_processes = []
        unhealthy_processes = []

        for proc in processes:
            try:
                # Check if process is responsive
                cpu_percent = proc.cpu_percent()
                memory_info = proc.memory_info()

                # Basic health checks
                is_healthy = True
                health_issues = []

                # Check CPU usage (if too high for too long, might be stuck)
                if cpu_percent > 95:
                    health_issues.append("high_cpu")
                    is_healthy = False

                # Check memory usage (basic check)
                memory_mb = memory_info.rss / (1024 * 1024)
                if memory_mb > 1000:  # > 1GB might indicate memory leak
                    health_issues.append("high_memory")

                process_info = {
                    "pid": proc.pid,
                    "cpu_percent": cpu_percent,
                    "memory_mb": round(memory_mb, 2),
                    "status": proc.status(),
                    "create_time": proc.create_time(),
                    "health_issues": health_issues,
                }

                if is_healthy:
                    healthy_processes.append(process_info)
                else:
                    unhealthy_processes.append(process_info)

            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                unhealthy_processes.append({
                    "pid": proc.pid,
                    "error": str(e),
                })

        # Determine overall service status
        if not processes:
            status = "not_running"
        elif unhealthy_processes:
            status = "degraded"
        else:
            status = "healthy"

        return {
            "status": status,
            "process_count": len(processes),
            "healthy_processes": healthy_processes,
            "unhealthy_processes": unhealthy_processes,
        }

    async def restart_service(self, service_name: str) -> Dict[str, Any]:
        """Restart a specific service."""
        recovery_key = f"recovery:{service_name}"
        current_time = time.time()

        # Check recovery attempts
        try:
            attempts_data = self.redis_client.get(recovery_key)
            if attempts_data:
                import json

                attempts_info = json.loads(attempts_data)

                # Check if we're in cooldown period
                if (
                    current_time - attempts_info.get("last_attempt", 0)
                    < self.recovery_cooldown
                ):
                    return {
                        "success": False,
                        "reason": "cooldown_period",
                        "next_attempt_in": self.recovery_cooldown
                        - (current_time - attempts_info.get("last_attempt", 0)),
                    }

                # Check max attempts
                if attempts_info.get("count", 0) >= self.max_recovery_attempts:
                    return {
                        "success": False,
                        "reason": "max_attempts_reached",
                        "attempts": attempts_info.get("count", 0),
                    }

                attempts_count = attempts_info.get("count", 0) + 1
            else:
                attempts_count = 1

        except Exception:
            attempts_count = 1

        # Record recovery attempt
        try:
            import json

            self.redis_client.setex(
                recovery_key,
                3600,  # 1 hour TTL
                json.dumps({
                    "count": attempts_count,
                    "last_attempt": current_time,
                    "service": service_name,
                }),
            )
        except Exception as e:
            logger.warning(f"Failed to record recovery attempt: {e}")

        # Attempt service restart
        try:
            result = await self._perform_service_restart(service_name)

            if result["success"]:
                # Clear recovery attempts on successful restart
                try:
                    self.redis_client.delete(recovery_key)
                except Exception:
                    pass

            return result

        except Exception as e:
            logger.error(f"Failed to restart service {service_name}: {e}")
            return {
                "success": False,
                "reason": "restart_failed",
                "error": str(e),
                "attempts": attempts_count,
            }

    async def _perform_service_restart(self, service_name: str) -> Dict[str, Any]:
        """Perform the actual service restart."""
        if service_name == "uvicorn":
            return await self._restart_uvicorn()
        elif service_name == "celery":
            return await self._restart_celery()
        elif service_name == "redis":
            return await self._restart_redis()
        elif service_name == "postgres":
            return await self._restart_postgres()
        else:
            return {
                "success": False,
                "reason": "unknown_service",
                "service": service_name,
            }

    async def _restart_uvicorn(self) -> Dict[str, Any]:
        """Restart Uvicorn application server."""
        try:
            # Get current Uvicorn processes
            processes = self.get_service_processes()["uvicorn"]

            if processes:
                # Graceful shutdown first
                for proc in processes:
                    try:
                        proc.send_signal(signal.SIGTERM)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

                # Wait for graceful shutdown
                await asyncio.sleep(5)

                # Force kill if still running
                for proc in processes:
                    try:
                        if proc.is_running():
                            proc.kill()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

            # Start new Uvicorn process
            # Note: This assumes the service is managed by a process manager
            # In production, you'd typically use systemd, supervisor, or similar
            restart_command = os.getenv("UVICORN_RESTART_COMMAND")
            if restart_command:
                process = await asyncio.create_subprocess_shell(
                    restart_command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await process.communicate()

                if process.returncode == 0:
                    return {
                        "success": True,
                        "method": "command_restart",
                        "stdout": stdout.decode(),
                    }
                else:
                    return {
                        "success": False,
                        "reason": "restart_command_failed",
                        "stderr": stderr.decode(),
                    }
            else:
                return {
                    "success": False,
                    "reason": "no_restart_command",
                    "message": "Set UVICORN_RESTART_COMMAND environment variable",
                }

        except Exception as e:
            return {
                "success": False,
                "reason": "restart_exception",
                "error": str(e),
            }

    async def _restart_celery(self) -> Dict[str, Any]:
        """Restart Celery workers."""
        try:
            # Similar logic to Uvicorn restart
            processes = self.get_service_processes()["celery"]

            for proc in processes:
                try:
                    proc.send_signal(signal.SIGTERM)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            await asyncio.sleep(5)

            restart_command = os.getenv("CELERY_RESTART_COMMAND")
            if restart_command:
                process = await asyncio.create_subprocess_shell(
                    restart_command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await process.communicate()

                return {
                    "success": process.returncode == 0,
                    "method": "command_restart",
                    "stdout": stdout.decode() if process.returncode == 0 else None,
                    "stderr": stderr.decode() if process.returncode != 0 else None,
                }
            else:
                return {
                    "success": False,
                    "reason": "no_restart_command",
                    "message": "Set CELERY_RESTART_COMMAND environment variable",
                }

        except Exception as e:
            return {
                "success": False,
                "reason": "restart_exception",
                "error": str(e),
            }

    async def _restart_redis(self) -> Dict[str, Any]:
        """Restart Redis service."""
        # Note: Redis restart should typically be handled by system service manager
        return {
            "success": False,
            "reason": "manual_restart_required",
            "message": "Redis restart should be handled by system service manager",
        }

    async def _restart_postgres(self) -> Dict[str, Any]:
        """Restart PostgreSQL service."""
        # Note: PostgreSQL restart should typically be handled by system service manager
        return {
            "success": False,
            "reason": "manual_restart_required",
            "message": "PostgreSQL restart should be handled by system service manager",
        }

    async def check_and_recover_services(self) -> Dict[str, Any]:
        """Check all services and attempt recovery if needed."""
        services_to_check = ["uvicorn", "celery"]  # Only restart application services
        results = {}

        for service_name in services_to_check:
            health = self.check_service_health(service_name)

            if health["status"] in ["not_running", "degraded"]:
                logger.warning(
                    f"Service {service_name} is {health['status']}, attempting recovery"
                )
                recovery_result = await self.restart_service(service_name)
                results[service_name] = {
                    "health": health,
                    "recovery": recovery_result,
                }
            else:
                results[service_name] = {
                    "health": health,
                    "recovery": {"success": True, "reason": "not_needed"},
                }

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "services": results,
        }


# Global service recovery manager instance
recovery_manager = ServiceRecoveryManager()


async def get_recovery_manager() -> ServiceRecoveryManager:
    """Dependency to get service recovery manager instance."""
    return recovery_manager
