import logging
import os
import functools
import traceback
from logging.handlers import RotatingFileHandler

class CustomLogger:
    def __init__(
        self,
        name: str = "agent_workflow",
        log_file: str = "logs/agent_workflow.log",
        level: int = logging.DEBUG,
        max_bytes: int = 5 * 1024 * 1024,
        backup_count: int = 5
    ):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False  # Prevent duplicate logs if root logger is used

        # Create logs directory if missing
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # File Handler (Rotating)
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8"
        )
        file_formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)

        # Console Handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter("[%(levelname)s] %(message)s")
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.DEBUG)

        # Add handlers only once
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def get_logger(self):
        return self.logger

    def log_workflow_start(self):
        self.logger.info("🚀 Agent Workflow STARTED")

    def log_workflow_end(self):
        self.logger.info("🛑 Agent Workflow COMPLETED")

    def log_agent_call(self, agent_name: str):
        self.logger.info(f"📞 Calling Agent: {agent_name}")

    def log_agent_success(self, agent_name: str, result=None):
        message = f"✅ Agent '{agent_name}' SUCCESS"
        if result:
            message += f" | Result: {result}"
        self.logger.info(message)

    def log_agent_error(self, agent_name: str, error: Exception):
        self.logger.error(f"❌ Agent '{agent_name}' FAILED | Error: {error}")
        self.logger.debug(traceback.format_exc())

    def log_state_transition(self):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(state, *args, **kwargs):
                self.logger.info(f"[{func.__name__}] ➡️ Input: {state}")
                result = func(state, *args, **kwargs)
                self.logger.info(f"[{func.__name__}] ⬅️ Output: {result}")
                return result
            return wrapper
        return decorator

    def log_errors(self):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    self.logger.exception(f"[{func.__name__}] ❌ Exception: {e}")
                    raise
            return wrapper
        return decorator

    def log(self, message: str, level: str = "info"):
        level = level.lower()
        if level == "debug":
            self.logger.debug(message)
        elif level == "warning":
            self.logger.warning(message)
        elif level == "error":
            self.logger.error(message)
        elif level == "critical":
            self.logger.critical(message)
        else:
            self.logger.info(message)
