# src/logger.py
import logging
import traceback
from langfuse import Langfuse
import os 

class PassengerAdvocateLogger:
    def __init__(
        self,
        logfile: str = "agent.log"
    ):
        
        secret = os.getenv("LANGFUSE_SECRET")
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")


        # Configure Python's built-in logger for console + file
        self.local_logger = logging.getLogger("PassengerAdvocateLogger")
        self.local_logger.setLevel(logging.DEBUG)

        # Avoid duplicate handlers if re-initialized
        if not self.local_logger.handlers:
            # Console handler
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

            # File handler
            fh = logging.FileHandler(logfile)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

            self.local_logger.addHandler(ch)
            self.local_logger.addHandler(fh)

        # Try to initialize Langfuse
        try:
            self.client = Langfuse(
                public_key=public_key,
                secret_key=secret,
                host=host,  # Optional, default shown
            )
            # self.client = Langfuse(secret=secret, public_key=public_key, host=host)
            self.enabled = True
        except Exception as e:
            self.local_logger.warning(f"Langfuse init failed, fallback to local logging: {e}")
            self.client = None
            self.enabled = False

    # Public methods
    def info(self, state: dict, node_name: str, message: str):
        self._log(state, node_name, "INFO")

    def error(self, state: dict, node_name: str, error: Exception):
        self._log(state, node_name, "ERROR", error=error)

    def debug(self, state: dict, node_name: str, message: str):
        self._log(state, node_name, "DEBUG")



    def _log(self, state: dict, node_name: str, user_msg: str, level: str = "INFO", error: Exception = None):
        """
        Logs agent state and node execution to console, file, and Langfuse.
        - state: AgentState dict
        - node_name: current node being executed
        - level: log level ("DEBUG", "INFO", "WARNING", "ERROR")
        - error: optional exception object
        """

        # Build message
        if error:
            message = f"[{node_name}] {level} {user_msg} ERROR: {error}\n{traceback.format_exc()}"
        else:
            message = f"[{node_name}] {level} SUCCESS: {state} - {user_msg}"

        # Always log locally (console + file)
        getattr(self.local_logger, level.lower(), self.local_logger.info)(message)

       