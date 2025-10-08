import logging
import os

logger = logging.getLogger(__name__)


def configure_from_env(env_var: str = "LOG_LEVEL") -> None:
    value = os.getenv(env_var)
    default_level = logging.INFO
    level = getattr(logging, value.upper(), None) if value else default_level
    if not isinstance(level, int):
        level = default_level
    logger.setLevel(level)

    # Announce the effective level to the terminal via the global handlers
    effective_name = logging.getLevelName(level)
    if value:
        if getattr(logging, value.upper(), None) is None:
            logging.warning(
                f"⚠️  Nilor-Nodes: {env_var}='{value}' in .env is invalid; defaulting to {effective_name}"
            )
        else:
            logging.info(
                f"ℹ️  Nilor-Nodes: {env_var}='{value}' in .env → level set to {effective_name}"
            )
    else:
        logging.info(
            f"ℹ️  Nilor-Nodes: {env_var} not set; defaulting to {effective_name}"
        )


__all__ = ["logger", "configure_from_env"]
