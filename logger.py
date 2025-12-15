import logging
import os

logger = logging.getLogger(__name__)


def configure_from_env(
    primary_env_var: str = "NILOR_LOG_LEVEL", fallback_env_var: str = "LOG_LEVEL"
) -> None:
    # Prefer NILOR_LOG_LEVEL; fall back to LOG_LEVEL for backward compatibility
    chosen_var = primary_env_var if os.getenv(primary_env_var) else fallback_env_var
    value = os.getenv(chosen_var)

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
                f"⚠️  Nilor-Nodes: {chosen_var}='{value}' is invalid; defaulting to {effective_name}"
            )
        else:
            logging.info(
                f"ℹ️  Nilor-Nodes: {chosen_var}='{value}' → level set to {effective_name}"
            )
    else:
        logging.info(
            f"ℹ️  Nilor-Nodes: {primary_env_var} or {fallback_env_var} not set; defaulting to {effective_name}"
        )


__all__ = ["logger", "configure_from_env"]
