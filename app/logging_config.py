import logging
import os

import structlog


def configure_logging() -> None:
    """Configure structlog with optional Betterstack log shipping.

    Bridges structlog to stdlib logging so LogtailHandler can capture every
    structured log event and ship it to Betterstack with all fields indexed.

    Console output:
      - JSON when JSON_LOGS=true (production)
      - Human-readable otherwise (development)

    Betterstack shipping enabled when LOGTAIL_TOKEN is set.
    """
    json_logs = os.getenv("JSON_LOGS", "false").lower() == "true"
    logtail_token = os.getenv("LOGTAIL_TOKEN")

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    renderer = (
        structlog.dev.ConsoleRenderer()
        if not json_logs
        else structlog.processors.JSONRenderer()
    )
    formatter = structlog.stdlib.ProcessorFormatter(processor=renderer)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(console_handler)

    if logtail_token:
        from logtail import LogtailHandler

        logtail_handler = LogtailHandler(source_token=logtail_token)
        root.addHandler(logtail_handler)
