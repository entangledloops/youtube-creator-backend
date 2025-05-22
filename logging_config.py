"""
Logging configuration for the application
"""
import logging.config
import sys

def configure_logging():
    """Configure logging for the application"""
    config = {
        "version": 1,
        "disable_existing_loggers": False,  # Important: don't disable existing loggers
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "use_colors": None
            },
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": "%(asctime)s - %(name)s - %(levelname)s - %(client_addr)s - \"%(request_line)s\" %(status_code)s"
            }
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr"
            },
            "access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout"
            }
        },
        "loggers": {
            "uvicorn": {
                "handlers": ["default"],
                "level": "INFO"
            },
            "uvicorn.error": {
                "level": "INFO",
                "handlers": ["default"],
                "propagate": True
            },
            "uvicorn.access": {
                "handlers": ["access"],
                "level": "INFO",
                "propagate": False
            },
            # Add our application loggers
            "creator_processor": {
                "handlers": ["default"],
                "level": "INFO",
                "propagate": True
            },
            "youtube_analyzer": {
                "handlers": ["default"],
                "level": "INFO",
                "propagate": True
            },
            "llm_analyzer": {
                "handlers": ["default"],
                "level": "INFO",
                "propagate": True
            }
        }
    }
    
    logging.config.dictConfig(config) 