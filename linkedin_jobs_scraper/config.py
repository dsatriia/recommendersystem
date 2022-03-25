import os
import logging


class Config:
    os.environ["LI_AT_COOKIE"] = "AQEDATnUOO0BfYygAAABf776ZHoAAAF_4wboelYAKm0H0j6x3xT0juMh9Xb_K8jwjwrXeZoUcO_m67Rde0hI9o85m56yc55lZJeatodO76iIdGjGOOjx4CEdL68meJDJ02UKObv6xSgX8fxjqiQHDs_g"
    LI_AT_COOKIE = os.environ['LI_AT_COOKIE'] if 'LI_AT_COOKIE' in os.environ else None
    print("debug baru")
    print(os.environ['LI_AT_COOKIE'])
    LOGGER_NAMESPACE = 'li:scraper'

    _level = logging.INFO

    if 'LOG_LEVEL' in os.environ:
        _level_env = os.environ['LOG_LEVEL'].upper().strip()

        if _level_env == 'DEBUG':
            _level = logging.DEBUG
        elif _level_env == 'INFO':
            _level = logging.INFO
        elif _level_env == 'WARN' or _level_env == 'WARNING':
            _level = logging.WARN
        elif _level_env == 'ERROR':
            _level = logging.ERROR
        elif _level_env == 'FATAL':
            _level = logging.FATAL

    LOGGER_LEVEL = _level
