from enum import Enum
from typing import Union

__all__ = ["Color", "LOG_LEVELS", "debug", "info", "warn", "error", "set_log_level", "get_log_level"]

class Color(Enum):
    BLACK          = '\033[30m'
    RED            = '\033[31m'
    GREEN          = '\033[32m'
    YELLOW         = '\033[33m'
    BLUE           = '\033[34m'
    MAGENTA        = '\033[35m'
    CYAN           = '\033[36m'
    WHITE          = '\033[37m'
    COLOR_DEFAULT  = '\033[39m'
    BOLD           = '\033[1m'
    UNDERLINE      = '\033[4m'
    INVISIBLE      = '\033[08m'
    REVERSE        = '\033[07m'
    BG_BLACK       = '\033[40m'
    BG_RED         = '\033[41m'
    BG_GREEN       = '\033[42m'
    BG_YELLOW      = '\033[43m'
    BG_BLUE        = '\033[44m'
    BG_MAGENTA     = '\033[45m'
    BG_CYAN        = '\033[46m'
    BG_WHITE       = '\033[47m'
    BG_DEFAULT     = '\033[49m'
    RESET          = '\033[0m'

    def __str__(self):
        return self.value
    
    def __call__(self, s):
        return str(self) + str(s) + str(Color.RESET)

LOG_LEVELS = {
    'debug': 0,
    'info':  1,
    'warn':  2,
    'error': 3,
}

log_level = 0

def set_log_level(level: Union[str, int]):
    global log_level
    if isinstance(level, str):
        log_level = LOG_LEVELS[level]
    else:
        log_level = level

def get_log_level():
    return log_level

def debug(*args):
    if log_level <= LOG_LEVELS['debug']:
        print(*args)
def info(*args):
    if log_level <= LOG_LEVELS['info']:
        print(*args)
def warn(*args, prefix=True):
    if log_level <= LOG_LEVELS['warn']:
        if prefix and any(args):
            print(
                Color.YELLOW('WARNING:'),
                *args
            )
        else:
            print(*args)
def error(*args, prefix=True):
    if log_level <= LOG_LEVELS['error']:
        if prefix and any(args):
            print(
                Color.RED('ERROR:'),
                *args
            )
        else:
            print(*args)
