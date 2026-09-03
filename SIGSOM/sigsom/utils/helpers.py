import logging
from logging.config import dictConfig
from typing import Tuple
from dataclasses import dataclass
from typing import Tuple, Optional

class LevelFilter(object):
    def __init__(self, level):
        self.__level = level

    def filter(self, logRecord):
        return logRecord.levelno <= self.__level

# @dataclass(frozen=True)
# class SOMParams:
#     map_size: Tuple[int, int]
#     input_dims: int
#     dump_path: str
#     lr: float
#     Tmax: float
#     Tmin: float
#     iterations: int
#     batch_size: int
#     def __post_init__(self):
#         assert len(self.map_size) == 2, "SOM map has to be 2D."

@dataclass(frozen=True)
class SOMParams:
    map_size: Tuple[int, int]
    input_dims: int
    dump_path: str
    lr: float
    Tmax: float
    Tmin: float
    iterations: int
    batch_size: int
    
    # jumps
    jump_penalty_epochs: Optional[int] = None  # Default: Not used if None
    lambda_jump: Optional[float] = None  # Default: Not used if None
    decay: Optional[str] = None  # Default: Not used if None

    # feature selection
    l1_lambda: Optional[float] = None  # Strength of L1 regularization (default: None)
    feature_update_epochs: Optional[int] = None  # Interval for updating feature weights

    def __post_init__(self):
        assert len(self.map_size) == 2, "SOM map has to be 2D."

logging_config = dict(
    version=1,
    filters={
        'level_info_filter': {
            '()': LevelFilter,
            'level': logging.INFO
        }
    },
    formatters={
        'f': {
            'format': '%(asctime)s %(module)15.15s %(levelname)-8s %(threadName)-12s %(processName)s %(message)s'
        }
    },
    handlers={
        'h': {
            'class': 'logging.StreamHandler',
            'formatter': 'f',
            'level': 'ERROR'
        },
        'h_info': {
            'class': 'logging.StreamHandler',
            'formatter': 'f',
            'level': 'INFO',
            'filters': ['level_info_filter']
        }
    },
    loggers={
        'app': {
            'handlers': ['h', 'h_info'],
            'level': 'DEBUG',
            'propagate': False
        },
    },
)
dictConfig(logging_config)

logger = logging.getLogger('app')

