# -*- encoding: utf-8 -*-
import os
import sys
import time
import logging

from alphaml.utils import dependencies
from alphaml.__version__ import __version__


__MANDATORY_PACKAGES__ = '''
numpy>=1.9
scikit-learn>=0.19,<0.20
lockfile>=0.10
smac>=0.8,<0.9
pyrfr>=0.6.1,<0.8
ConfigSpace>=0.4.0,<0.5
'''

dependencies.verify_packages(__MANDATORY_PACKAGES__)

if os.name != 'posix':
    raise ValueError(
        'Detected unsupported operating system: %s. Please check '
        'the compability information of auto-sklearn: http://automl.github.io'
        '/auto-sklearn/stable/installation.html#windows-osx-compability' %
        sys.platform
    )

if sys.version_info < (3, 5):
    raise ValueError(
        'Unsupported python version %s found. Auto-sklearn requires Python '
        '3.5 or higher.' % sys.version_info
    )


# Initialize the logging module.
# logger = logging.getLogger(__name__)
# logger.setLevel(level=logging.INFO)
# local_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
# handler = logging.FileHandler("data/log_%s.txt" % local_time)
# handler.setLevel(logging.DEBUG)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
#
# console = logging.StreamHandler()
# console.setLevel(logging.DEBUG)
# console.setFormatter(formatter)
#
# logger.addHandler(handler)
# logger.addHandler(console)

import json
import logging.config
with open("logging_configuration.json", 'r') as logging_configuration_file:
    config_dict = json.load(logging_configuration_file)

logging.config.dictConfig(config_dict)

# Log that the logger was configured
logger = logging.getLogger(__name__)
