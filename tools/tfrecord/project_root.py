"""Import this file to automatically add the project root path to sys.path.
"""

import os
import sys


# Project root must not have __init__.py.
dir_path = os.path.dirname(os.path.abspath(__file__))
while os.path.exists(os.path.join(dir_path, '__init__.py')):
    dir_path = os.path.abspath(os.path.join(dir_path, os.path.pardir))

sys.path.append(dir_path)
