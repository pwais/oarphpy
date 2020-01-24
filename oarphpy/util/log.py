# Copyright 2020 Maintainers of OarphPy
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys

try:
  # FIXME For Tensorflow double-logging bug:
  # FIXME(https://github.com/abseil/abseil-py/issues/99)
  # FIXME(https://github.com/abseil/abseil-py/issues/102)
  # Unfortunately, many libraries that include absl (including Tensorflow)
  # will get bitten by double-logging due to absl's incorrect use of
  # the python logging library:
  #   2019-07-19 23:47:38,829 my_logger   779 : test
  #   I0719 23:47:38.829330 139904865122112 foo.py:63] test
  #   2019-07-19 23:47:38,829 my_logger   779 : test
  #   I0719 23:47:38.829469 139904865122112 foo.py:63] test
  # The code below fixes this double-logging.  FMI see:
  #   https://github.com/tensorflow/tensorflow/issues/26691#issuecomment-500369493
  
  import logging
  
  import absl.logging
  logging.root.removeHandler(absl.logging._absl_handler)
  absl.logging._warn_preinit_stderr = False
except ImportError as e:
  pass
except Exception as e:
  print("Failed to fix absl logging bug", e)
  pass

_LOGS = {}
def create_log(name='oarph'):
  global _LOGS
  if name not in _LOGS:
    import logging
    LOG_FORMAT = "%(asctime)s\t%(name)-4s %(process)d : %(message)s"
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    log.addHandler(console_handler)
    _LOGS[name] = log
  return _LOGS[name]

# NB: Spark workers will lazy-construct and cache logger instances
log = create_log()