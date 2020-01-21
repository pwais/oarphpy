# Copyright 2019 Maintainers of OarphPy
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

import pytest

try:
  from oarphpy import spark as S
  from oarphpy.spark import SessionFactory
    # Will throw `ImportError` on import if no Spark
  HAVE_SPARK = True
except ImportError:
  HAVE_SPARK = False
  class SessionFactory(object):
    pass

class LocalSpark(SessionFactory):
  import multiprocessing
  MASTER = 'local[%s]' % multiprocessing.cpu_count()

skip_if_no_spark = pytest.mark.skipif(not HAVE_SPARK, reason="Requires Spark")

def get_fixture_path(fixture_name):
  import os
  import oarphpy_test
  path = os.path.join(
              os.path.dirname(oarphpy_test.__file__),
              'fixtures',
              fixture_name)
  assert os.path.exists(path), "Can't find %s" % path
  return path
