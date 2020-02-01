#!/usr/bin/env python
# vim: tabstop=2 shiftwidth=2 expandtab

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

# mylib/util.py
UTIL_PY_SRC = """
const_e = 2.718
"""

# mylib/prog.py
PROG_PY_SRC = """
def test_mylib():
  # NB: the current working directory is on the PYTHONPATH in Spark local mode,
  # so you might need to pop that path to force loading from the egg.  In this
  # test however, we run from /tmp, which doesn't have the mylib source.
  # See also `oarphpy.spark.test_egg()`
  # import sys
  # if '' in sys.path:
  #   sys.path.remove('')

  import re
  import mylib

  # The module should come from the included egg
  imp_path = mylib.__file__
  assert re.match(
    r'^(.*)spark-(.*)/mylib-0\.0\.0-py(.+)\.egg/mylib/__init__\.py$',
    imp_path), "Got " + imp_path

  # Now verify the module itself
  from mylib import util
  assert util.const_e == 2.718
  
  return True

if __name__ == '__main__':
  # NB: Below we test SessionFactory's source root auto-deduction.
  from oarphpy import spark as S
  with S.SessionFactory.sess() as spark:
    res = S.for_each_executor(spark, test_mylib)
    assert res and all(res)
  print('prog.py success!')
"""

if __name__ == '__main__':
  import os
  import sys
  
  from oarphpy import util

  TEST_TEMPDIR_ROOT = '/tmp/test_spark_script_in_standalone_lib'

  # First create a clean dir and a custom python library
  my_lib_root = os.path.join(TEST_TEMPDIR_ROOT, 'my_lib_root')
  util.cleandir(my_lib_root)

  CREATE_LIB_SCRIPT = """
    mkdir -p {src_root} &&
    mkdir -p {src_root}/mylib &&
    touch {src_root}/mylib/__init__.py
  """.format(src_root=my_lib_root)
  util.run_cmd(CREATE_LIB_SCRIPT)

  with open(os.path.join(my_lib_root, 'mylib', 'util.py'), 'w') as f:
    f.write(UTIL_PY_SRC)
  with open(os.path.join(my_lib_root, 'mylib', 'prog.py'), 'w') as f:
    f.write(PROG_PY_SRC)

  # Make sure the custom library works
  TEST_CMD = """
    cd {src_root} &&
    {python} -c 'from mylib.util import const_e; print(const_e)'
  """.format(src_root=my_lib_root, python=sys.executable)
  out = util.run_cmd(TEST_CMD, collect=True)
  assert out == b'2.718\n', "Check contents of %s" % (my_lib_root,)

  # Now test the embedded spark program
  TEST_CMD = """
    cd /tmp && {python} {src_root}/mylib/prog.py
  """.format(src_root=my_lib_root, python=sys.executable)
  out = util.run_cmd(TEST_CMD, collect=True)
  assert b'prog.py success!' in out, "Test failed %s" % (out,)

  print('test_spark_script_in_standalone_lib success!')
