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

if __name__ == '__main__':
  import os
  import sys
  
  from oarphpy import util

  TEST_TEMPDIR_ROOT = '/tmp/oarphpy_test_spark'

  # First create a clean dir and a custom python library
  my_src_root = os.path.join(TEST_TEMPDIR_ROOT, 'my_src_root')
  util.cleandir(my_src_root)

  CREATE_LIB_SCRIPT = """
    mkdir -p {src_root} &&
    mkdir -p {src_root}/mymodule &&
    touch {src_root}/mymodule/__init__.py &&
    echo "pi = 3.14" > {src_root}/mymodule/foo.py
  """.format(src_root=my_src_root)
  util.run_cmd(CREATE_LIB_SCRIPT)

  # Make sure the custom library works
  TEST_SCRIPT = """
    cd {src_root} &&
    {python} -c 'from mymodule.foo import pi; print(pi)'
  """.format(src_root=my_src_root, python=sys.executable)
  out = util.run_cmd(TEST_SCRIPT, collect=True)
  assert out == b'3.14\n', "Check contents of %s" % (my_src_root)

  # Force the spark session to use our custom source root
  from oarphpy_test.testutil import LocalSpark
  class LocalSparkWithCustomLib(LocalSpark):
    SRC_ROOT = os.path.join(my_src_root, 'mymodule')

  def test_my_lib():
    import re
    import mymodule

    # The module should come from the included egg
    imp_path = mymodule.__file__
    assert re.match(
      r'^(.*)spark-(.*)/my_src_root-0\.0\.0-py(.+)\.egg/mymodule/__init__\.py$',
      imp_path)

    # Now verify the module itself
    from mymodule import foo
    assert foo.pi == 3.14
    
    return True

  # Now test that the lib gets egg-ified and shipped as a SparkFile
  from oarphpy import spark as S
  with LocalSparkWithCustomLib.sess() as spark:
    res = S.for_each_executor(spark, test_my_lib)
    assert res and all(res)

  print('test_spark_with_custom_library success!')
