# Copyright 2023 Maintainers of OarphPy
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

import os
import datetime

from oarphpy import coopdiskcache as coopc

def test_coopdc_basic():
  config = coopc.Config()

  kf = coopc.Client(config=config)

  now = datetime.datetime(2000, 1, 1)

  basedir = kf.basedir_for_key('my-key1', now=now)
  cached_data_paths = []
  for i in range(10):
    dest = os.path.join(basedir, "%s.txt" % i)
    with open(dest, 'w') as f:
      f.write('datas!')
    cached_data_paths.append(dest)
  

  for path in cached_data_paths:
    key = kf.get_key_for_path(path)
    expected_key = ''
    assert key == expected_key

    success = kf.update_key_time(
                    path=path,
                    expire_at=now + datetime.timedelta(days=1))
    assert success

    
    success = kf.update_key_time(
                    key=key,
                    expire_at=now + datetime.timedelta(days=1))
    assert success


  evictor = coopc.Evictor(config=config, now=now)
  evictor.run()
  
