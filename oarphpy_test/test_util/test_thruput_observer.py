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

import re

from oarphpy import util
from oarphpy_test import testutil

requires_tabulate = testutil.importorskip(
  'tabulate',
  reason="Used for ThruputObserver.__str__() to create a table")

@requires_tabulate
def test_empty(self):
  t1 = util.ThruputObserver()
  assert str(t1)
  
@requires_tabulate
def test_some_thru(self):
  t2 = util.ThruputObserver()
  
  import random
  import time
  MAX_WAIT = 0.01
  for _ in range(10):
    with t2.observe(n=1, num_bytes=1):
      time.sleep(random.random() * MAX_WAIT)
  
  assert str(t2)
  assert t2.total_time <= 10. * MAX_WAIT
  assert re.search('N thru.*10', str(t2))
  assert re.search('N chunks.*10', str(t2))
  
@requires_tabulate
def test_union(self):
  t1 = util.ThruputObserver()
  t2 = util.ThruputObserver()
  t2.update_tallies(n=10)
  u = util.ThruputObserver.union((t1, t2))
  assert str(u) == str(t2)

@requires_tabulate
def test_some_blocks_thru(self):
  t3 = util.ThruputObserver(name='test_thruput_observer', n_total=10)
  for _ in range(10):
    t3.update_tallies(n=1, new_block=True)
    t3.maybe_log_progress()
  t3.stop_block()
  assert re.search('N thru.*10', str(t3))
  assert re.search('N chunks.*10', str(t3))

@requires_tabulate
def test_decorated(self):
  @util.ThruputObserver.wrap_func
  def monitored_func(x):
    import time
    time.sleep(x)

  monitored_func(0.01)
  monitored_func(0.01)
  monitored_func(0.01)
  assert re.search('N thru.*3', str(monitored_func.observer))
  assert re.search('N chunks.*3', str(monitored_func.observer))
  assert re.search('Total time.*0.03 seconds', str(monitored_func.observer))
