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

import pytest

from oarphpy import util
from oarphpy_test import testutil


pytest.importorskip(
  'tabulate',
  reason="Used for ThruputObserver.__str__() to create a table")


def test_empty():
  t1 = util.ThruputObserver()
  assert str(t1)


def test_some_thru():  
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


def test_some_thru2():  
  t = util.ThruputObserver()
  for _ in range(10):
    with t.observe() as my_t:
      my_t.update_tallies(n=1)
  assert t.n == 10


def test_union():
  t1 = util.ThruputObserver()
  t2 = util.ThruputObserver()
  t2.update_tallies(n=10)
  u = util.ThruputObserver.union((t1, t2))
  tail = lambda t: str(t).split()[2:] # Skip PID / ID header
  assert tail(u) == tail(t2)


def test_some_blocks_thru():
  t3 = util.ThruputObserver(name='test_thruput_observer', n_total=10)
  for _ in range(10):
    t3.update_tallies(n=1, new_block=True)
    t3.maybe_log_progress()
  t3.stop_block()
  assert re.search('N thru.*10', str(t3))
  assert re.search('N chunks.*10', str(t3))


def test_decorated():
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


def test_wrap_generator():
  import itertools
  
  def gen():
    for i in range(10):
      yield i
  
  tgen = util.ThruputObserver.to_monitored_generator(gen())

  assert re.search('N thru.*0', str(tgen))
  assert [0, 1, 2] == list(itertools.islice(tgen, 3))
  assert re.search('N thru.*3', str(tgen))

  assert [3, 4, 5, 6, 7, 8, 9] == list(tgen)
  assert re.search('N thru.*10', str(tgen))

def test_works_in_counter():
  from collections import Counter

  t1 = util.ThruputObserver(name='t1', n_total=3)
  t1.start_block()
  t1.stop_block(n=1, num_bytes=2)
  counter1 = Counter()
  counter1['thruput'] = t1

  t2 = util.ThruputObserver(name='t2', n_total=3)
  t2.start_block()
  t2.stop_block(n=1, num_bytes=2)
  t2.start_block()
  t2.stop_block(n=1, num_bytes=2)
  counter2 = Counter()
  counter2['thruput'] = t2

  final_counter = counter1 + counter2

  final = final_counter['thruput']
  assert final.name == 't1' # The first observer added to the counter wins
  assert final.n == 3
  assert final.num_bytes == 6
  assert final.n_total == 3
  assert len(final.ts) == 3
