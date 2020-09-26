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

import os
import unittest
import sys

import pytest

from oarphpy import util
from oarphpy_test import testutil


def get_imagenet_fixture(fname):
  return testutil.get_fixture_path(os.path.join('images/imagenet', fname))


def test_np_truthy():
  np = pytest.importorskip("numpy")
  assert util.np_truthy(np.array([])) == False
  assert util.np_truthy(np.array([[]])) == False
  assert util.np_truthy(np.array([[0]])) == True
  assert util.np_truthy(False) == False
  assert util.np_truthy(0) == False
  assert util.np_truthy(1) == True


class TestGetSizeOfDeep(unittest.TestCase):
  def test_basic(self):
    pytest.importorskip('six', reason='Uses six for compatibility')
    
    assert util.get_size_of_deep("") == sys.getsizeof("")
    assert util.get_size_of_deep(0) == sys.getsizeof(0)

    bs = b"abc"
    assert util.get_size_of_deep(bytes(bs)) == sys.getsizeof(bytes(bs))
    assert \
      util.get_size_of_deep(bytearray(bs)) == sys.getsizeof(bytearray(bs))

  def test_sequences(self):
    pytest.importorskip('six', reason='Uses six for compatibility')

    assert util.get_size_of_deep([0]) == sys.getsizeof(0)
    assert util.get_size_of_deep([0, 0]) == 2 * sys.getsizeof(0)
    assert util.get_size_of_deep([[0, 0]]) == 2 * sys.getsizeof(0)

    assert util.get_size_of_deep({0: 0}) == 2 * sys.getsizeof(0)
    assert util.get_size_of_deep({0: [0]}) == 2 * sys.getsizeof(0)

  def test_generators(self):
    a = [1, 2]
    assert util.get_size_of_deep(iter(a)) == sys.getsizeof(iter(a))

    def ima_gen():
      for i in range(10):
        yield i
    
    assert util.get_size_of_deep(ima_gen()) == sys.getsizeof(ima_gen())

  def test_obj(self):
    pytest.importorskip('six', reason='Uses six for compatibility')

    class Obj(object):
      # Has a __dict__ attribute
      def __init__(self):
        self.x = 0
    expected = sys.getsizeof('x') + sys.getsizeof(0)
    assert util.get_size_of_deep(Obj()) == expected
    assert util.get_size_of_deep([Obj()]) == expected
    assert util.get_size_of_deep([Obj(), Obj()]) == 2 * expected

    class Slotted(object):
      __slots__ = ['x']
      def __init__(self):
        self.x = 0
    assert util.get_size_of_deep(Slotted()) == sys.getsizeof(0)
    assert util.get_size_of_deep([Slotted()]) == sys.getsizeof(0)
    assert util.get_size_of_deep([Slotted(), Slotted()]) == 2 * sys.getsizeof(0)
  
  def test_numpy(self):
    pytest.importorskip('six', reason='Uses six for compatibility')
    
    np = pytest.importorskip("numpy")
    
    arr = np.array([0])
    assert util.get_size_of_deep(arr) == arr.nbytes
    assert util.get_size_of_deep([arr]) == arr.nbytes
    assert util.get_size_of_deep([arr, arr]) == 2 * arr.nbytes
    assert util.get_size_of_deep({0: arr}) == (sys.getsizeof(0) + arr.nbytes)

  def test_big_lists(self):
    pytest.importorskip('six', reason='Uses six for compatibility')

    # Case: consider we have a large array, like a Tensor
    # but as a list. We want get_size_of_deep() to be fast.
    arr = list(range(int(1e6)))
    assert util.get_size_of_deep(arr) == (len(arr) * sys.getsizeof(0))

    # A big list of strings will still require a slower reduce
    ss = list(str(i) for i in range(100))
    assert util.get_size_of_deep(ss) == sum(sys.getsizeof(s) for s in ss)


def test_stable_hash():
  # pickle serializes objects differently in 2 vs 3
  if sys.version_info[0] < 3:
    # Python2: pinned to pickle protocol 2
    assert util.stable_hash("foo") == 231290827449591418329105380467963919721
    assert util.stable_hash(5) == 335425996500234075338788053721602902452
    assert util.stable_hash(object()) == 282611958877452113859631854844508872387
  else:
    # Python3: pinned to pickle protocol 3
    assert util.stable_hash("foo") == 315796808559568577246078438196621234360
    assert util.stable_hash(5) == 160000050781405605007085407481456440406
    assert util.stable_hash(object()) == 303935568206769042132548381433370108855


def test_ichunked():
  
  def list_ichunked(*args):
    return list(util.ichunked(*args))
  
  for n in range(10):
    assert list_ichunked([], n) == []
  
  for n in range(10):
    assert list_ichunked([1], n) == [(1,)]
  
  assert list_ichunked((1, 2, 3, 4, 5), 1) == [(1,), (2,), (3,), (4,), (5,)]
  assert list_ichunked((1, 2, 3, 4, 5), 2) == [(1, 2), (3, 4), (5,)]
  assert list_ichunked((1, 2, 3, 4, 5), 3) == [(1, 2, 3), (4, 5)]
  assert list_ichunked((1, 2, 3, 4, 5), 4) == [(1, 2, 3, 4), (5,)]
  assert list_ichunked((1, 2, 3, 4, 5), 5) == [(1, 2, 3, 4, 5)]
  assert list_ichunked((1, 2, 3, 4, 5), 6) == [(1, 2, 3, 4, 5)]
  
  assert list_ichunked('abcde', 4) == [('a', 'b', 'c', 'd'), ('e',)]


def test_roundrobin():
  pytest.importorskip('six', reason='Uses six.next')

  def list_roundrobin(*args):
    return list(util.roundrobin(*args))
  
  assert list_roundrobin() == []
  assert list_roundrobin('a') == ['a']
  assert list_roundrobin('ab') == ['a', 'b']
  assert list_roundrobin('ab', 'c') == ['a', 'c', 'b']
  assert list_roundrobin('abc', 'd', 'ef') == ['a', 'd', 'e', 'b', 'f', 'c']

  from collections import OrderedDict
  d = OrderedDict((
    ('a', [1, 2, 3]),
    ('b', [4]),
    ('c', [5, 6]),
  ))
  assert list_roundrobin(*d.values()) == [1, 4, 5, 2, 6, 3]


def test_row_of_constants():
  as_row = util.as_row_of_constants

  # Don't row-ify junk
  assert as_row(5) == {}
  assert as_row('moof') == {}
  assert as_row(dict) == {}
  assert as_row([]) == {}

  # Row-ify class and instance constants
  class Foo(object):
    nope = 1
    _NOPE = 2
    YES = 3
    YES2 = {'bar': 'baz'}
    def __init__(self):
      self.YUP = 4
      self._nope = 5
  
  assert as_row(Foo) == {
                      'YES': 3,
                      'YES2': {'bar': 'baz'},
                    }
  assert as_row(Foo()) == {
                      'YES': 3,
                      'YES2': {'bar': 'baz'},
                      'YUP': 4,
                    }
  
  # Don't row-ify containers of stuff
  assert as_row([Foo()]) == {}

  # Do recursively row-ify
  class Bar(object):
    FOO_CLS = Foo
    FOO_INST = Foo()
    def __init__(self):
      self.MY_FOO = Foo()

  assert as_row(Bar) == {
                      'FOO_CLS': 'Foo',
                      'FOO_CLS_YES': 3,
                      'FOO_CLS_YES2': {'bar': 'baz'},

                      'FOO_INST': 'Foo',
                      'FOO_INST_YES': 3,
                      'FOO_INST_YES2': {'bar': 'baz'},
                      'FOO_INST_YUP': 4,
                    }
  assert as_row(Bar()) == {
                      'FOO_CLS': 'Foo',
                      'FOO_CLS_YES': 3,
                      'FOO_CLS_YES2': {'bar': 'baz'},

                      'FOO_INST': 'Foo',
                      'FOO_INST_YES': 3,
                      'FOO_INST_YES2': {'bar': 'baz'},
                      'FOO_INST_YUP': 4,

                      'MY_FOO': 'Foo',
                      'MY_FOO_YES': 3,
                      'MY_FOO_YES2': {'bar': 'baz'},
                      'MY_FOO_YUP': 4,
                    }


class TestProxy(unittest.TestCase):
  class Foo(object):
    def __init__(self):
      self.x = 1

  def test_raw_obj(self):
    global_foo = TestProxy.Foo()
    global_foo.x = 0

    foo = TestProxy.Foo()
    assert foo.x == 1
    del foo
    global_foo.x == 0

  def test_wrapped_with_custom_dtor(self):
    global_foo = TestProxy.Foo()
    global_foo.x = 0

    class FooProxy(util.Proxy):
      def _on_delete(self):
        global_foo.x = 2
  
    foo = FooProxy(TestProxy.Foo())
    assert foo.x == 1
    del foo
    global_foo.x == 2


def test_sys_info():
  info = util.get_sys_info()
  assert 'oarphpy' in info['filepath']


def test_ds_store_is_stupid():
  assert util.is_stupid_mac_file('/yay/.DS_Store')
  assert util.is_stupid_mac_file('.DS_Store')
  assert util.is_stupid_mac_file('._.DS_Store')


def test_get_jpeg_size():
  for fname in ('1292397550_115450d9bc.jpg', '1345687_fde3a33c03.jpg'):
    path = get_imagenet_fixture(fname)
    jpeg_bytes = open(path, 'rb').read()
    width, height = util.get_jpeg_size(jpeg_bytes)

    imageio = pytest.importorskip("imageio")
    expected_h, expected_w, expected_c = imageio.imread(path).shape
    assert (width, height) == (expected_w, expected_h)


def test_all_files_recursive():
  import oarphpy
  paths = util.all_files_recursive(os.path.dirname(oarphpy.__file__))
  assert any('misc.py' in p for p in paths)

  fixture_path = get_imagenet_fixture('1292397550_115450d9bc.jpg')
  paths = util.all_files_recursive(
    os.path.join(os.path.dirname(fixture_path), '../../..'))
  assert any('1292397550_115450d9bc.jpg' in p for p in paths)


### GPU Utils

NVIDIA_SMI_MOCK_OUTPUT = (
b"""index, name, utilization.memory [%], name, memory.total [MiB], memory.free [MiB], memory.used [MiB]
0, GeForce GTX 1060 with Max-Q Design, 2, GeForce GTX 1060 with Max-Q Design, 6072, 5796, 276
""")

NVIDIA_SMI_MOCK_OUTPUT_8_K80s = (
b"""index, name, utilization.memory [%], name, memory.total [MiB], memory.free [MiB], memory.used [MiB]
0, Tesla K80, 0, Tesla K80, 11441, 11441, 0
1, Tesla K80, 0, Tesla K80, 11441, 11441, 0
2, Tesla K80, 0, Tesla K80, 11441, 11441, 0
3, Tesla K80, 0, Tesla K80, 11441, 11441, 0
4, Tesla K80, 0, Tesla K80, 11441, 11441, 0
5, Tesla K80, 0, Tesla K80, 11441, 11441, 0
6, Tesla K80, 0, Tesla K80, 11441, 11441, 0
7, Tesla K80, 5, Tesla K80, 11441, 11441, 0
"""
)

def test_gpu_get_infos(monkeypatch):
  # Test Smoke
  rows = util.GPUInfo.get_infos()
  
  # Test parsing a fixture
  def mock_run_cmd(*args, **kwargs):
    return NVIDIA_SMI_MOCK_OUTPUT
  from oarphpy.util import misc
  monkeypatch.setattr(misc, 'run_cmd', mock_run_cmd)
  rows = util.GPUInfo.get_infos()
  
  expected = util.GPUInfo()
  expected.index = 0
  expected.name = 'GeForce GTX 1060 with Max-Q Design'
  expected.mem_util_frac = 0.02
  expected.mem_free = 5796000000
  expected.mem_used = 276000000
  expected.mem_total = 6072000000
  assert rows == [expected]
