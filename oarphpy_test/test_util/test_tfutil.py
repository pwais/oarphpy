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

import pytest

from oarphpy import util
from oarphpy_test import testutil

tf = pytest.importorskip("tensorflow")
tf.compat.v1.disable_v2_behavior()

def test_tf_data_session():
  expected = [[0, 1], [2, 3], [4, 5]]
  ds = tf.data.Dataset.from_tensor_slices(expected)
  with util.tf_data_session(ds) as (sess, iter_dataset):
    actual = list(v.tolist() for v in iter_dataset())
    assert expected == actual

def test_tf_records_file_as_list_of_str():
  TEST_TEMPDIR = testutil.test_tempdir(
                      'test_tf_records_file_as_list_of_str')
  util.cleandir(TEST_TEMPDIR)
  
  # Create the fixture: simply three strings in the file.  A TFRecords file
  # is just a size-delimited concatenation of string records.
  ss = [b'foo', b'bar', b'bazzzz']
  fixture_path = os.path.join(TEST_TEMPDIR, 'test.tfrecord')

  with tf.io.TFRecordWriter(fixture_path) as writer:
    for s in ss:
      writer.write(s)
  
  # Test reading!
  tf_lst = util.TFRecordsFileAsListOfStrings(open(fixture_path, 'rb'))
  assert len(tf_lst) == len(ss)
  assert sorted(tf_lst) == sorted(ss)
  for i in range(len(ss)):
    assert tf_lst[i] == ss[i]
