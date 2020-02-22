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

from oarphpy_test import testutil
from oarphpy_test.testutil import LocalSpark
from oarphpy_test.testutil import skip_if_no_spark


@skip_if_no_spark
def test_union_dfs():
  from oarphpy import spark as S
  with testutil.LocalSpark.sess() as spark:
    from pyspark.sql import Row
    df1 = spark.createDataFrame([Row(a=1, b=2.0)])
    df2 = spark.createDataFrame([Row(a=3, c='foo')])

    unioned = S.union_dfs(df1, df2)
    
    EXPECTED = """
         a    b     c
      0  1  2.0  None
      1  3  NaN   foo
    """
    actual = str(unioned.toPandas())
    def to_lines(str):
      lines = [line.strip() for line in str.split('\n')]
      return [l for l in lines if l]
    assert to_lines(EXPECTED) == to_lines(actual)


@skip_if_no_spark
def test_spark_selftest():
  assert testutil.LocalSpark.selftest(modname='oarphpy_test')


@skip_if_no_spark
def test_spark_with_custom_library_script():
  # We must run this test in a subprocess in order for it to have an isolated
  # Spark session
  SCRIPT_PATH = testutil.get_fixture_path('test_spark_with_custom_library.py')
  
  from oarphpy import util
  out = util.run_cmd(sys.executable + ' ' + SCRIPT_PATH, collect=True)
  assert b'test_spark_with_custom_library success!' in out


@skip_if_no_spark
def test_spark_script_in_standalone_lib():
  # We must run this test in a subprocess in order for it to have an isolated
  # Spark session
  SCRIPT_PATH = testutil.get_fixture_path(
    'test_spark_script_in_standalone_lib.py')
  
  from oarphpy import util
  out = util.run_cmd(sys.executable + ' ' + SCRIPT_PATH, collect=True)
  assert b'test_spark_script_in_standalone_lib success!' in out


@skip_if_no_spark
def test_spark_with_custom_library_in_notebook():
  pytest.importorskip("jupyter")

  import re
  from oarphpy import util
  
  NB_PATH = testutil.get_fixture_path(
    'test_spark_ships_custom_library_in_notebook.ipynb')

  # We need to fork off nbconvert to run the test
  TEST_CMD = """
    cd /tmp && 
    jupyter-nbconvert --ExecutePreprocessor.timeout=3600 \
      --to notebook --execute --output /tmp/out \
      {notebook_path}
  """.format(notebook_path=NB_PATH)
  out = util.run_cmd(TEST_CMD, collect=True)
  assert re.search('Writing .* bytes to /tmp/out.ipynb', out.decode())


@skip_if_no_spark
def test_cluster_get_info_smoke():
  with testutil.LocalSpark.sess() as spark:
    from oarphpy import spark as S
    S.cluster_get_info(spark)


@skip_if_no_spark
def test_pi():
  with testutil.LocalSpark.sess() as spark:
    from oarphpy import spark as S
    S.test_pi(spark)


@skip_if_no_spark
def test_spark_tensorflow():
  from oarphpy import spark as S
  pytest.importorskip("tensorflow")
  with testutil.LocalSpark.sess() as spark:
    S.test_tensorflow(spark)


@skip_if_no_spark
def test_spark_cpu_count():
  from oarphpy import spark as S
  import multiprocessing
  with testutil.LocalSpark.sess() as spark:
    assert multiprocessing.cpu_count() == S.cluster_cpu_count(spark)


@skip_if_no_spark
class TestArchiveRDD(unittest.TestCase):

  def _check_rdd(self, fw_rdd, fixture_strs):
    names = fw_rdd.map(lambda fw: fw.name).collect()
    expected_names = [s.decode('utf-8') for s in fixture_strs]
    assert sorted(names) == sorted(expected_names)

    values = fw_rdd.map(lambda fw: fw.data).collect()
    assert sorted(values) == sorted(fixture_strs)

  def test_archive_rdd_zip(self):
    TEST_TEMPDIR = testutil.test_tempdir('test_archive_rdd_zip')
    fixture_path = os.path.join(TEST_TEMPDIR, 'test.zip')
      
    # Create the fixture:
    # test.zip
    #  |- foo: "foo"
    #  |- bar: "bar"
    # ... an archive with a few files, where each file contains just a string
    #   that matches the name of the file in the archive
    ss = [b'foo', b'bar', b'baz']
      
    import zipfile
    with zipfile.ZipFile(fixture_path, mode='w') as z:
      for s in ss:
        z.writestr(s.decode('utf-8'), s)
      
    # Test Reading!
    with testutil.LocalSpark.sess() as spark:
      from oarphpy import spark as S
      fw_rdd = S.archive_rdd(spark, fixture_path)
      self._check_rdd(fw_rdd, ss)

  def test_archive_rdd_tar(self):
    TEST_TEMPDIR = testutil.test_tempdir('test_archive_rdd_tar')
    fixture_path = os.path.join(TEST_TEMPDIR, 'test.tar')

    # Create the fixture:
    # test.tar
    #  |- foo: "foo"
    #  |- bar: "bar"
    # ... an archive with a few files, where each file contains just a string
    #   that matches the name of the file in the archive
    ss = [b'foo', b'bar', b'baz']

    import tarfile
    with tarfile.open(fixture_path, mode='w') as t:
      for s in ss:
        from io import BytesIO
        buf = BytesIO()
        buf.write(s)
        buf.seek(0)

        buf.seek(0, os.SEEK_END)
        buf_len = buf.tell()
        buf.seek(0)
        
        info = tarfile.TarInfo(name=s.decode('utf-8'))
        info.size = buf_len
        
        t.addfile(tarinfo=info, fileobj=buf)
      
    # Test Reading!
    with testutil.LocalSpark.sess() as spark:
      from oarphpy import spark as S
      fw_rdd = S.archive_rdd(spark, fixture_path)
      self._check_rdd(fw_rdd, ss)


@skip_if_no_spark
def test_get_balanced_sample():
  from oarphpy import spark as S
  from pyspark.sql import Row
  
  VAL_TO_COUNT = {
    'a': 10,
    'b': 100,
    'c': 1000,
  }
  rows = []
  for val, count in VAL_TO_COUNT.items():
    for _ in range(count):
      i = len(rows)
      rows.append(Row(id=i, val=val))
  
  def _get_category_to_count(df):
    from collections import defaultdict
    rows = df.collect()

    category_to_count = defaultdict(int)
    for row in rows:
      category_to_count[row.val] += 1
    return category_to_count

  def check_sample_in_expectation(df, n_per_category, expected):
    import numpy as np
    import pandas as pd
    
    # NB: even with a fixed seed below, this test's behavior is dependent
    # on the number of system CPUs since Spark draws random numbers
    # concurrently.  To simulate a low-cpu system, use a Spark master local[1].
    # We do more trials to accomodate low-cpu systems (e.g. CircleCI).
    N_TRIALS = 30
    rows = [
      _get_category_to_count(
        S.get_balanced_sample(
          df, 'val',
          n_per_category=n_per_category,
          seed=100*s))
      for s in range(N_TRIALS)
    ]
    pdf = pd.DataFrame(rows)
    pdf = pdf.fillna(0)

    ks = sorted(expected.keys())
    mu = pdf.mean()
    actual_arr = np.array([mu[k] for k in ks])
    expected_arr = np.array([expected[k] for k in ks])
    
    import numpy.testing as npt
    npt.assert_allclose(actual_arr, expected_arr, rtol=0.2)
      # NB: We can only test to about 20% accuracy with this few samples

  with testutil.LocalSpark.sess() as spark:
    df = spark.createDataFrame(rows)

    check_sample_in_expectation(
      df, n_per_category=1, expected={'a': 1, 'b': 1, 'c': 1})

    check_sample_in_expectation(
      df, n_per_category=10, expected={'a': 10, 'b': 10, 'c': 10})

    check_sample_in_expectation(
      df, n_per_category=20, expected={'a': 10, 'b': 10, 'c': 10})

    check_sample_in_expectation(
      df, n_per_category=None, expected={'a': 10, 'b': 10, 'c': 10})


@skip_if_no_spark
def test_spark_df_to_tf_dataset():
  pytest.importorskip("tensorflow")

  from oarphpy.spark import spark_df_to_tf_dataset
  with testutil.LocalSpark.sess() as spark:

    import numpy as np
    import tensorflow as tf
    from pyspark.sql import Row

    def tf_dataset_to_list(ds):
      from oarphpy import util
      with util.tf_data_session(ds) as (sess, iter_dataset):
        return list(iter_dataset())

    df = spark.createDataFrame([
      Row(id='r1', x=1, y=[3., 4., 5.]),
      Row(id='r2', x=2, y=[6.]),
      Row(id='r3', x=3, y=[7., 8., 9.]),
    ])

    # Test empty
    ds = spark_df_to_tf_dataset(
            df.filter('x == False'), # Empty!,
            'id',
            spark_row_to_tf_element=lambda r: ('test',),
            tf_element_types=(tf.string,))
    assert tf_dataset_to_list(ds) == []

    # Test simple
    ds = spark_df_to_tf_dataset(
            df,
            'id',
            spark_row_to_tf_element=lambda r: (r.x,),
            tf_element_types=(tf.int64,))
    assert sorted(tf_dataset_to_list(ds)) == [(1,), (2,), (3,)]

    # Test Complex
    ds = spark_df_to_tf_dataset(
            df,
            'id',
            spark_row_to_tf_element=lambda r: (r.x, r.id, r.y),
            tf_element_types=(tf.int64, tf.string, tf.float64))
    expected = [
      (1, b'r1', np.array([3., 4., 5.])),
      (2, b'r2', np.array([6.])),
      (3, b'r3', np.array([7., 8., 9.])),
    ]
    items = list(zip(sorted(tf_dataset_to_list(ds)), sorted(expected)))
    for actual, exp in items:
      print('actual', actual,'exp', exp)
      assert len(actual) == len(exp)
      for i in range(len(actual)):
        np.testing.assert_array_equal(actual[i], exp[i])
    

################################################################################
### Test Row Adapter

## NB: these classes must be declared package-level (rather than test-scoped)
## for cloudpickle / spark to discover them properly

class Slotted(object):
  __slots__ = ('foo', 'bar', '_not_hidden')
  
  def __init__(self, **kwargs):
    # NB: ctor for convenience; SparkTypeAdapter does not require it
    for k in self.__slots__:
      setattr(self, k, kwargs.get(k))
  
  def __repr__(self):
    # For unit tests below
    return "Slotted(%s)" % ([(k, getattr(self, k)) for k in self.__slots__],)


class Unslotted(object):
  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      setattr(self, k, v)
  
  def __repr__(self):
    # For unit tests below
    # Hide any 'hidden' attributes since `RowAdapter` will elide them
    attrs = dict(
              (k, v) for k, v in self.__dict__.items()
              if not k.startswith('__')
    )
    return "Unslotted(%s)" % str(sorted(attrs.items()))


def _pandas_compare_str(pdf, expected):
  def cleaned(s):
    lines = [l.strip() for l in s.split('\n')]
    return '\n'.join(l for l in lines if l)
  assert cleaned(pdf.to_string()) == cleaned(expected)


def _check_serialization(spark, rows, testname, schema=None):
  from oarphpy import util
  from oarphpy.spark import RowAdapter 

  TEST_TEMPDIR = testutil.test_tempdir('spark_row_adapter_test')

  adapted_rows = [RowAdapter.to_row(r) for r in rows]
  if schema:
    df = spark.createDataFrame(
      adapted_rows, schema=schema, verifySchema=False)
        # verifySchema is expensive and improperly erros on mostly empty rows
  else:
    df = spark.createDataFrame(adapted_rows)
      # Automatically samples both rows to get schema
  outpath = os.path.join(TEST_TEMPDIR, 'rowdata_%s' % testname)
  df.write.parquet(outpath)

  df2 = spark.read.parquet(outpath)
  decoded_wrapped_rows = df2.collect()
  
  decoded_rows = [
    RowAdapter.from_row(row)
    for row in decoded_wrapped_rows
  ]
  
  # We can't do assert sorted(rows) == sorted(decoded_rows)
  # because numpy syntatic sugar breaks ==
  import pprint
  def sorted_row_str(rowz):
    return pprint.pformat(sorted(rowz))
  assert sorted_row_str(rows) == sorted_row_str(decoded_rows)
  
  return df


@skip_if_no_spark
def test_row_adapter_basic():
  import numpy as np

  from pyspark.sql import Row

  from oarphpy.spark import RowAdapter
  
  rows = [
    Row(
      id=1,
      np_number=np.float32(1.),
      a=np.array([1]), 
      b={
        'foo': np.array( [ [1] ], dtype=np.uint8)
      },
      c=[
        np.array([[[1.]], [[2.]], [[3.]]])
      ],
      d=Slotted(foo=5, bar="abc", _not_hidden=1),
      e=[Slotted(foo=6, bar="def", _not_hidden=1)],
      f=Unslotted(meow=4, _not_hidden=1, __hidden=2),
      g=Unslotted(), # Intentionally empty; adapter should set nothing
      h=Row(i=1, j=2),
    ),

    # Include a mostly empty row below to exercise Spark type validation.
    # Spark will ensure the row below and row above have the same schema;
    # note that `None` (or 'null') is only allowed for Struct / Row types.
    Row(
      id=2,
      np_number=np.float32(2.),
      a=np.array([]),
      b={},
      c=[],
      d=None,
      e=[],
      f=None,
      g=None,
      h=Row(i=3, j=3),
    ),
  ]

  with testutil.LocalSpark.sess() as spark:

    ## Test basic round-trip serialization and adaptation
    
    df_all = spark.createDataFrame([RowAdapter.to_row(r) for r in rows])
    EXPECTED_ALL = """
                                                                              0                                            1
    a                                (oarphpy.spark.Tensor, int64, C, [1], [1])  (oarphpy.spark.Tensor, float64, C, [0], [])
    b              {'foo': ('oarphpy.spark.Tensor', 'uint8', 'C', [1, 1], [1])}                                           {}
    c          [(oarphpy.spark.Tensor, float64, C, [3, 1, 1], [1.0, 2.0, 3.0])]                                           []
    d                              (oarphpy_test.test_spark.Slotted, 1, abc, 5)                                         None
    e                            [(oarphpy_test.test_spark.Slotted, 1, def, 6)]                                           []
    f                                 (oarphpy_test.test_spark.Unslotted, 1, 4)                                         None
    g                                      (oarphpy_test.test_spark.Unslotted,)                                         None
    h                                                                    (1, 2)                                       (3, 3)
    id                                                                        1                                            2
    np_number                                                                 1                                            2
    """
    _pandas_compare_str(df_all.orderBy('id').toPandas().T, EXPECTED_ALL)

    _check_serialization(spark, rows, 'basic')

    ## Test Schema Deduction
    mostly_empty = Row(
      id=2,
      np_number=None,
      a=None,
      b={},
      c=[],
      d=None,
      e=[],
      f=None,
      g=None,
      h=None,
    )
    mostly_empty_adapted = RowAdapter.to_row(mostly_empty)

    # Spark can't deduce schema from the empty-ish row ...
    with pytest.raises(ValueError) as excinfo:
      df = spark.createDataFrame([mostly_empty_adapted])
    assert "Some of types cannot be determined" in str(excinfo.value)
    
    # ... but this works if we tell it the schema!
    schema = RowAdapter.to_schema(rows[0])
    df = spark.createDataFrame(
      [mostly_empty_adapted], schema=schema, verifySchema=False)

    EXPECTED_SCHEMA = [
      ('a', 'struct<__pyclass__:string,dtype:string,order:string,shape:array<bigint>,values:array<bigint>>'),
      ('b', 'map<string,struct<__pyclass__:string,dtype:string,order:string,shape:array<bigint>,values:array<bigint>>>'),
      ('c', 'array<struct<__pyclass__:string,dtype:string,order:string,shape:array<bigint>,values:array<double>>>'),
      ('d', 'struct<__pyclass__:string,_not_hidden:bigint,bar:string,foo:bigint>'),
      ('e', 'array<struct<__pyclass__:string,_not_hidden:bigint,bar:string,foo:bigint>>'),
      ('f', 'struct<__pyclass__:string,_not_hidden:bigint,meow:bigint>'),
      ('g', 'struct<__pyclass__:string>'),
      ('h', 'struct<i:bigint,j:bigint>'),
      ('id', 'bigint'),
      ('np_number', 'double'),
    ]
    assert df.dtypes == EXPECTED_SCHEMA
    
    # Check that pyspark retains the empty values in `mostly_empty`
    for colname in sorted(df.columns):
      values = df.select(colname).collect()
      assert len(values) == 1
      assert mostly_empty[colname] == values[0][colname]

    # ... and we can also read/write the empty-ish row!
    _check_serialization(spark, [mostly_empty], 'with_schema', schema=schema)


### Test With `attrs` Package

try:
  import attr
  import numpy as np
  
  # NB: Need these classes defined package-level
  
  @attr.s(eq=True)
  class AttrsUnslotted(object):
    foo = attr.ib(default="moof")
    bar = attr.ib(default=5)
    arr = attr.ib(default=np.array([1.]))
  
  @attr.s(slots=True, eq=True)
  class AttrsSlotted(object):
    foo = attr.ib(default="moof")
    bar = attr.ib(default=5)
    arr = attr.ib(default=np.array([1.]))

except ImportError:
  pass


def _test_attrs_objs(spark, objs, testname):
  from oarphpy.spark import RowAdapter

  schema = RowAdapter.to_schema(objs[0])
  rows = [RowAdapter.to_row(obj) for obj in objs]

  df = spark.createDataFrame(rows, schema=schema, verifySchema=False)

  # NB: both attrs-based examples have the same schema
  EXPECTED_SCHEMA = [
    ('__pyclass__', 'string'),
    ('arr', 'struct<__pyclass__:string,dtype:string,order:string,shape:array<bigint>,values:array<double>>'),
    ('bar', 'bigint'),
    ('foo', 'string'),
  ]
  assert df.dtypes == EXPECTED_SCHEMA

  _check_serialization(spark, objs, testname, schema=schema)

  return df

@skip_if_no_spark
def test_row_adapter_with_attrs():
  pytest.importorskip('attr')

  objs = [
    AttrsUnslotted(),
    AttrsUnslotted(foo="foom"),
    AttrsUnslotted(foo="123", arr=np.array([1., 2., 3.])),
  ]

  with testutil.LocalSpark.sess() as spark:
    df = _test_attrs_objs(spark, objs, 'test_row_adapter_with_attrs')

    EXPECTED = """
                                  __pyclass__                                                       arr  bar   foo
    0  oarphpy_test.test_spark.AttrsUnslotted  (oarphpy.spark.Tensor, float64, C, [3], [1.0, 2.0, 3.0])    5   123
    1  oarphpy_test.test_spark.AttrsUnslotted            (oarphpy.spark.Tensor, float64, C, [1], [1.0])    5  foom
    2  oarphpy_test.test_spark.AttrsUnslotted            (oarphpy.spark.Tensor, float64, C, [1], [1.0])    5  moof
    """
    _pandas_compare_str(df.orderBy('foo').toPandas(), EXPECTED)


@skip_if_no_spark
def test_row_adapter_with_slotted_attrs():
  pytest.importorskip('attr')

  objs = [
    AttrsSlotted(),
    AttrsSlotted(foo="foom"),
  ]

  with testutil.LocalSpark.sess() as spark:
    df = _test_attrs_objs(spark, objs, 'test_row_adapter_with_slotted_attrs')

    EXPECTED = """
                                __pyclass__                                             arr  bar   foo
    0  oarphpy_test.test_spark.AttrsSlotted  (oarphpy.spark.Tensor, float64, C, [1], [1.0])    5  foom
    1  oarphpy_test.test_spark.AttrsSlotted  (oarphpy.spark.Tensor, float64, C, [1], [1.0])    5  moof
    """
    _pandas_compare_str(df.orderBy('foo').toPandas(), EXPECTED)


### Extended Tests of `Tensor`

@skip_if_no_spark
def test_row_adapter_packed_numpy_arr():
  import sys
  import numpy as np
  from oarphpy.spark import RowAdapter
  from oarphpy.spark import TENSOR_AUTO_PACK_MIN_KBYTES
  
  with testutil.LocalSpark.sess() as spark:
    
    expect_unpacked = np.ones((5, 2))
    df = _check_serialization(
      spark, [expect_unpacked], 'unpacked_numpy_arr')
    
    N = int(TENSOR_AUTO_PACK_MIN_KBYTES * (2**10) / np.dtype(int).itemsize) + 1
    expect_packed = np.reshape(np.array(range(2 * N)), (2, N))
    schema = RowAdapter.to_schema(np.ones((5, 2)))
    dfp = _check_serialization(
      spark, [expect_packed], 'packed_numpy_arr', schema=schema)

    # import pdb; pdb.set_trace()
    # print()

