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

from oarphpy import plotting as pl
from oarphpy import util

from oarphpy_test.testutil import LocalSpark
from oarphpy_test.testutil import skip_if_no_spark


def test_hash_to_rgb():
  pytest.importorskip("numpy")
  assert pl.hash_to_rbg('moof') == (145, 40, 204)
  assert pl.hash_to_rbg(5) == (186, 204, 40)
  assert pl.hash_to_rbg('moof1') != pl.hash_to_rbg('moof')


@skip_if_no_spark
def test_spark_histogram():
  np = pytest.importorskip("numpy")

  with LocalSpark.sess() as spark:
    from pyspark.sql import Row
    df = spark.createDataFrame([
      Row(a=a, b=a * a) for a in range(101)
    ])

    def check(ahist, ehist, aedges, eedges):
      np.testing.assert_array_equal(ahist, ehist)
      np.testing.assert_array_equal(aedges, eedges)

    hist, edges = pl.df_histogram(df, 'a', 1)
    check(
      hist,   np.array([101]),
      edges,  np.array([0., 100.]))

    hist, edges = pl.df_histogram(df, 'a', 2)
    check(
      hist,   np.array([50, 51]),
      edges,  np.array([0., 50., 100.]))

    hist, edges = pl.df_histogram(df, 'b', 4)
    check(
      hist,   np.array([50, 21, 16, 14]),
      edges,  np.array([0, 2500, 5000, 7500, 10000]))


@skip_if_no_spark
def test_histogram_with_examples():
  pytest.importorskip('bokeh')
  np = pytest.importorskip('numpy')

  from oarphpy import util
  from oarphpy_test.testutil import get_fixture_path

  TEST_TEMPDIR = '/tmp/oarphpy/test_histogram_with_examples'
  util.cleandir(TEST_TEMPDIR)
  
  def check_fig(fig, fixture_name):
    actual_path = os.path.join(TEST_TEMPDIR, 'actual_' + fixture_name)
    util.log.info("Saving actual plot to %s" % actual_path)
    pl.save_bokeh_fig(fig, actual_path, title=fixture_name)
    
    actual_png_path = actual_path.replace('html', 'png')
    util.log.info("Saving screenshot of plot to %s" % actual_png_path)
    from bokeh.io import export_png
    export_png(fig, actual_png_path)

    expected_path = get_fixture_path(fixture_name)
    expected_png_path = expected_path.replace('html', 'png')

    # Compare using PNGs because writing real selenium tests is too much effort
    # for the value at this time.  We tried comparing the raw HTML but bokeh
    # appears to write non-deterministically and/or includes timestamped
    # material.
    import imageio
    actual = imageio.imread(actual_png_path)
    expected = imageio.imread(expected_png_path)
    util.log.info('Comparing against expected at %s' % expected_png_path)

    np.testing.assert_array_equal(
      actual, expected,
      err_msg=(
        "Page mismatch, actual %s != expected %s, check HTML and PNGs" % (
          actual_path, expected_path)))

  with LocalSpark.sess() as spark:
    
    # A simple table:
    # +------+------+---+                                                             
    # |mod_11|square|  x|
    # +------+------+---+
    # |     0|     0|  0|
    # |     1|     1|  1|
    # |     2|     4|  2|
    # |     3|     9|  3|
    #    ...
    # +------+------+---+
    from pyspark.sql import Row
    df = spark.createDataFrame([
      Row(x=x, mod_11=int(x % 11), square=x*x)
      for x in range(101)
    ])

    ### Check basic plotting
    plotter = pl.HistogramWithExamplesPlotter()
    fig = plotter.run(df, 'x')
    check_fig(fig, 'test_histogram_with_examples_1.html')

    ### Check plotting with custom example plotter
    class PlotterWithMicroFacet(pl.HistogramWithExamplesPlotter):
      SUB_PIVOT_COL = 'mod_11'
      NUM_BINS = 25

      def display_bucket(self, sub_pivot, bucket_id, irows):
        rows_str = "<br />".join(
            "x: {x} square: {square} mod_11: {mod_11}".format(**row.asDict())
            for row in sorted(irows, key=lambda r: r.x))
        TEMPLATE = """
          <b>Pivot: {spv} Bucket: {bucket_id} </b> <br/>
          {rows}
          <br/> <br/>
        """
        disp = TEMPLATE.format(
                  spv=sub_pivot,
                  bucket_id=bucket_id,
                  rows=rows_str)
        return bucket_id, disp
    
    plotter = PlotterWithMicroFacet()
    fig = plotter.run(df, 'square')
    check_fig(fig, 'test_histogram_with_examples_2.html')
