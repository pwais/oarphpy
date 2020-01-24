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

from oarphpy import util


def hash_to_rbg(x, s=0.8, v=0.8):
  """Given some value `x` (integral types work best), hash `x`
  to an `(r, g, b)` color tuple using a hue based on the hash
  and the given `s` (saturation) and `v` (lightness)."""

  import colorsys
  import sys
  import numpy as np

  # NB: ideally we just use __hash__(), but as of Python 3 it's not stable,
  # so we use a trick based upon the Knuth hash
  import hashlib
  h_i = int(hashlib.md5(str(x).encode('utf-8')).hexdigest(), 16)
  h = (h_i % 2654435769) / 2654435769.
  rgb = 255 * np.array(colorsys.hsv_to_rgb(h, s, v))
  return tuple(rgb.astype(int).tolist())


def img_to_data_uri(img, format='jpg', jpeg_quality=75):
  """Given a numpy array `img`, return a `data:` URI suitable for use in 
  an HTML image tag."""

  from io import BytesIO
  out = BytesIO()

  import imageio
  kwargs = dict(format=format)
  if format == 'jpg':
    kwargs.update(quality=jpeg_quality)
  imageio.imwrite(out, img, **kwargs)

  from base64 import b64encode
  data = b64encode(out.getvalue()).decode('ascii')
  
  from six.moves.urllib import parse
  data_url = 'data:image/png;base64,{}'.format(parse.quote(data))
  
  return data_url


def get_hw_in_viewport(img_hw, viewport_hw):
  vh, vw = viewport_hw
  h, w = img_hw
  if h > vh:
    rescale = float(vh) / h
    h = rescale * h
    w = rescale * w
  if w > vw:
    rescale = float(vw) / w
    h = rescale * h
    w = rescale * w
  return int(h), int(w)


def img_to_img_tag(
    img,
    display_viewport_hw=None, # E.g. (1000, 1000)
    image_viewport_hw=None,   # E.g. (1000, 1000)
    format='jpg',
    jpeg_quality=75):

  if image_viewport_hw is not None:
    th, tw = get_hw_in_viewport(img.shape[:2], image_viewport_hw)
    th = max(1, th)
    tw = max(1, tw)
    import cv2
    img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_NEAREST)
  
  dh, dw = img.shape[:2]
  if display_viewport_hw is not None:
    dh, dw = get_hw_in_viewport((dh, dw), display_viewport_hw)

  src = img_to_data_uri(img, format=format, jpeg_quality=jpeg_quality)
  TEMPLATE = """<img src="{src}" height="{dh}" width="{dw}" />"""
  return TEMPLATE.format(src=src, dh=dh, dw=dw)


def _unpack_pyspark_row(r):
  """Unpack a `pyspark.sql.Row` that contains a single value."""
  return r[0]
    # NB: 0 as in 0th column; pyspark.sql.Row provides indexing
    # for syntactic sugar


def df_histogram(spark_df, col, num_bins):
  """Compute and return a histogram of `bins` of the values in the column
  named `col` in spark Dataframe `spark_df`.  Return type is designed
  to match `numpy.histogram()`.
  """
  import numpy as np
  assert num_bins >= 1
  col_val_rdd = spark_df.select(col).rdd.map(_unpack_pyspark_row)
  buckets, counts = col_val_rdd.histogram(num_bins)
  return np.array(counts), np.array(buckets)


def save_bokeh_fig(fig, dest, title=None):
  from bokeh import plotting
  if not title:
    title = os.path.split(dest)[-1]
  plotting.output_file(dest, title=title, mode='inline')
  plotting.save(fig)
  util.log.info("Wrote Bokeh figure to %s" % dest)


class HistogramWithExamplesPlotter(object):
  """Create and return a Bokeh plot depicting a histogram of a single column in
  a Spark DataFrame.  Clicking on a bar in the histogram will interactively
  show examples from that bucket.
  
  `SUB_PIVOT_COL` - Optionally choose an additional dimension of the data and
  include histograms of the data pivoted by that dimension. For example, if we
  are histogramming the "height" dimension over a population, and we set 
  `SUB_PIVOT_COL` to the "gender" column, then we'll get a histogram of height
  over ALL genders as well as height histograms for each distinct value in the
  "gender" column.
  
  The user can override how examples are displayed; subclasses can override
  `HistogramWithExamplesPlotter::display_bucket()`

  See `HistogramWithExamplesPlotter::run()`.
  """

  ## Core Params
  NUM_BINS = 50

  SUB_PIVOT_COL = None

  WIDTH = 1000
    # Bokeh's plots (especially in single-column two-row layout we use) work
    # best with a fixed width

  ## Plotting params
  TITLE = None  # By default use DataFrame Column name

  def display_bucket(self, sub_pivot, bucket_id, irows):
    import itertools
    rows_str = "<br />".join(str(r) for r in itertools.islice(irows, 5))
    TEMPLATE = """
      <b>Pivot: {spv} Bucket: {bucket_id} </b> <br/>
      {rows}
      <br/> <br/>
    """
    disp = TEMPLATE.format(spv=sub_pivot, bucket_id=bucket_id, rows=rows_str)
    return bucket_id, disp

  def _build_data_source_for_sub_pivot(self, spv, df, col):
    import numpy as np
    import pandas as pd

    util.log.info("... building data source for %s ..." % spv)

    if spv == 'ALL':
      sp_src_df = df
    else:
      sp_src_df = df.filter(df[self.SUB_PIVOT_COL] == spv)
      
    util.log.info("... histogramming %s ..." % spv)
    hist, edges = df_histogram(sp_src_df, col, self.NUM_BINS)

    # Use this Pandas Dataframe to serve as a bokeh data source
    # for the plot
    sp_df = pd.DataFrame(dict(
      count=hist, proportion=hist / np.sum(hist),
      left=edges[:-1], right=edges[1:],
    ))
    sp_df['legend'] = str(spv)

    from bokeh.colors import RGB
    sp_df['color'] = RGB(*hash_to_rbg(spv))
    
    util.log.info("... display-ifying examples for %s ..." % spv)
    def get_display():
      # First, Re-bucket each row using what (in SQL) looks like a CASE-WHEN
      # statement:
      #     SELECT
      #     CASE
      #     WHEN 0 <= val AND val < 10 THEN 0
      #     WHEN 10 <= val AND val < 20 THEN 10
      #     ...
      #     END AS bucket, ...
      # We use the DataFrame API to construct the query because it's easier.
      # Spark will compile it to native code on-the-fly.
      from pyspark.sql import functions as F
      col_def = None
      buckets = list(zip(edges[:-1], edges[1:]))
      for bucket_id, (lo, hi) in enumerate(buckets):
        # The last spark histogram bucket is closed, but we want open
        if bucket_id == len(buckets) - 1:
          hi += 1e-9
        args = (
          (sp_src_df[col] >= float(lo)) & (sp_src_df[col] < float(hi)),
          bucket_id
        )
        if col_def is None:
          col_def = F.when(*args)
        else:
          col_def = col_def.when(*args)
      col_def = col_def.otherwise(-1)
      df_bucketed = sp_src_df.withColumn('au_plot_bucket', col_def)
      
      # Second, we collect chunks of rows partitioned by bucket ID so that we
      # can run our display function in parallel over buckets.
      bucketed_chunks = df_bucketed.rdd.groupBy(lambda r: r.au_plot_bucket)
      bucket_disp = bucketed_chunks.map(
                      lambda b_irows: 
                        self.display_bucket(spv, b_irows[0], b_irows[1]))
      bucket_to_disp = dict(bucket_disp.collect())
      
      # Finally, return a column of display strings ordered by buckets so that
      # we can add this column to the output histogram DataFrame.
      return [
        bucket_to_disp.get(b, '')
        for b in range(len(buckets))
      ]
    sp_df['display'] = get_display()
    return sp_df

  def run(self, df, col):
    """Compute histograms and return the final plot.

    Args:
      df (pyspark.sql.DataFrame): Read from this DataFrame.  The caller may
        want to `cache()` the DataFrame as this routine will do a variety of
        random reads and aggregations on the data.
      col (str): The x-axis for the computed histogram shall this this column
        in `df` as the chosen metric.  Spark automatically ignores nulls and
        nans.

    Returns:
      bokeh layout object with a plot.
    """
    import pyspark.sql
    assert isinstance(df, pyspark.sql.DataFrame)
    assert col in df.columns
    
    util.log.info("Plotting histogram for %s of %s ..." % (col, df))
    
    sub_pivot_values = ['ALL']
    if self.SUB_PIVOT_COL:
      distinct_rows = df.select(self.SUB_PIVOT_COL).distinct()

      sub_pivot_values.extend(
        sorted(
          distinct_rows.rdd.map(_unpack_pyspark_row).collect()))
    
    ## Compute a data source Pandas Dataframe for every sub-pivot
    spv_to_panel_df = dict(
      (spv, self._build_data_source_for_sub_pivot(spv, df, col))
      for spv in sub_pivot_values)
    
    ## Make the plot
    from bokeh import plotting
    fig = plotting.figure(
            title=self.TITLE or col,
            tools='tap,pan,wheel_zoom,box_zoom,reset',
            width=self.WIDTH,
            x_axis_label=col,
            y_axis_label='Count')
    for spv in sub_pivot_values:
      plot_src = spv_to_panel_df[spv]
      from bokeh.models import ColumnDataSource
      plot_src = ColumnDataSource(plot_src)
      r = fig.quad(
        source=plot_src, bottom=0, top='count', left='left', right='right',
        color='color', fill_alpha=0.5,
        hover_fill_color='color', hover_fill_alpha=1.0,
        legend='legend')
      from bokeh.models import HoverTool
      fig.add_tools(
        HoverTool(
          renderers=[r],
            # For whatever reason, adding a hover tool for each quad
            # makes the interface dramatically faster in the browser
          mode='vline',
          tooltips=[
            ('Sub-pivot', '@legend'),
            ('Count', '@count'),
            ('Proportion', '@proportion'),
            ('Value of %s' % col, '@left'),
          ]))

      fig.legend.click_policy = 'hide'

    ## Add the 'show examples' tool and div
    from bokeh.models.widgets import Div
    ctxbox = Div(width=self.WIDTH, text=
        "Click on a histogram bar to show examples.  "
        "Click on the legend to show/hide a series.")


    from bokeh.models import TapTool
    taptool = fig.select(type=TapTool)

    from bokeh.models import CustomJS
    taptool.callback = CustomJS(
      args=dict(ctxbox=ctxbox),
      code="""
        var idx = cb_data.source.selected['1d'].indices[0];
        ctxbox.text='' + cb_data.source.data.display[idx];
      """)

    from bokeh.layouts import column
    layout = column(fig, ctxbox)
    return layout
