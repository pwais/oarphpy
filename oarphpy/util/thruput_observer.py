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
import time
from contextlib import contextmanager


class ThruputObserver(object):
  """A utility for measuring the runtime and throughput of a subroutine.
  Similar in spirit to `tqdm`, except `ThruputObserver`:
   * Tracks not just time but a size metric (e.g. memory) in bytes
   * Reports percentiles
   * Simply logs strings and is not terminal-interactive
  
  While `tqdm` is useful for notebooks, `ThruputObserver` seeks to be more
  useful for longer-running batch jobs.
  """
  
  def __init__(
      self,
      name='',
      log_on_del=False,
      only_stats=None,
      log_freq=100,
      n_total=None,
      n_total_chunks=None):
    self.n = 0
    self.num_bytes = 0
    self.ts = []
    self.name = name
    self.log_on_del = log_on_del
    self.only_stats = only_stats or []
    self.n_total = max(n_total, 1) if n_total is not None else None
    self.n_total_chunks = (
      max(n_total_chunks, 1) if n_total_chunks is not None else None)
    self._start = None
    self.__log_freq = log_freq
    self.__last_log = 0
  
  @contextmanager
  def observe(self, n=0, num_bytes=0):
    """
    NB: contextmanagers appear to be expensive due to object creation.
    Use ThurputObserver#{start,stop}_block() for <10ms ops. 
    FMI https://stackoverflow.com/questions/34872535/why-contextmanager-is-slow
    """

    self.start_block()
    yield
    self.stop_block(n=n, num_bytes=num_bytes)
  
  def start_block(self):
    self._start = time.time()
  
  def update_tallies(self, n=0, num_bytes=0, new_block=False):
    self.n += n
    self.num_bytes += num_bytes
    if new_block:
      self.stop_block()
      self.start_block()
  
  def stop_block(self, n=0, num_bytes=0):
    end = time.time()
    self.n += n
    self.num_bytes += num_bytes
    if self._start is not None:
      self.ts.append(end - self._start)
    self._start = None
  
  def maybe_log_progress(self, every_n=-1):
    if every_n >= 0:
      self.__log_freq = every_n
    if self.n >= self.__last_log + self.__log_freq:
      from oarphpy.util import log
      log.info("Progress for \n" + str(self))
      self.__last_log = self.n
        # Track last log because `n` may increase inconsistently
      if every_n == -1 and (self.n >= (1.7 * self.__log_freq)):
        self.__log_freq = int(1.7 * self.__log_freq)
          # Exponentially decay logging frequency. Don't decay quite as
          # fast as Vowpal Wabbit did, though.

  @staticmethod
  def union(thruputs):
    u = ThruputObserver()
    for t in thruputs:
      u += t
    return u

  @property
  def total_time(self):
    return sum(self.ts)

  def get_stats(self):
    import numpy as np
    from humanfriendly import format_size
    from humanfriendly import format_timespan

    total_time = self.total_time

    stats = [
      ('Thruput', ''),
      ('N thru', (self.n
                    if self.n_total is None
                    else '%s (of %s)' % (self.n, self.n_total))),
      ('N chunks', (len(self.ts)
                    if self.n_total_chunks is None
                    else '%s (of %s)' % (len(self.ts), self.n_total_chunks))),
      ('Total time', format_timespan(total_time) if total_time else '-'),
      ('Total thru', format_size(self.num_bytes)),
      ('Rate', 
        format_size(self.num_bytes / total_time) + ' / sec'
        if total_time else '-'),
      ('Hz', float(self.n) / total_time if total_time else '-'),
    ]
    percent_complete = None
    if self.n_total is not None:
      percent_complete = 100. * float(self.n) / self.n_total
    elif self.n_total_chunks is not None:
      percent_complete = 100. * float(len(self.ts)) / self.n_total_chunks
    if percent_complete is not None:
      eta_sec = (
        (100. - percent_complete) * 
        (total_time / (percent_complete + 1e-10)))
      stats.extend([
        ('Progress', ''),
        ('Percent Complete', percent_complete),
        ('Est. Time To Completion', format_timespan(eta_sec)),
      ])
    if len(self.ts) >= 2:
      format_t = lambda t: format_timespan(t, detailed=True)
      stats.extend([
        ('Latency (per chunk)', ''),
        ('Avg', format_t(np.mean(self.ts))),
        ('p50', format_t(np.percentile(self.ts, 50))),
        ('p95', format_t(np.percentile(self.ts, 95))),
        ('p99', format_t(np.percentile(self.ts, 99))),
      ])
    if self.only_stats:
      stats = tuple(
        (name, value)
        for name, value in stats
        if name in self.only_stats
      )
    return stats

  def __iadd__(self, other):
    self.n += other.n
    self.num_bytes += other.num_bytes
    self.ts.extend(other.ts)
    return self

  def __str__(self):
    import tabulate
    stats = self.get_stats()
    summary = tabulate.tabulate(stats)
    if self.name:
      prefix = '%s [Pid:%s Id:%s]' % (self.name, os.getpid(), id(self))
      summary = prefix + '\n' + summary
    return summary
  
  def __del__(self):
    if self.log_on_del:
      self.stop_block()

      from oarphpy.util import create_log
      log = create_log()
      log.info('\n' + str(self) + '\n')
  
  @staticmethod
  def monitoring_tensor(name, tensor, **observer_init_kwargs):
    """Monitor the size of the given tensorflow `Tensor` and record a
    text TF Summary with the contents of this ThruputObserver."""

    class Observer(object):
      def __init__(self, dtype_size_bytes):
        self.observer = ThruputObserver(name=name, **observer_init_kwargs)
        self.dtype_size_bytes = dtype_size_bytes
      def __call__(self, t_shape):
        import numpy as np
        n = t_shape[0]
        num_bytes = np.prod(t_shape) * self.dtype_size_bytes
        self.observer.stop_block(n=n, num_bytes=num_bytes)
        self.observer.maybe_log_progress()
        
        # Tensorboard is very picky about wanting Markdown :P
        import tabulatehelper as th
        stats = self.observer.get_stats()
        out = th.md_table(stats, headers=[name])

        self.observer.start_block()
        return out
    
    import tensorflow as tf
    obs_str_tensor = tf.compat.v1.py_func(
              Observer(tensor.dtype.size), [tf.shape(tensor)], tf.string)
    tf.summary.text(name + '/ThruputObserver', obs_str_tensor)
    return obs_str_tensor
  
  @staticmethod
  def wrap_func(func, **observer_init_kwargs):
    """Decorate `func` and observe a block on each call"""
    class MonitoredFunc(object):
      def __init__(self, func, observer_init_kwargs):
        self.func = func
        self.observer = ThruputObserver(**observer_init_kwargs)
      def __call__(self, *args, **kwargs):
        from oarphpy.util.misc import get_size_of_deep
        self.observer.start_block()
        ret = self.func(*args, **kwargs)
        self.observer.stop_block(n=1, num_bytes=get_size_of_deep(ret))
        self.observer.maybe_log_progress()
        return ret
    return MonitoredFunc(func, observer_init_kwargs)
