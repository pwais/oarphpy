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


"""

Cooperative Disk Cache: A disk-based cache where the user is responsible
for running the eviction sweep process.




want to think:
 * can make this more nvme friendly?
 * what about if the storage is pmem / persistant memory?
 * what about if storage is blob store / s3?


YYYY / MM / DD / HH / MM / SS / key1
YYYY / MM / DD / HH / MM / SS / key2
YYYY / MM / DD / HH / MM / SS / key3

leading partition is just key *creation* time

to achieve multiple caches where there's priority about deleting one 
cache / channel before the other, just create a separate root(s) and
give them *more* quota.  b/c the caches with least quota will get
evicted first.
"""

import datetime
import os



class RootConfig(object):
  """Encapsulates configuration for one (of potentially many) 
  root directories in the cache space.
  """

  __slots__ = (
    'bucket_size_timedelta_seconds',
    'quota_max_used_bytes',
    'quota_min_free_bytes',
    'root_dirs',
  )

  def __init__(self):
    pass


class Client(object):

  def __init__(self, config=None):
    self._config = config or Config()

  def basedir_for_key(self, key, now=None):
    now = now or datetime.datetime.utcnow()

    # TODO validate key can be a directory name

    """
    mkdir-p a bucket based on `now` and create a dir for key as well
    """
  
  def update_key_time(self, key=None, path=None, now=None):
    now = now or datetime.datetime.utcnow()

    """
    maybe this just updates the key metadata file
    """


class Evictor(object):
  
  def __init__(self, config=None, now=None):
    self._config = config or Config()
    self._now = now or datetime.datetime.utcnow()


  def run(self, delete_older_than_t=None):
    self.update_all_metadata()
    self._do_evictions(delete_older_than_t=delete_older_than_t)

  def update_all_metadata(self):
    """
    for each root:
      for each key root directory:
        compute key metadata. this is mainly the size of items in the
        key dir.  do also re-compute the dir size, but perhaps only if
        the filesystem says files have changed?  maybe just checking
        changed info is as slow as getting file sizes

    """
    pass
  
  def _do_evictions(self, delete_older_than_t=None):
    """
    for each root:
      should_sweep_root = (
        is the root over quota?
        do we have delete_older_than_t?
      )

      compute key directories to expire.  this is either:
       * LRU / MRU: sort by key times
       * [key for key in keys 
            if key.creation_time before delete_older_than_t ]
      
      do em deletings!!
      
    """
    pass


###############################################################################
## Run Evictor Main Entrypoint

RUN_EVICTOR_DESC = """
TODO
"""

def create_arg_parser():
  import argparse

  parser = argparse.ArgumentParser(
                      description=RUN_EVICTOR_DESC,
                      formatter_class=argparse.RawDescriptionHelpFormatter)
  
  # Configuration
  parser.add_argument(
    '--config',
    help=('Use the cache config at this path as well as transitively all '
    'other configs referenced in this config.'))
  parser.add_argument(
    '--single-root-config',
    help=('Use the cache config at this path ONLY and ignore any other '
          'configs referenced in this one')
  parser.add_argument(
    '--now',
    help='Override the time of `now` to this ISO time string')
  parser.add_argument(
    '--delete-older-than',
    help=('Delete data older than this ISO time string, even if usage '
          'is under quota'))

  # Actions
  parser.add_argument(
    '--init-root',
    help=('Initialize a cache root in the given directory path by writing a '
          'default config'))
  parser.add_argument(
    '--only-update-metadata', default=False, action='store_true',
    help='Do not evict, just update all metadata on disk')
  parser.add_argument(
    '--only-report-usage', default=False, action='store_true',
    help='Just print a report about usage and quota to stdout')
  

  return parser

def run_evictor(args=None):
  if not args:
    parser = create_arg_parser()
    args = parser.parse_args()

  """
  try to find config from args or from env

  evictor = Evictor(config)


  """

if __name__ == '__main__':
  run_evictor()
