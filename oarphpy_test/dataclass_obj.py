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

# Dataclass using Python 3 type annotations; used in test_spark.py.  We
# define this in a separate file so that Python 2 can ignore it

try:
  from dataclasses import dataclass
  @dataclass
  class DataclassObj:
    x: str
    y: float
except Exception:
  pass