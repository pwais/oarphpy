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