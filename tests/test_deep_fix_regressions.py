import importlib.util
from pathlib import Path
import unittest
import pandas as pd
spec=importlib.util.spec_from_file_location('splits_under_test',Path(__file__).resolve().parents[1]/'src/ngsatml/splits.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class SplitRegression(unittest.TestCase):
 def test_valid_boundaries(self):
  df=pd.DataFrame({'date':['2020-01-01','2020-02-01','2020-03-01','2020-04-01']})
  self.assertEqual(m.temporal_split(df,'date','2020-01-01','2020-02-01','2020-04-01').tolist(),['train','validation','gap','test'])
 def test_inverted_or_equal_boundaries(self):
  df=pd.DataFrame({'date':['2020-01-01']})
  for bounds in [('2020-01-01','2020-04-01','2020-02-01'),('2020-02-01','2020-01-01','2020-04-01'),('2020-01-01','2020-01-01','2020-04-01')]:
   with self.assertRaises(ValueError):m.temporal_split(df,'date',*bounds)
 def test_missing_boundary(self):
  with self.assertRaises(ValueError):m.temporal_split(pd.DataFrame({'date':['2020-01-01']}),'date','NaT','2020-02-01','2020-04-01')
if __name__=='__main__':unittest.main()
