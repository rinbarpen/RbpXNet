import os
import pandas as pd
import unittest
import numpy as np
import torch

class TestPandasCSVWriter(unittest.TestCase):
  def setUp(self):
    self.filename = 'tests/test_results.csv'
    self.df = pd.DataFrame()

  def tearDown(self):
    if os.path.exists(self.filename):
      os.remove(self.filename)

  def test_write_header(self):
    headers = ['Epoch', 'Loss', 'Accuracy']
    self.df = self.df[headers]
    self.df.to_csv(self.filename, index=False)

    # Verify the header was written correctly
    df_read = pd.read_csv(self.filename)
    self.assertEqual(list(df_read.columns), headers)

  def test_write(self):
    self.df['Epoch'] = [1, 2, 3]
    self.df['Loss'] = np.array([0.5, 0.3, 0.8], dtype=float)
    self.df['Accuracy'] = torch.tensor([0.9, 0.8, 0.8])
    self.df.to_csv(self.filename, index=False, mode='a')

    # Verify the data was written correctly
    df_read = pd.read_csv(self.filename)
    self.assertEqual(list(df_read['Epoch']), [1, 2, 3])
    self.assertTrue(np.allclose(df_read['Loss'], [0.5, 0.3, 0.8]))
    self.assertTrue(np.allclose(df_read['Accuracy'], [0.9, 0.8, 0.8]))
