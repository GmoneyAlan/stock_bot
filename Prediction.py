import os
import datetime

import IPython 
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf 

class StockGenerator():
    def __init__(self, input_width, label_width, shift,
                  train_df=train_df, val_df=val_df, test_df=test_df, label_columns=None):
        
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
    
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.label_columns_indices = {i for i, name in enumerate(label_columns)}
        
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        
        self.total_window_size = input_width + shift
        self.input_slice = np.arrange(self.total_window_size)[self.input_slice]
        
        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arrange(self.total_window_size)[self.labels_slice]
        
    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column names: {self.label_columns}'
        ])
        
    
    
df = pd.read_csv('AAPL.csv')
#print(df.head())
