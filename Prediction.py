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

df = pd.read_csv('AAPL.csv')
train_df = df[:int(len(df)*.7)]
val_df = df[int(len(df)*.7):int(len(df)*.9)]
test_df = df[int(len(df)*.9):]
MAX_EPOCHS = 20

class StockGenerator():
    def __init__(self, input_width, label_width, shift,
                  train_df=train_df, val_df=val_df, test_df=test_df, label_columns=None):
        
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
    
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        
        self.column_indices = {
            name: i for i, name in enumerate(train_df.columns)
        }
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        
        self.total_window_size = input_width + shift
        
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]
        
        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
        
    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column names: {self.label_columns}'
        ])       
        
    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1
            )
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        
        return inputs, labels
    
    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32            
        )
        
        ds = ds.map(self.split_window)
        return ds
    
    def plot(self, model=None, plot_col='Close', max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(3, 1, n+1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                        label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index-1],
                edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                    marker='X', edgecolors='k', label='Predictions',
                    c='#ff7f0e', s=64)

        if n == 0:
            plt.legend()

        plt.xlabel('Time [d]')
        plt.ylabel('Closing Price')
        plt.show()

            
    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)
    
    @property
    def test(self):
        return self.make_dataset(self.test_df)
    
    @property
    def example(self):
        result = getattr(self, '_example', None)
        if result is None:
            result = next(iter(self.train))
            self._example = result
        return result
    
class Baseline(tf.keras.Model):
    def __init__(self, label_index=None):
        super().__init__()
        self.label_index = label_index
        
    def call(self, inputs):
        if self.label_index is None:
            return inputs
        result = inputs[:,:,self.label_index]
        return result[:,:,tf.newaxis]
         
    
def compile_and_fit(model, window, patience=2):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')
    
    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])
    
    history = model.fit(window.train, epochs=MAX_EPOCHS,
                        validation_data=window.val,
                        callbacks=[early_stopping])
    
    return history

if __name__ == '__main__':
    
    #print(df.head())
    #Testing will be done on a 70%, 20%, 10% where
        #70% = training
        #20% = validation
        #10% = test sets
    single_step_window = StockGenerator(input_width=60, label_width=60,shift=1,label_columns=['Close'])
    print(single_step_window)
    #print(df.columns)
    

    column_indices = {name: i for i, name in enumerate(df.columns)}
    


    train = single_step_window.train
    val = single_step_window.val
    test = single_step_window.test
    
    #Normalize data
    train_mean = single_step_window.train_df.mean()
    train_std = single_step_window.train_df.std()

    train_df = (single_step_window.train_df - train_mean)/train_std
    val_df = (single_step_window.val_df - single_step_window.train_df) / train_std
    test_df = (single_step_window.test_df - train_mean) / train_std
    
    baseline = Baseline(label_index=column_indices['Close'])
    baseline.compile(loss=tf.losses.MeanSquaredError(),
                     metrics=[tf.metrics.MeanAbsoluteError()])
    val_performance = {}
    performance = {}
    val_performance['Baseline'] = baseline.evaluate(single_step_window.val)
    performance['Baseline'] = baseline.evaluate(single_step_window.test, verbose=0)
    '''
    for example_inputs, example_labels in single_step_window.train.take(1):
        print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
        print(f'Labels shape (batch, time, features): {example_labels.shape}')
    print('Input Shape:', single_step_window.example[0].shape)
    print('Output Shape:', baseline(single_step_window.example[0]).shape)
    '''
    single_step_window.plot(baseline)

    dense = tf.keras.Sequential([
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=64,activation='relu'),
        tf.keras.layers.Dense(units=1)
    ])
    
    history = compile_and_fit(dense, single_step_window)
    
    val_performance['Dense'] = dense.evaluate(single_step_window.val)
    performance['Dense'] = dense.evaluate(single_step_window.test, verbose=0)
    
    single_step_window.plot(dense)
