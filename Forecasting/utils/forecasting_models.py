from tensorflow import keras
import pandas as pd
from sklearn.base import TransformerMixin
from tqdm import tqdm
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import metrics
from tensorflow.keras.models import load_model

from sklearn.model_selection import train_test_split
import os
from os import mkdir, makedirs
from os.path import isdir
from glob import glob
from os import chdir
from os.path import splitext, split
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.layers import DenseFlipout
from statsmodels.tsa.api import ARIMA, VARMAX, ExponentialSmoothing
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
import statsmodels.api as sm
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.api import OLS
from adjdatatools.preprocessing import AdjustedScaler
import warnings

warnings.simplefilter('ignore')  ##


class preprocess():

    def __init__(self,
                 data_path,
                 drop=[],
                 collapse=['Территория'],
                 thresh=0.6,
                 y_col=['HC', 'HC%'],  # define y variable, i.e., what we want to predict
                 shift=1,  # date-shifting to avoid data leakage
                 ):
        self.data_path = data_path
        self.drop = drop
        self.thresh = thresh
        self.collapse = collapse
        self.y_col = y_col
        self.shift = shift

    def load_data(self):
        """
        loads dataset
        """
        try:
            df = pd.read_csv(self.data_path)
        except Exception:
            print(f'Invalid path: {self.data_path}')

        # Turn cyrillic symbols to latin ones
        df = df.rename({'НС': 'HC',
                        'НС%': 'HC%'}, axis=1)
        return (df)


    def clean_data(self, df):
        """
        cleans dataset
        drop - list of columns to delete
        collapse - columns to use as unique identifier
        """
        for regex in self.drop:
            df = df[df.columns.drop(list(df.filter(regex=regex)))]
        print('[INFO] data cleaned')

        for regex in self.collapse:
            df = df[df.columns.drop(list(df.filter(regex=regex)))]
        return df

    def set_index(self, df):
        df.set_index(keys='Год', drop=True, append=False, inplace=True)
        df.index.name = None
        return df

    def fill_missing(self, df):
        print('[INFO] imputing NA')
        df = df.dropna(thresh=df.shape[0] * self.thresh, how='all', axis=1)
        num_cols = df._get_numeric_data().columns
        imp = IterativeImputer(max_iter=10, random_state=0)
        imp.fit(df[num_cols])
        IterativeImputer(random_state=0)
        df[num_cols] = imp.transform(df[num_cols])
        self.missing_values(df)
        return (df)

    def missing_values(self, df):
        """
        tracks missing values
        """
        mis_val = df.isnull().sum()
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        mis_val_table_ren_columns = mis_val_table.rename(
            columns={0: 'Missing Values', 1: '% of Total Values'})
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
            '% of Total Values', ascending=False).round(1)
        print("[INFO] the dataframe has " + str(df.shape[1]) + " columns in total and " +
              str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        return mis_val_table_ren_columns

    def scale(self, df):
        """
        scale df
        ensure entire df is strictly positive 
        
        """
        df_scaler = AdjustedScaler()
        df_scaler.fit(df)
        scaled_df = df_scaler.transform(df)  # Scale and map to a (0;1) interval
        scaled_df = scaled_df.where(scaled_df > 0, 1e-5)
        scaled_df = scaled_df.where(scaled_df < 1, 1)
        print('[INFO] data scaled')
        return scaled_df, df_scaler

    def reindex(self, df):
        """
        reindexes the dataframe to eliminate leakage
        
        """
        df = df.where(df > 0, 1e-10)  # strictly positive
        df.index = pd.DatetimeIndex(df.index, freq='AS')  # enables date manipulation
        hist = df.copy()  # historical data
        y = df[self.y_col]
        hist.index = df.index.shift(-self.shift)
        hist = hist.shift(self.shift).dropna(axis=0)
        y.index = df.index.shift(self.shift)
        y = y.shift(-self.shift).dropna(axis=0)

        df = hist.copy()
        df[self.y_col] = y.values
        return hist, df

    def create_lagged_variable(self, data, window_size=3):
        """
        Creates <windows_size> lagged target-variable dataset
        :param targets:
        :param window_size:
        :return:
        """
        for i in range(1, window_size + 1):
            for target in self.y_col:
                data[f'lagged_{target}_{i}'] = data.loc[:, target].shift(i)

        return data

    def clip_time_series_data(self, data: pd.DataFrame, dates_column:str = 'Год', clip_intervals: tuple = (2006, 2021)):
        data[dates_column]  = pd.to_datetime(data[dates_column])
        return data[(data[dates_column].dt.year >= clip_intervals[0]) & (data[dates_column].dt.year <= clip_intervals[1])]

class ES(preprocess):
    """
    exponential smoothing layer
    
    """

    def __init__(self,
                 internals_path='internals',
                 alpha=0.05,
                 loc='United Kingdom'):

        self.internals_path = internals_path
        self.alpha = alpha
        self.loc = loc


    def get_internals(self, x_i, dates):
        fit = ExponentialSmoothing(x_i, trend='add', dates=dates).fit()
        internals = pd.DataFrame(np.c_[x_i, fit.level, fit.slope, fit.fittedvalues],
                                 columns=[r'$y_t$', r'$l_t$', r'$b_t$', r'$\hat{y}_t$'], index=dates)
        return internals

    def es(self, df):

        internals_dict = dict()
        if isdir(self.loc + '/' + str(self.alpha) + '/' + self.internals_path) == False:
            makedirs(f"./forecasting_results/{self.loc}/{self.alpha}/{self.internals_path}")
            # get and save internals
            for col in df.columns.to_list():
                i = df.columns.to_list().index(col)
                ind = str(i).zfill(2)
                vars()["int_df_{}".format(ind)] = self.get_internals(df.iloc[:, i], df.index)
                vars()["int_df_{}".format(ind)].to_pickle("./forecasting_results/" + self.loc + '/' + str(
                    self.alpha) + '/' + self.internals_path + "/int_df_{}".format(ind) + ".pkl")
                internals_dict["int_df_{}".format(ind)] = vars()["int_df_{}".format(ind)]
            print('[INFO] internals executed')
        else:
            # load internals
            chdir(self.loc + '/' + str(self.alpha) + '/' + self.internals_path + "/")
            for file in glob("*df*"):
                vars()[splitext(file)[0]] = pd.read_pickle(file)
                internals_dict[splitext(file)[0]] = vars()[splitext(file)[0]]
            print('[INFO] internals loaded')

            chdir("..")
            chdir("..")
            chdir('..')

        return internals_dict

    def deTS(sel, scaled_df, internals_dict):
        """
        detrend
        
        """
        # internals for scaled_df
        df_level = pd.DataFrame(index=scaled_df.index)

        for i in sorted(internals_dict.keys()):
            df_level[i] = internals_dict[i]['$l_t$']

        # smooth for scaled_df
        es_scaled = scaled_df.values - df_level.values  # exponentially smoothed (extracted trend)
        es_scaled_df = pd.DataFrame(es_scaled, columns=scaled_df.columns, index=scaled_df.index)

        return es_scaled_df, df_level  # exponentially smoothed and scaled


class naive_forecaster():
    """
    class that implements a naive hypothesis that future value will be just like previous one
    """

    def __init__(self, loc, results_path, alpha):
        self.loc = loc
        self.results_path = results_path
        self.alpha = alpha

    def forecast_naive(self, df):
        original = df.copy()
        shifted = df.copy()
        shifted.index = df.index.shift(1)

        result = pd.DataFrame({'HC_true' : original.values,
                               'HC_pred' : shifted.values},
                              index = shifted.index)
        return result

    def save_forecast(self, forecast_df, forecast_filename):
        makedirs('./forecasting_results/' + self.loc + '/' + str(self.alpha) + '/' + self.results_path, exist_ok=True)
        forecast_df.to_pickle(
            './forecasting_results/' + self.loc + '/' + str(self.alpha) + '/' + self.results_path + forecast_filename)

class lstm():
    """
    deep learning layer
    this layer is also bundled with the postprocessing methods
    
    """

    def __init__(self,
                 valid_test=0.75,
                 y_col=['HC', 'HC%'],  # predictand(s)
                 n_input_train=3,
                 b_size_train=1,
                 n_input_valid=3,
                 b_size_valid=1,
                 lstm_size=50,
                 activation='relu',
                 optimizer='adam',
                 loss='mse',
                 epochs = 30,
                 runs = 100,
                 alpha=  0.05,
                 verbose = 1,
                 loc='United Kingdom',
                 results_path='results/mes_lstm/',
                 pretrained_model_path = None,
                 stateful = True,
                 early_stopping = 5,
                 learning_rate = 1e-3
                 ):

        self.valid_test = valid_test
        self.y_col = y_col
        self.n_input_train = n_input_train
        self.b_size_train = b_size_train
        self.n_input_valid = n_input_valid
        self.b_size_valid = b_size_valid
        self.lstm_size = lstm_size
        self.activation = activation
        self.optimizer = optimizer
        self.loss = loss
        self.epochs = epochs
        self.runs = runs
        self.alpha = alpha
        self.results_path = results_path
        self.loc = loc
        self.verbose = verbose
        self.pretrained_model_path = pretrained_model_path
        self.stateful = stateful
        self.early_stopping = early_stopping
        self.learning_rate = learning_rate

    def split(self, es_scaled_df):
        length = self.n_input_train  # length of time series data to perform forecast

        # Last 'length' observations preserved for testing.
        test = es_scaled_df[-length:]
        # All data beyond that is used for training
        train = es_scaled_df[:-length]

        # split x and y only
        x_train = train.drop(self.y_col, axis=1).copy()
        y_train = train[self.y_col].copy()

        x_test = test.drop(self.y_col, axis=1).copy()
        y_test = test[self.y_col].copy()

        return x_train, y_train, x_test, y_test


    def pi_model(self, x_train, y_train, x_test, y_test):
        """
        returns model prediction intervals
        """
        results_path = f"./forecasting_results/{self.loc}/{self.alpha}/{self.results_path}"
        makedirs(results_path, exist_ok=True)

        # Define window size and prepare data
        train_val_losses = []

        # Append test data with some train observations to start prediction on
        x_test = pd.concat([x_train.iloc[-self.n_input_train:, :], x_test])
        y_test = pd.concat([y_train.iloc[-self.n_input_train:, :], y_test])

        train_generator = TimeseriesGenerator(x_train.values, y_train.values, length=self.n_input_train,
                                              batch_size=self.b_size_train)

        test_generator = TimeseriesGenerator(x_test.values, y_test.values, length=self.n_input_valid,
                                             batch_size=self.b_size_valid)


        # Initialize model
        model = Sequential()
        model.add(LSTM(self.lstm_size,
                       activation=self.activation,
                       return_sequences=False,
                       stateful=self.stateful,
                       batch_input_shape=(self.b_size_train, self.n_input_train, x_train.shape[1])
                       ))
        model.add(tfp.layers.DenseFlipout(len(self.y_col)))
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer,
                      loss=self.loss)
        print("Модель скомпилирована:", model.summary())

        if self.pretrained_model_path:
            # Load pre_trained model
            try:
                model.load_weights(self.pretrained_model_path)

            except Exception as exc:
                print(exc)
                print(f'Specified model: {self.pretrained_model_path} path is invalid. Initialized model with no pre-trained weights')


        early_stopping = EarlyStopping(monitor='val_loss',
                                       patience=self.early_stopping,
                                       restore_best_weights=True)

        checkpoint = ModelCheckpoint(f'{results_path}/best_model.hdf5',
                                     monitor='val_loss',
                                     save_best_only=True,
                                     mode='min')

        history = model.fit_generator(train_generator,
                                      validation_data = test_generator,
                                      epochs = self.epochs,
                                      callbacks=[early_stopping, checkpoint],
                                      verbose = self.verbose
                                      )


        train_val_losses = {'val_loss': history.history['val_loss'],
                            'train_loss': history.history['loss']}

        # saving lossess history information in pkl format
        pd.DataFrame(train_val_losses).to_pickle(f'{results_path}/train_val_losses.pkl')

        # Restore best model weights
        model.load_weights(f'{results_path}/best_model.hdf5')

        # If use pretrained model, also save weights for global usage
        if self.pretrained_model_path:
            model.save_weights(self.pretrained_model_path)


        vars()[f"pi_{self.y_col[0]}"] = pd.DataFrame()
        vars()[f"pi_{self.y_col[1]}"] = pd.DataFrame()
        vars()[f"pi_train_{self.y_col[0]}"] = pd.DataFrame()
        vars()[f"pi_train_{self.y_col[1]}"] = pd.DataFrame()

        for run in range(self.runs):
            y_pred_scaled = model.predict(test_generator)
            vars()[f"pi_{self.y_col[0]}"][str(run)] = y_pred_scaled[:, 0]
            vars()[f"pi_{self.y_col[1]}"][str(run)] = y_pred_scaled[:, 1]

            y_pred_scaled_train = model.predict_generator(train_generator)
            vars()[f"pi_train_{self.y_col[0]}"][str(run)] = y_pred_scaled_train[:, 0]
            vars()[f"pi_train_{self.y_col[1]}"][str(run)] = y_pred_scaled_train[:, 1]

        pi_pred_es_scaled = pd.DataFrame()
        pi_pred_es_scaled_train = pd.DataFrame()
        y_pred_es_scaled = pd.DataFrame()
        y_pred_es_scaled_train = pd.DataFrame()

        # Define confidence interval width (for alpha = 0.05, confidence interval is going to be 0.95)
        lower_q = self.alpha / 2
        upper_q = 1 - self.alpha / 2

        # Get target variables forecast and intervals
        pi_pred_es_scaled[self.y_col[0] + '_lower'] = np.quantile(vars()[f"pi_{self.y_col[0]}"], lower_q, axis=1)
        pi_pred_es_scaled[self.y_col[0] + '_upper'] = np.quantile(vars()[f"pi_{self.y_col[0]}"], upper_q, axis=1)
        pi_pred_es_scaled[self.y_col[1] + '_lower'] = np.quantile(vars()[f"pi_{self.y_col[1]}"], lower_q, axis=1)
        pi_pred_es_scaled[self.y_col[1] + '_upper'] = np.quantile(vars()[f"pi_{self.y_col[1]}"], upper_q, axis=1)

        pi_pred_es_scaled_train[self.y_col[0] + '_lower'] = np.quantile(vars()[f"pi_train_{self.y_col[0]}"], lower_q,
                                                                        axis=1)
        pi_pred_es_scaled_train[self.y_col[0] + '_upper'] = np.quantile(vars()[f"pi_train_{self.y_col[0]}"], upper_q,
                                                                        axis=1)
        pi_pred_es_scaled_train[self.y_col[1] + '_lower'] = np.quantile(vars()[f"pi_train_{self.y_col[1]}"], lower_q,
                                                                        axis=1)
        pi_pred_es_scaled_train[self.y_col[1] + '_upper'] = np.quantile(vars()[f"pi_train_{self.y_col[1]}"], upper_q,
                                                                        axis=1)
        y_pred_es_scaled[self.y_col[0]] = np.mean(vars()[f"pi_{self.y_col[0]}"], axis=1)
        y_pred_es_scaled[self.y_col[1]] = np.mean(vars()[f"pi_{self.y_col[1]}"], axis=1)
        y_pred_es_scaled_train[self.y_col[0]] = np.mean(vars()[f"pi_train_{self.y_col[0]}"], axis=1)
        y_pred_es_scaled_train[self.y_col[1]] = np.mean(vars()[f"pi_train_{self.y_col[1]}"], axis=1)

        print('[INFO] prediction intervals computed')

        tf.keras.backend.clear_session()
        return y_pred_es_scaled, y_pred_es_scaled_train, pi_pred_es_scaled, pi_pred_es_scaled_train

    def reTS(self, y_pred_es_scaled, es_scaled, df_level, df_scaler, df, forecast_filename='forecast.pkl'):
        """
        re-trend, re-seasonalize, descale
        returns forecasts and truth
        
        """
        # descale & desmooth forecast
        placehold_df = es_scaled.copy()


        if forecast_filename == 'forecast.pkl':
            start_index = -y_pred_es_scaled.shape[0]
            end_index = None

        elif forecast_filename == 'forecast_train.pkl':
            start_index = self.n_input_train
            end_index = self.n_input_train + y_pred_es_scaled.shape[0]

        placehold_df[self.y_col[0]][start_index:end_index] = y_pred_es_scaled[self.y_col[0]]
        placehold_df[self.y_col[1]][start_index:end_index] = y_pred_es_scaled[self.y_col[1]]

        placehold_df = placehold_df + df_level.values  # desmooth
        df_scaler.fit(df)
        placehold_df = df_scaler.inverse_transform(placehold_df)  # descale
        y_pred = placehold_df[self.y_col][start_index:end_index]
        forecasts = pd.DataFrame({self.y_col[0] + '_true': df[self.y_col[0]][start_index:end_index].values,
                                  self.y_col[0] + '_pred': y_pred.iloc[:, 0],
                                  self.y_col[1] + '_true': df[self.y_col[1]][start_index:end_index].values,
                                  self.y_col[1] + '_pred': y_pred.iloc[:, 1]})

        print(forecasts)
        makedirs('./forecasting_results/' + self.loc + '/' + str(self.alpha) + '/' + self.results_path, exist_ok=True)
        forecasts.to_pickle(
            './forecasting_results/' + self.loc + '/' + str(self.alpha) + '/' + self.results_path + forecast_filename)
        print('[INFO] forecasts saved in results folder')
        return forecasts

    def reTS_pi(self, pi_pred_es_scaled, df_level, df_scaler, df, pi_filename='pi.pkl'):
        """
        re-trend, re-seasonalize, descale
        returns PIs and truth

        """
        placehold_df = df.copy()

        if pi_filename == 'pi.pkl':
            start_index = -pi_pred_es_scaled.shape[0]
            end_index = None

        elif pi_filename == 'pi_train.pkl':
            start_index = self.n_input_train
            end_index = self.n_input_train + pi_pred_es_scaled.shape[0]

        # descale & desmooth lower
        placehold_df[self.y_col[0]][start_index:end_index] = pi_pred_es_scaled[self.y_col[0] + '_lower']
        placehold_df[self.y_col[1]][start_index:end_index] = pi_pred_es_scaled[self.y_col[1] + '_lower']
        placehold_df = placehold_df + df_level.values  # desmooth
        df_scaler.fit(df)
        placehold_df = df_scaler.inverse_transform(placehold_df)  # descale
        pi_pred_0 = placehold_df[self.y_col][start_index:end_index]

        # descale & desmooth second predictant
        placehold_df = df.copy()
        placehold_df[self.y_col[0]][start_index:end_index] = pi_pred_es_scaled[self.y_col[0] + '_upper']
        placehold_df[self.y_col[1]][start_index:end_index] = pi_pred_es_scaled[self.y_col[1] + '_upper']
        placehold_df = placehold_df + df_level.values  # desmooth
        df_scaler.fit(df)
        placehold_df = df_scaler.inverse_transform(placehold_df)  # descale
        pi_pred_1 = placehold_df[self.y_col][start_index:end_index]  # pi_1
        pi = pd.DataFrame({self.y_col[0] + '_true': df[self.y_col[0]][start_index:end_index].values,
                           self.y_col[0] + '_lower': pi_pred_0.iloc[:, 0],
                           self.y_col[1] + '_lower': pi_pred_0.iloc[:, 1],
                           self.y_col[1] + '_true': df[self.y_col[1]][start_index:end_index].values,
                           self.y_col[0] + '_upper': pi_pred_1.iloc[:, 0],
                           self.y_col[1] + '_upper': pi_pred_1.iloc[:, 1], })
        makedirs('./forecasting_results/' + self.loc + '/' + str(self.alpha) + '/' + self.results_path, exist_ok=True)
        pi.to_pickle(
            './forecasting_results/' + self.loc + '/' + str(self.alpha) + '/' + self.results_path + pi_filename)
        print('[INFO] prediction intervals saved in results folder')
        return pi

    def descale(self, y_pred_scaled, scaled_df, df_scaler, df, forecast_filename='forecast.pkl'):
        """
        descale
        returns forecasts and truth
        
        """
        if forecast_filename == 'forecast.pkl':
            start_index = -y_pred_scaled.shape[0]
            end_index = None

        elif forecast_filename == 'forecast_train.pkl':
            start_index = self.n_input_train
            end_index = self.n_input_train + y_pred_scaled.shape[0]


        placehold_df = scaled_df.copy()
        placehold_df[self.y_col[0]][start_index:end_index] = y_pred_scaled[self.y_col[0]]
        placehold_df[self.y_col[1]][start_index:end_index] = y_pred_scaled[self.y_col[1]]
        df_scaler.fit(df)
        placehold_df = df_scaler.inverse_transform(placehold_df)  # descale

        y_pred = placehold_df[self.y_col][start_index:end_index]

        forecasts = pd.DataFrame({self.y_col[0] + '_true': df[self.y_col[0]][start_index:end_index].values,
                                  self.y_col[0] + '_pred': y_pred.iloc[:, 0],
                                  self.y_col[1] + '_true': df[self.y_col[1]][start_index:end_index].values,
                                  self.y_col[1] + '_pred': y_pred.iloc[:, 1]})

        print(forecasts)

        if isdir('./forecasting_results/' + self.loc + '/' + str(self.alpha) + '/' + self.results_path) == False:
            mkdir('./forecasting_results/' + self.loc + '/' + str(self.alpha) + '/' + self.results_path)
        forecasts.to_pickle(
            './forecasting_results/' + self.loc + '/' + str(self.alpha) + '/' + self.results_path + forecast_filename)
        print('[INFO] forecasts saved in results folder')
        return forecasts

    def descale_pi(self, pi_pred_scaled, scaled_df, df_scaler, df, pi_filename='pi.pkl'):
        """
        descale
        returns PIs and truth
        
        """

        placehold_df = scaled_df.copy()
        placehold_df[self.y_col[0]][-pi_pred_scaled.shape[0]:] = pi_pred_scaled[self.y_col[0] + '_lower']
        placehold_df[self.y_col[1]][-pi_pred_scaled.shape[0]:] = pi_pred_scaled[self.y_col[1] + '_lower']
        df_scaler.fit(df)
        placehold_df = df_scaler.inverse_transform(placehold_df)  # descale
        pi_pred_0 = placehold_df[self.y_col][-pi_pred_scaled.shape[0]:]  # pi_0

        placehold_df = scaled_df.copy()
        placehold_df[self.y_col[0]][-pi_pred_scaled.shape[0]:] = pi_pred_scaled[self.y_col[0] + '_upper']
        placehold_df[self.y_col[1]][-pi_pred_scaled.shape[0]:] = pi_pred_scaled[self.y_col[1] + '_upper']
        df_scaler.fit(df)
        placehold_df = df_scaler.inverse_transform(placehold_df)  # descale

        pi_pred_1 = placehold_df[self.y_col][-pi_pred_scaled.shape[0]:]  # pi_1

        pi = pd.DataFrame({self.y_col[0] + '_true': df[self.y_col[0]][-pi_pred_scaled.shape[0]:].values,
                           self.y_col[0] + '_lower': pi_pred_0.iloc[:, 0],
                           self.y_col[1] + '_lower': pi_pred_0.iloc[:, 1],
                           self.y_col[1] + '_true': df[self.y_col[1]][-pi_pred_scaled.shape[0]:].values,
                           self.y_col[0] + '_upper': pi_pred_1.iloc[:, 0],
                           self.y_col[1] + '_upper': pi_pred_1.iloc[:, 1], })
        if isdir('./forecasting_results/' + self.loc + '/' + str(self.alpha) + '/' + self.results_path) == False:
            mkdir('./forecasting_results/' + self.loc + '/' + str(self.alpha) + '/' + self.results_path)
        pi.to_pickle(
            './forecasting_results/' + self.loc + '/' + str(self.alpha) + '/' + self.results_path + pi_filename)
        print('[INFO] prediction intervals saved in results folder')
        return pi


class stats():
    """
    statistical benchmarks
    
    """

    def __init__(self,
                 y_col=['HC', 'HC%'],
                 n_input_train=14,
                 alpha=0.05,
                 loc='United Kingdom',
                 results_path='results/varmax/'
                 ):

        self.y_col = y_col
        self.alpha = alpha
        self.results_path = results_path
        self.loc = loc

    def forecast_varmax(self, x_train, y_train, x_test, y_test):
        """
        returns model forecast & predition intervals
        
        """

        model = VARMAX(endog=y_train,
                       exog=x_train,
                       enforce_invertibility=False,
                       )

        model_fit = model.fit(disp=False)
        print('[INFO] VARMAX fitting complete')

        model_forecast = model_fit.get_prediction(start=model.nobs, end=model.nobs + x_test.shape[0] - 1, exog=x_test)
        y_pred_scaled = model_forecast.predicted_mean  # forecast
        pi_pred_scaled = model_forecast.conf_int(alpha=self.alpha)

        model_forecast = model_fit.get_prediction(start = 0, end = y_train.shape[0] - 1, exog=x_train)
        y_pred_scaled_train = model_forecast.predicted_mean  # forecast
        pi_pred_scaled_train = model_forecast.conf_int(alpha=self.alpha)

        return y_pred_scaled, pi_pred_scaled, y_pred_scaled_train, pi_pred_scaled_train

    def forecast_ES(self, y_train, y_test):
        """
        returns model forecast & predition intervals
        
        """
        y_pred_scaled = pd.DataFrame(columns=self.y_col)
        pi_pred_scaled = pd.DataFrame()

        y_pred_scaled_train = pd.DataFrame(columns=self.y_col)
        pi_pred_scaled_train = pd.DataFrame()

        for col in self.y_col:
            # Build model
            ets_model = ETSModel(
                endog = y_train[col],  # y should be a pd.Series
                trend = 'add',
                seasonal = None,
                initialization_method = 'known',
                initial_level = y_train[col][0],
                initial_trend = 0
            )
            # Fit model
            ets_result = ets_model.fit(disp=False)
            # Simulate predictions
            n_repetitions = 500

            df_simul = ets_result.simulate(
                nsimulations=y_test.shape[0],
                repetitions=n_repetitions,
                anchor='end',
            )

            ci = 1 - self.alpha
            upper_ci = df_simul.quantile(q=1 - ci / 2, axis='columns')
            lower_ci = df_simul.quantile(q=ci / 2, axis='columns')
            model_forecast = df_simul.mean(axis=1)

            y_pred_scaled[col] = model_forecast  # last observations referred as test
            y_pred_scaled_train[col] = ets_result.fittedvalues  # first observations are train

            pi = pd.DataFrame({'lower ' + col: lower_ci,
                               'upper ' + col: upper_ci})

            pi_train = pd.DataFrame({'lower ' + col: lower_ci,
                                     'upper ' + col: upper_ci})
            for pi_col in pi.columns:
                pi_pred_scaled[pi_col] = pi[pi_col]
                pi_pred_scaled_train[pi_col] = pi_train[pi_col]

        print('[INFO] ES fitting complete')
        return y_pred_scaled, pi_pred_scaled, y_pred_scaled_train, pi_pred_scaled_train

    def descale(self, y_pred_scaled, df_scaler, df, forecast_filename='forecast.pkl'):
        """
        descale & prune (latter for coparison with MES-RNN)
        returns forecasts and truth
        
        """
        if forecast_filename == 'forecast.pkl':
            start_index = -y_pred_scaled.shape[0]
            end_index = None

        elif forecast_filename == 'forecast_train.pkl':
            start_index = 0
            end_index = y_pred_scaled.shape[0]

        placehold_df = df.copy()
        placehold_df[self.y_col[0]][start_index:end_index] = y_pred_scaled[self.y_col[0]]
        placehold_df[self.y_col[1]][start_index:end_index] = y_pred_scaled[self.y_col[1]]
        df_scaler.fit(df)
        placehold_df = df_scaler.inverse_transform(placehold_df)
        y_pred = placehold_df[self.y_col][start_index:end_index]

        forecasts = pd.DataFrame({self.y_col[0] + '_true': df[self.y_col[0]][start_index:end_index].values,
                                  self.y_col[0] + '_pred': y_pred.iloc[:, 0],
                                  self.y_col[1] + '_true': df[self.y_col[1]][start_index:end_index].values,
                                  self.y_col[1] + '_pred': y_pred.iloc[:, 1]})
        makedirs('./forecasting_results/' + self.loc + '/' + str(self.alpha) + '/' + self.results_path, exist_ok=True)
        forecasts.to_pickle(
            './forecasting_results/' + self.loc + '/' + str(self.alpha) + '/' + self.results_path + forecast_filename)
        print('[INFO] forecasts saved in results folder')
        return forecasts

    def descale_pi(self, pi_pred_scaled, scaled_df, df_scaler, df):
        """
        descale & prune
        returns PIs and truth
        
        """
        # lower
        placehold_df = scaled_df.copy()
        placehold_df[self.y_col[0]][-pi_pred_scaled.shape[0]:] = pi_pred_scaled['lower ' + self.y_col[0]]
        placehold_df[self.y_col[1]][-pi_pred_scaled.shape[0]:] = pi_pred_scaled['lower ' + self.y_col[1]]
        df_scaler.fit(df)
        placehold_df = df_scaler.inverse_transform(placehold_df)  # descale
        pi_pred_0 = placehold_df[self.y_col][-pi_pred_scaled.shape[0]:]

        # upper
        placehold_df = scaled_df.copy()
        placehold_df[self.y_col[0]][-pi_pred_scaled.shape[0]:] = pi_pred_scaled['upper ' + self.y_col[0]]
        placehold_df[self.y_col[1]][-pi_pred_scaled.shape[0]:] = pi_pred_scaled['upper ' + self.y_col[1]]
        df_scaler.fit(df)
        placehold_df = df_scaler.inverse_transform(placehold_df)  # descale
        pi_pred_1 = placehold_df[self.y_col][-pi_pred_scaled.shape[0]:]

        pi = pd.DataFrame({self.y_col[0] + '_true': df[self.y_col[0]][-pi_pred_scaled.shape[0]:].values,
                           self.y_col[0] + '_lower': pi_pred_0.iloc[:, 0],
                           self.y_col[1] + '_lower': pi_pred_0.iloc[:, 1],
                           self.y_col[1] + '_true': df[self.y_col[1]][-pi_pred_scaled.shape[0]:].values,
                           self.y_col[0] + '_upper': pi_pred_1.iloc[:, 0],
                           self.y_col[1] + '_upper': pi_pred_1.iloc[:, 1], })
        makedirs('./forecasting_results/' + self.loc + '/' + str(self.alpha) + '/' + self.results_path, exist_ok=True)
        pi.to_pickle('./forecasting_results/' + self.loc + '/' + str(self.alpha) + '/' + self.results_path + 'pi.pkl')
        print('[INFO] prediction intervals saved in results folder')
        return pi
