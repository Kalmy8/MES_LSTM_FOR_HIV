from datetime import timedelta
from utils.forecasting_models import *
from utils.metrics import *
import os
import warnings
warnings.simplefilter('ignore')
import pandas as pd
import shutil

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Run config
config = {'WINDOW_SIZE': 3,
          'ALPHA': 0.05,
          'EPOCHS_NUMBER': 500,
          'LSTM_SIZE': 50,
          'PRETRAINED_MODEL_PATH': None,
          'EARLY_STOPPING': 5,
          'STATEFUL': True,
          'LEARNING_RATE': 1e-3}
WINDOW_SIZE = config['WINDOW_SIZE']
ALPHA = config['ALPHA']
EPOCHS_NUMBER = config['EPOCHS_NUMBER']
LSTM_SIZE = config['LSTM_SIZE']
PRETRAINED_MODEL_PATH = config['PRETRAINED_MODEL_PATH']
EARLY_STOPPING = config['EARLY_STOPPING']
STATEFUL = config['STATEFUL']
LEARNING_RATE = config['LEARNING_RATE']

# Constants
DATA_PATH = './forecasting_data'

def regions_loop(dataframe, pre_layer):
    subject = dataframe['Территория']

    # Preprocessing
    dataframe = pre_layer.clip_time_series_data(dataframe, 'Год', (2006, 2021))
    dataframe = pre_layer.clean_data(dataframe)  # Drop columns, set indexer
    dataframe = pre_layer.set_index(dataframe)
    dataframe = pre_layer.fill_missing(dataframe)  # FillNa

    # Add padding to perform actual prediction
    actual_date = dataframe.index[-1]
    actual_date += timedelta(days=365)
    dataframe.loc[actual_date] = None

    hist, dataframe = pre_layer.reindex(dataframe)

    # scale historic data
    scaled_df, df_scaler = pre_layer.scale(hist)

    # exponentially smooth historic data
    mes_layer = ES(loc=subject)
    internals = mes_layer.es(scaled_df)
    _, df_level = mes_layer.deTS(scaled_df, internals)

    # scale test input
    scaled_df, _ = pre_layer.scale(dataframe)
    es_scaled, _ = mes_layer.deTS(scaled_df, internals)

    results_HC_test = pd.DataFrame(columns=['mape_meslstm', 'rmse_meslstm', 'mis_meslstm', 'cov_meslstm',
                                            'mape_lstm', 'rmse_lstm', 'mis_lstm', 'cov_lstm',
                                            'mape_varmax', 'rmse_varmax', 'mis_varmax', 'cov_varmax',
                                            'mape_ES', 'rmse_ES', 'mis_ES', 'cov_ES'])
    results_HC_train = results_HC_test.copy()

    results_HC_percent_test = pd.DataFrame(columns=['mape_meslstm', 'rmse_meslstm', 'mis_meslstm', 'cov_meslstm',
                                                    'mape_lstm', 'rmse_lstm', 'mis_lstm', 'cov_lstm',
                                                    'mape_varmax', 'rmse_varmax', 'mis_varmax', 'cov_varmax',
                                                    'mape_ES', 'rmse_ES', 'mis_ES', 'cov_ES'])

    results_HC_percent_train = results_HC_percent_test.copy()

    def append_results(result_HC, result_HC_percent, forecasts, pi=None):
        result_HC.append(mape(forecasts['HC_true'], forecasts['HC_pred']))
        result_HC_percent.append(mape(forecasts['HC%_true'], forecasts['HC%_pred']))
        result_HC.append(rmse(forecasts['HC_true'], forecasts['HC_pred']))
        result_HC_percent.append(rmse(forecasts['HC%_true'], forecasts['HC%_pred']))
        if pi is not None:
            result_HC.append(
                mis(pi['HC_lower'].values, pi['HC_upper'].values, pi['HC_true'].values, alpha=dl_layer.alpha))
            result_HC_percent.append(
                mis(pi['HC%_lower'].values, pi['HC%_upper'].values, pi['HC%_true'].values, alpha=dl_layer.alpha))
            result_HC.append(coverage(pi['HC_lower'].values, pi['HC_upper'].values, pi['HC_true'].values))
            result_HC_percent.append(coverage(pi['HC%_lower'].values, pi['HC%_upper'].values, pi['HC%_true'].values))

    result_HC_train, result_HC_test, result_HC_percent_train, result_HC_percent_test = [], [], [], []
    print(f'\n [INFO] trial for: {subject}')

    # MES_LSTM
    dl_layer = lstm(loc=subject,
                    alpha=ALPHA,
                    lstm_size=LSTM_SIZE,
                    epochs=EPOCHS_NUMBER,
                    n_input_train=WINDOW_SIZE,
                    b_size_train=1,
                    n_input_valid=WINDOW_SIZE,
                    b_size_valid=1,
                    pretrained_model_path=PRETRAINED_MODEL_PATH,
                    stateful=STATEFUL,
                    learning_rate=LEARNING_RATE,
                    early_stopping=EARLY_STOPPING)

    x_train, y_train, x_test, y_test = dl_layer.split(es_scaled)

    # Forecasts and prediction intervals
    y_pred_es_scaled, y_pred_es_scaled_train, pi_pred_es_scaled, pi_pred_es_scaled_train = dl_layer.pi_model(x_train,
                                                                                                             y_train,
                                                                                                             x_test,
                                                                                                             y_test)

    forecasts = dl_layer.reTS(y_pred_es_scaled, es_scaled, df_level, df_scaler, dataframe, forecast_filename='forecast.pkl')
    forecasts_train = dl_layer.reTS(y_pred_es_scaled_train, es_scaled, df_level, df_scaler, dataframe,
                                    forecast_filename="forecast_train.pkl")

    pi = dl_layer.reTS_pi(pi_pred_es_scaled, df_level, df_scaler, dataframe, pi_filename='pi.pkl')
    pi_train = dl_layer.reTS_pi(pi_pred_es_scaled_train, df_level, df_scaler, dataframe, pi_filename='pi_train.pkl')

    # metrics
    append_results(result_HC_test, result_HC_percent_test, forecasts.drop(forecasts.tail(1).index, axis=0),
                   pi.drop(pi.tail(1).index, axis=0))
    append_results(result_HC_train, result_HC_percent_train, forecasts_train, pi_train)

    # Simple LSTM

    dl_layer = lstm(results_path='results/pure_lstm/',
                    loc=subject,
                    alpha=ALPHA,
                    lstm_size=LSTM_SIZE,
                    epochs=EPOCHS_NUMBER,
                    n_input_train=WINDOW_SIZE,
                    b_size_train=1,
                    n_input_valid=WINDOW_SIZE,
                    b_size_valid=1,
                    pretrained_model_path=PRETRAINED_MODEL_PATH,
                    stateful=STATEFUL,
                    learning_rate=LEARNING_RATE,
                    early_stopping=EARLY_STOPPING)

    x_train, y_train, x_test, y_test = dl_layer.split(scaled_df)

    # Forecasts and prediction intervals
    y_pred_scaled, y_pred_es_scaled_train, pi_pred_scaled, pi_pred_es_scaled_train = dl_layer.pi_model(x_train, y_train,
                                                                                                       x_test, y_test)

    forecasts = dl_layer.descale(y_pred_scaled, scaled_df, df_scaler, dataframe)
    forecasts_train = dl_layer.descale(y_pred_es_scaled_train, scaled_df, df_scaler, dataframe,
                                       forecast_filename="forecast_train.pkl")
    pi = dl_layer.descale_pi(pi_pred_scaled, scaled_df, df_scaler, dataframe)
    pi_train = dl_layer.descale_pi(pi_pred_es_scaled_train, scaled_df, df_scaler, dataframe, 'pi_train.pkl')

    # metrics
    append_results(result_HC_test, result_HC_percent_test, forecasts.drop(forecasts.tail(1).index, axis=0),
                   pi.drop(pi.tail(1).index, axis=0))
    append_results(result_HC_train, result_HC_percent_train, forecasts_train, pi_train)

    # VARMAX
    bench = stats(loc=subject, results_path='results/VARMAX/', alpha=ALPHA)
    y_pred_scaled, pi_pred_scaled, y_pred_es_scaled_train, pi_pred_es_scaled_train = bench.forecast_varmax(x_train,
                                                                                                           y_train,
                                                                                                           x_test,
                                                                                                           y_test)
    forecasts = bench.descale(y_pred_scaled, df_scaler, dataframe)
    pi = bench.descale_pi(pi_pred_scaled, scaled_df, df_scaler, dataframe)

    forecasts_train = bench.descale(y_pred_es_scaled_train, df_scaler, dataframe, forecast_filename="forecast_train.pkl")
    pi_train = bench.descale_pi(pi_pred_es_scaled_train, scaled_df, df_scaler, dataframe)

    append_results(result_HC_test, result_HC_percent_test, forecasts.drop(forecasts.tail(1).index, axis=0),
                   pi.drop(pi.tail(1).index, axis=0))
    append_results(result_HC_train, result_HC_percent_train, forecasts_train, pi_train)

    # ES

    bench = stats(loc=subject, results_path='results/ES/', alpha=ALPHA)
    y_pred_scaled, pi_pred_scaled, y_pred_scaled_train, pi_pred_scaled_train = bench.forecast_ES(y_train, y_test)
    forecasts = bench.descale(y_pred_scaled, df_scaler, dataframe)
    pi = bench.descale_pi(pi_pred_scaled, scaled_df, df_scaler, dataframe)

    forecasts_train = bench.descale(y_pred_scaled_train, df_scaler, dataframe, forecast_filename="forecast_train.pkl")
    pi_train = bench.descale_pi(pi_pred_scaled_train, scaled_df, df_scaler, dataframe)

    append_results(result_HC_test, result_HC_percent_test, forecasts.drop(forecasts.tail(1).index, axis=0),
                   pi.drop(pi.tail(1).index, axis=0))
    append_results(result_HC_train, result_HC_percent_train, forecasts_train, pi_train)

    # Naive forecast
    #    NF = naive_forecaster(loc = subject, results_path = 'results/ES/', alpha=ALPHA)
    #    forecasts = NF.forecast_naive(hist)
    #    forecasts = forecasts.drop(forecasts.tail(1).index, axis=0) # Do not need last one
    #    forecasts_test = forecasts[-WINDOW_SIZE:]
    #    NF.save_forecast(forecasts_test, 'forecast_test.pkl'#)

    #    forecasts_train = forecasts[:-WINDOW_SIZE]
    #    NF.save_forecast(forecasts_train, 'forecast_train.pkl'#)

    #    append_results(result_HC_test, result_HC_percent_test, forecasts_test)
    #    append_results(result_HC_train, result_HC_percent_train, forecasts_train#)

    # Reformating and saving results
    results_HC_test = results_HC_test.append(
        pd.DataFrame(np.array(result_HC_test).reshape(1, -1), columns=list(results_HC_test)), ignore_index=True)
    results_HC_percent_test = results_HC_percent_test.append(
        pd.DataFrame(np.array(result_HC_percent_test).reshape(1, -1), columns=list(results_HC_percent_test)),
        ignore_index=True)

    results_HC_train = results_HC_train.append(
        pd.DataFrame(np.array(result_HC_train).reshape(1, -1), columns=list(results_HC_train)), ignore_index=True)
    results_HC_percent_train = results_HC_percent_train.append(
        pd.DataFrame(np.array(result_HC_percent_train).reshape(1, -1), columns=list(results_HC_percent_train)),
        ignore_index=True)

    print('[INFO] ---------------------- DONE -----------------------------')

    # Add new column to identify territory we are making forecast for
    results_HC_test['Territory'] = subject
    results_HC_percent_test['Territory'] = subject
    results_HC_train['Territory'] = subject
    results_HC_percent_train['Territory'] = subject

    results_HC_test['Train/Test'] = 'Test'
    results_HC_percent_test['Train/Test'] = 'Test'
    results_HC_train['Train/Test'] = 'Train'
    results_HC_percent_train['Train/Test'] = 'Train'

    results_HC_test['Target'] = 'HC'
    results_HC_percent_test['Target'] = 'HC_percent'
    results_HC_train['Target'] = 'HC'
    results_HC_percent_train['Target'] = 'HC_percent'

    results_HC_test.to_pickle(f'./forecasting_results/{subject}/{ALPHA}/results/multiple_runs_HC_test.pkl')
    results_HC_percent_test.to_pickle(
        f'./forecasting_results/{subject}/{ALPHA}/results/multiple_runs_HC_percent_test.pkl')

    results_HC_train.to_pickle(f'./forecasting_results/{subject}/{ALPHA}/results/multiple_runs_HC_train.pkl')
    results_HC_percent_train.to_pickle(
        f'./forecasting_results/{subject}/{ALPHA}/results/multiple_runs_HC_percent_train.pkl')


# Remove directories from exception list
def remove_directory(directory_path):
    """
  This function removes a directory and all its contents recursively.

  Args:
      directory_path (str): The path to the directory to be removed.
  """
    try:
        shutil.rmtree(directory_path)
        print(f"Directory '{directory_path}' removed successfully.")
    except FileNotFoundError:
        print(f"Directory '{directory_path}' not found.")
    except PermissionError:
        print(
            f"Permission error while removing directory '{directory_path}'. You might not have permission to remove this directory.")
    except OSError as e:
        print(f"An error occurred while removing directory '{directory_path}': {e}")

def main():
    # load data
    pre_layer = preprocess(data_path = DATA_PATH,
                           thresh = 0.6)
    df = pre_layer.load_data()

    # Territories which meant to be re-runed
    exception_list = []

    # Get list of entries in the current directory
    existing_directories = []
    try:
        existing_directories = os.listdir("./forecasting_results")
    except Exception:
        print('No ./forecasting_results directory, all Territories are processed...')

    for dirname in existing_directories:
        if dirname in exception_list:
            remove_directory(f"./forecasting_results/{dirname}")

    # Proceed learning and forecasting
    country_set = list((set(df['Территория']) - set(existing_directories)).union(set(exception_list)))
    print(country_set, sep='\n')

    for country in country_set:
        piece = df[df['Территория'] == country]
        regions_loop(piece, pre_layer)

if __name__ == '__main__':
    main()
