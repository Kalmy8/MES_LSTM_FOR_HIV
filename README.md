HIV morbidity prediction with use of MES_LSTM architecture and social-demographic factors for every subject in Russian Federation
 
### File Structure
```
./EDA/
    - EDA_results/                             # Folder containing EDA plots/reports/graphs
    - original_data/                           # Original HIV and socio-economics data
    - draw_map.py                              # Module for HIV morbidity choropleth map construciton 
    - perform_EDA.ipynb                        # Notebook performing EDA
               
./Forecasting/
    - forecasting_data/                        # Gathered dataframe ready for ML process 
    - forecasting_results/                     # Achieved forecasts for every subject 
        - Subject 1/
            - ..ES/
            - ..mes_lstm/
            - ..pure_lstm/
            - ..VARMAX/
        - Subject 2/
        - ...
    - utils/
        - metrics.py                          # Implemented metrics for forecast quality measurement
        
    - forecasting_models.py                   # Implemented forecasting models
    - process_forecasting_results.ipynb       # Forecasting results processing, visualization
    - run_forecast.py                         # Forecast invokation script
    - gcollab_version.ipynb                   # Notebook designed specifically for google collab use,
                                              # ready to launch forecast right away.

```

### Suggested Citations

#### Forecasting
```
@Article{forecast4010001,
AUTHOR = {Mathonsi, Thabang and van Zyl, Terence L.},
TITLE = {A Statistics and Deep Learning Hybrid Method for Multivariate Time Series Forecasting and Mortality Modeling},
JOURNAL = {Forecasting},
VOLUME = {4},
YEAR = {2022},
NUMBER = {1},
PAGES = {1--25},
DOI = {10.3390/forecast4010001}
}

```

#### Anomaly Detection
```
@article{s00521-021-06697-x,
  title={Multivariate anomaly detection based on prediction intervals constructed using deep learning},
  author={Mathonsi, Thabang and {van Zyl}, Terence L},
  journal={Neural Computing and Applications},
  pages={1--15},
  year={2022},
  publisher={Springer},
  doi = {10.1007/s00521-021-06697-x}
}
```



