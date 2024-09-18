## From https://towardsdatascience.com/mixture-of-kan-experts-for-high-performance-time-series-forecasting-5227e1d2aba2

!pip install git+https://github.com/Nixtla/neuralforecast.git

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datasetsforecast.long_horizon import LongHorizon

from neuralforecast.core import NeuralForecast
from neuralforecast.losses.pytorch import MAE, MSE
from neuralforecast.models import TSMixer, PatchTST, iTransformer, RMoK

from utilsforecast.losses import mae, mse
from utilsforecast.evaluation import evaluate

def load_data(name):
    if name == 'Ettm1':
        Y_df, *_ = LongHorizon.load(directory='./', group='ETTm1')
        Y_df['ds'] = pd.to_datetime(Y_df['ds'])
        freq = '15T'
        h = 96
        val_size = 11520
        test_size = 11520
    elif name == 'Ettm2':
        Y_df, *_ = LongHorizon.load(directory='./', group='ETTm2')
        Y_df['ds'] = pd.to_datetime(Y_df['ds'])
        freq = '15T'
        h = 96
        val_size = 11520
        test_size = 11520

    return Y_df, h, val_size, test_size, freq

DATASETS = ['Ettm1', 'Ettm2']

for dataset in DATASETS:

    Y_df, horizon, val_size, test_size, freq = load_data(dataset)

    rmok_model = RMoK(input_size=horizon,
                      h=horizon, 
                      n_series=7,
                      num_experts=4,
                      dropout=0.1,
                      revine_affine=True,
                      learning_rate=0.001,
                      scaler_type='identity',
                      max_steps=1000,
                      early_stop_patience_steps=5)
    
    patchtst_model = PatchTST(input_size=horizon, 
                              h=horizon, 
                              encoder_layers=3,
                              n_heads=4,
                              hidden_size=16,
                              dropout=0.3,
                              patch_len=16,
                              stride=8,
                              scaler_type='identity', 
                              max_steps=1000, 
                              early_stop_patience_steps=5)
    
    iTransformer_model = iTransformer(input_size=horizon, 
                                      h=horizon, 
                                      n_series=7,
                                      e_layers=2,
                                      hidden_size=128,
                                      d_ff=128,
                                      scaler_type='identity', 
                                      max_steps=1000, 
                                      early_stop_patience_steps=3)
    
    tsmixer_model = TSMixer(input_size=horizon,
                            h=horizon,
                            n_series=7,
                            n_block=2,
                            dropout=0.9,
                            ff_dim=64,
                            learning_rate=0.001,
                            scaler_type='identity',
                            max_steps=1000,
                            early_stop_patience_steps=5)
    models = [rmok_model, patchtst_model, iTransformer_model, tsmixer_model]

    nf = NeuralForecast(models=models, freq=freq)
    nf_preds = nf.cross_validation(df=Y_df, val_size=val_size, test_size=test_size, n_windows=None)
    nf_preds = nf_preds.reset_index()

    evaluation = evaluate(df=nf_preds, metrics=[mae, mse], models=['RMoK', 'PatchTST', 'iTransformer', 'TSMixer'])
    evaluation.to_csv(f'{dataset}_results.csv', index=False, header=True)


### Evaluation
ettm1_eval = pd.read_csv('Ettm1_results.csv')
ettm1_eval = ettm1_eval.drop(['unique_id'], axis=1).groupby('metric').mean().reset_index()

ettm2_eval = pd.read_csv('Ettm2_results.csv')
ettm2_eval = ettm2_eval.drop(['unique_id'], axis=1).groupby('metric').mean().reset_index()
