# From https://towardsdatascience.com/time-llm-reprogram-an-llm-for-time-series-forecasting-e2558087b8ac

# To more detail, see : https://github.com/KimMeen/Time-LLM
class TimeLLM(BaseWindows):
    def __init__(self,
                 h,
                 input_size,
                 patch_len: int = 16,
                 stride: int = 8,
                 d_ff: int = 128,
                 top_k: int = 5,
                 d_llm: int = 768,
                 d_model: int = 32,
                 n_heads: int = 8,
                 enc_in: int = 7,
                 dec_in: int  = 7,
                 llm = None,
                 llm_config = None,
                 llm_tokenizer = None,
                 llm_num_hidden_layers = 32,
                 llm_output_attention: bool = True,
                 llm_output_hidden_states: bool = True,
                 prompt_prefix: str = None,
                 dropout: float = 0.1,
                # Inherited parameters of BaseWindows

    def forward(self, windows_batch):
       insample_y = windows_batch['insample_y']
    
       x = insample_y.unsqueeze(-1)
    
       y_pred = self.forecast(x)
       y_pred = y_pred[:, -self.h:, :]
       y_pred = self.loss.domain_map(y_pred)
    
      return y_pred


## Prediction with TimeLLM

import time
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from neuralforecast import NeuralForecast
from neuralforecast.models import TimeLLM
from neuralforecast.losses.pytorch import MAE
from neuralforecast.tsdataset import TimeSeriesDataset
from neuralforecast.utils import AirPassengers, AirPassengersPanel, AirPassengersStatic, augment_calendar_df

from transformers import GPT2Config, GPT2Model, GPT2Tokenizer

AirPassengersPanel, calendar_cols = augment_calendar_df(df=AirPassengersPanel, freq='M')

Y_train_df = AirPassengersPanel[AirPassengersPanel.ds<AirPassengersPanel['ds'].values[-12]]
Y_test_df = AirPassengersPanel[AirPassengersPanel.ds>=AirPassengersPanel['ds'].values[-12]].reset_index(drop=True)

gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')
gpt2 = GPT2Model.from_pretrained('openai-community/gpt2',config=gpt2_config)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2')

prompt_prefix = "The dataset contains data on monthly air passengers. There is a yearly seasonality"

timellm = TimeLLM(h=12,
                 input_size=36,
                 llm=gpt2,
                 llm_config=gpt2_config,
                 llm_tokenizer=gpt2_tokenizer,
                 prompt_prefix=prompt_prefix,
                 max_steps=100,
                 batch_size=24,
                 windows_batch_size=24)

nf = NeuralForecast(
    models=[timellm],
    freq='M'
)

nf.fit(df=Y_train_df, val_size=12)
forecasts = nf.predict(futr_df=Y_test_df)

## Forecasting with N-BEATS and MLP

nbeats = NBEATS(h=12, input_size=36, max_steps=100)
mlp = MLP(h=12, input_size=36, max_steps=100)

nf = NeuralForecast(models=[nbeats, mlp], freq='M')

nf.fit(df=Y_train_df, val_size=12)


from neuralforecast.losses.numpy import mae

mae_timellm = mae(Y_test_df['y'], Y_test_df['TimeLLM'])
mae_nbeats = mae(Y_test_df['y'], Y_test_df['NBEATS'])
mae_mlp = mae(Y_test_df['y'], Y_test_df['MLP'])

data = {'Time-LLM': [mae_timellm], 
       'N-BEATS': [mae_nbeats],
       'MLP': [mae_mlp]}

metrics_df = pd.DataFrame(data=data)
metrics_df.index = ['mae']

metrics_df.style.highlight_min(color='lightgreen', axis=1)
forecasts = nf.predict(futr_df=Y_test_df)

