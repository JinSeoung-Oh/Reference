"""
From https://towardsdatascience.com/chronos-the-latest-time-series-forecasting-foundation-model-by-amazon-2687d641705a

Chronos is essentially exploring the potential of utilizing large language models (LLMs) for time series forecasting,
a concept initiated by questioning the adaptability of good language models to this specific domain. 
Unlike natural language processing (NLP), where models predict the next word from a fixed vocabulary, 
time series forecasting involves predicting the next value in a sequence that can be infinitely expanding or contracting.

The Chronos framework begins by addressing the challenge of adapting LLMs to handle time series data. 
It involves a multi-step process:

1. Tokenization of Time Series
   To make time series data compatible with LLMs, the data must be tokenized. 
   This involves scaling the data and then quantizing it into a fixed set of tokens. 
   Scaling ensures that the data values are within a manageable range, while quantization maps continuous values to discrete tokens.
   The proposed method involves mean scaling followed by quantization, with techniques such as percentile binning used to create fixed token sets.

2. Training
   Once tokenized, the time series data is fed into the LLM for training. 
   The model learns to associate nearby tokens and understand the distribution of data. 
   The choice of categorical cross-entropy loss function facilitates learning arbitrary distributions, 
   enhancing the model's ability to generalize for zero-shot forecasting.

3. Inference
   After training, the model is capable of forecasting future values. 
   It takes context tokens as input and generates a sequence of future tokens, 
   which are then dequantized to obtain scaled predictions. This process reverses the scaling applied during tokenization, 
   resulting in forecasts on the original scale of the data.

Moreover, Chronos tackles the challenge of data availability for training by proposing two data augmentation techniques:

1. TSMix (Time Series Mixup)
   This method randomly samples subsequences of existing time series, scales them, and combines them using convex combinations. 
   It increases the diversity of the training set by generating synthetic variations of existing data.

2. KernelSynth
   KernelSynth leverages Gaussian processes to generate synthetic time series data. 
   It involves defining kernels representing general time series behaviors (e.g., linear, periodic) and combining them 
   to create diverse synthetic data points.

Overall, Chronos not only presents a framework for adapting LLMs to time series forecasting but also addresses the challenge of data 
scarcity through innovative augmentation techniques. This comprehensive approach offers a promising avenue 
for leveraging powerful language models in the realm of time series analysis.
"""

pip install git+https://github.com/amazon-science/chronos-forecasting.git
import time
from datasetsforecast.m3 import M3
import torch
from chronos import ChronosPipeline

Y_df, *_ = M3.load(directory='./', group='Monthly')
pipeline = ChronosPipeline.from_pretrained(
  "amazon/chronos-t5-large",
  device_map="cuda",
  torch_dtype=torch.bfloat16,
)

# Predicting with Chronos
horizon = 12
batch_size = 12


actual = []
chronos_large_preds = []

start = time.time()
all_timeseries = [
    torch.tensor(sub_df["y"].values[:-horizon])
    for _, sub_df in Y_df.groupby("unique_id")
]
for i in tqdm(range(0, len(all_timeseries), batch_size)):
    batch_context = all_timeseries[i : i + batch_size]
    forecast = pipeline.predict(batch_context, horizon)
    predictions = np.quantile(forecast.numpy(), 0.5, axis=1)

    chronos_large_preds.append(predictions)

chronos_large_preds = np.concatenate(chronos_large_preds)
chronos_large_duration = time.time() - start
print(chronos_large_duration)

# Forecasting with MLP and N-BEATS
from neuralforecast.models import MLP, NBEATS
from neuralforecast.losses.pytorch import HuberLoss
from neuralforecast.core import NeuralForecast

horizon = 12
val_size = 12
test_size = 12

# Fit an MLP
mlp = MLP(h=horizon, input_size=3*horizon, loss=HuberLoss())

nf = NeuralForecast(models=[mlp], freq='M')
mlp_forecasts_df = nf.cross_validation(df=Y_df, val_size=val_size, test_size=test_size, n_windows=None, verbose=True)

# Fit N-BEATS
nbeats = NBEATS(h=horizon, input_size=3*horizon, loss=HuberLoss())

nf = NeuralForecast(models=[nbeats], freq='M')
nbeats_forecasts_df = nf.cross_validation(df=Y_df, val_size=val_size, test_size=test_size, n_windows=None, verbose=True)








