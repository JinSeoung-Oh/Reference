### From https://www.marktechpost.com/2025/08/24/a-coding-guide-to-build-flexible-multi-model-workflows-in-gluonts-with-synthetic-data-evaluation-and-advanced-visualizations/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.dataset.artificial import ComplexSeasonalTimeSeries

try:
   from gluonts.torch import DeepAREstimator
   TORCH_AVAILABLE = True
except ImportError:
   TORCH_AVAILABLE = False

try:
   from gluonts.mx import DeepAREstimator as MXDeepAREstimator
   from gluonts.mx import SimpleFeedForwardEstimator
   MX_AVAILABLE = True
except ImportError:
   MX_AVAILABLE = False

def create_synthetic_dataset(num_series=50, length=365, prediction_length=30):
   """Generate synthetic multi-variate time series with trends, seasonality, and noise"""
   np.random.seed(42)
   series_list = []
  
   for i in range(num_series):
       trend = np.cumsum(np.random.normal(0.1 + i*0.01, 0.1, length))
      
       daily_season = 10 * np.sin(2 * np.pi * np.arange(length) / 7) 
       yearly_season = 20 * np.sin(2 * np.pi * np.arange(length) / 365.25) 
      
       noise = np.random.normal(0, 5, length)
       values = np.maximum(trend + daily_season + yearly_season + noise + 100, 1)
      
       dates = pd.date_range(start='2020-01-01', periods=length, freq='D')
      
       series_list.append(pd.Series(values, index=dates, name=f'series_{i}'))
  
   return pd.concat(series_list, axis=1)

print(" Creating synthetic multi-series dataset...")
df = create_synthetic_dataset(num_series=10, length=200, prediction_length=30)
dataset = PandasDataset(df, target=df.columns.tolist())
training_data, test_gen = split(dataset, offset=-60)
test_data = test_gen.generate_instances(prediction_length=30, windows=2)
print(" Initializing forecasting models...")

models = {}

if TORCH_AVAILABLE:
   try:
       models['DeepAR_Torch'] = DeepAREstimator(
           freq='D',
           prediction_length=30
       )
       print(" PyTorch DeepAR loaded")
   except Exception as e:
       print(f" PyTorch DeepAR failed to load: {e}")


if MX_AVAILABLE:
   try:
       models['DeepAR_MX'] = MXDeepAREstimator(
           freq='D',
           prediction_length=30,
           trainer=dict(epochs=5)
       )
       print(" MXNet DeepAR loaded")
   except Exception as e:
       print(f" MXNet DeepAR failed to load: {e}")
  
   try:
       models['FeedForward'] = SimpleFeedForwardEstimator(
           freq='D',
           prediction_length=30,
           trainer=dict(epochs=5)
       )
       print(" FeedForward model loaded")
   except Exception as e:
       print(f" FeedForward failed to load: {e}")


if not models:
   print(" Using artificial dataset with built-in models...")
   artificial_ds = ComplexSeasonalTimeSeries(
       num_series=10,
       prediction_length=30,
       freq='D',
       length_low=150,
       length_high=200
   ).generate()
  
   training_data, test_gen = split(artificial_ds, offset=-60)
   test_data = test_gen.generate_instances(prediction_length=30, windows=2)

trained_models = {}
all_forecasts = {}

if models:
   for name, estimator in models.items():
       print(f" Training {name} model...")
       try:
           predictor = estimator.train(training_data)
           trained_models[name] = predictor
          
           forecasts = list(predictor.predict(test_data.input))
           all_forecasts[name] = forecasts
           print(f" {name} training completed!")
          
       except Exception as e:
           print(f" {name} training failed: {e}")
           continue

print(" Evaluating model performance...")
evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
evaluation_results = {}

for name, forecasts in all_forecasts.items():
   if forecasts: 
       try:
           agg_metrics, item_metrics = evaluator(test_data.label, forecasts)
           evaluation_results[name] = agg_metrics
           print(f"\n{name} Performance:")
           print(f"  MASE: {agg_metrics['MASE']:.4f}")
           print(f"  sMAPE: {agg_metrics['sMAPE']:.4f}")
           print(f"  Mean wQuantileLoss: {agg_metrics['mean_wQuantileLoss']:.4f}")
       except Exception as e:
           print(f" Evaluation failed for {name}: {e}")

def plot_advanced_forecasts(test_data, forecasts_dict, series_idx=0):
   """Advanced plotting with multiple models and uncertainty bands"""
   fig, axes = plt.subplots(2, 2, figsize=(15, 10))
   fig.suptitle('Advanced GluonTS Forecasting Results', fontsize=16, fontweight='bold')
  
   if not forecasts_dict:
       fig.text(0.5, 0.5, 'No successful forecasts to display',
               ha='center', va='center', fontsize=20)
       return fig
  
   if series_idx < len(test_data.label):
       ts_label = test_data.label[series_idx]
       ts_input = test_data.input[series_idx]['target']
      
       colors = ['blue', 'red', 'green', 'purple', 'orange']
      
       ax1 = axes[0, 0]
       ax1.plot(range(len(ts_input)), ts_input, 'k-', label='Historical', alpha=0.8, linewidth=2)
       ax1.plot(range(len(ts_input), len(ts_input) + len(ts_label)),
               ts_label, 'k--', label='True Future', alpha=0.8, linewidth=2)
      
       for i, (name, forecasts) in enumerate(forecasts_dict.items()):
           if series_idx < len(forecasts):
               forecast = forecasts[series_idx]
               forecast_range = range(len(ts_input), len(ts_input) + len(forecast.mean))
              
               color = colors[i % len(colors)]
               ax1.plot(forecast_range, forecast.mean,
                       color=color, label=f'{name} Mean', linewidth=2)
              
               try:
                   ax1.fill_between(forecast_range,
                                  forecast.quantile(0.1), forecast.quantile(0.9),
                                  alpha=0.2, color=color, label=f'{name} 80% CI')
               except:
                   pass 
      
       ax1.set_title('Multi-Model Forecasts Comparison', fontsize=12, fontweight='bold')
       ax1.legend()
       ax1.grid(True, alpha=0.3)
       ax1.set_xlabel('Time Steps')
       ax1.set_ylabel('Value')
      
       ax2 = axes[0, 1]
       if all_forecasts:
           first_model = list(all_forecasts.keys())[0]
           if series_idx < len(all_forecasts[first_model]):
               forecast = all_forecasts[first_model][series_idx]
               ax2.scatter(ts_label, forecast.mean, alpha=0.7, s=60)
              
               min_val = min(min(ts_label), min(forecast.mean))
               max_val = max(max(ts_label), max(forecast.mean))
               ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
              
               ax2.set_title(f'Prediction vs Actual - {first_model}', fontsize=12, fontweight='bold')
               ax2.set_xlabel('Actual Values')
               ax2.set_ylabel('Predicted Values')
               ax2.grid(True, alpha=0.3)
      
       ax3 = axes[1, 0]
       if all_forecasts:
           first_model = list(all_forecasts.keys())[0]
           if series_idx < len(all_forecasts[first_model]):
               forecast = all_forecasts[first_model][series_idx]
               residuals = ts_label - forecast.mean
               ax3.hist(residuals, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
               ax3.axvline(x=0, color='r', linestyle='--', linewidth=2)
               ax3.set_title(f'Residuals Distribution - {first_model}', fontsize=12, fontweight='bold')
               ax3.set_xlabel('Residuals')
               ax3.set_ylabel('Frequency')
               ax3.grid(True, alpha=0.3)
      
       ax4 = axes[1, 1]
       if evaluation_results:
           metrics = ['MASE', 'sMAPE'] 
           model_names = list(evaluation_results.keys())
           x = np.arange(len(metrics))
           width = 0.35
          
           for i, model_name in enumerate(model_names):
               values = [evaluation_results[model_name].get(metric, 0) for metric in metrics]
               ax4.bar(x + i*width, values, width,
                      label=model_name, color=colors[i % len(colors)], alpha=0.8)
          
           ax4.set_title('Model Performance Comparison', fontsize=12, fontweight='bold')
           ax4.set_xlabel('Metrics')
           ax4.set_ylabel('Value')
           ax4.set_xticks(x + width/2 if len(model_names) > 1 else x)
           ax4.set_xticklabels(metrics)
           ax4.legend()
           ax4.grid(True, alpha=0.3)
       else:
           ax4.text(0.5, 0.5, 'No evaluation\nresults available',
                   ha='center', va='center', transform=ax4.transAxes, fontsize=14)
  
   plt.tight_layout()
   return fig


if all_forecasts and test_data.label:
   print(" Creating advanced visualizations...")
   fig = plot_advanced_forecasts(test_data, all_forecasts, series_idx=0)
   plt.show()
  
   print(f"\n Tutorial completed successfully!")
   print(f" Trained {len(trained_models)} model(s) on {len(df.columns) if 'df' in locals() else 10} time series")
   print(f" Prediction length: 30 days")
  
   if evaluation_results:
       best_model = min(evaluation_results.items(), key=lambda x: x[1]['MASE'])
       print(f" Best performing model: {best_model[0]} (MASE: {best_model[1]['MASE']:.4f})")
  
   print(f"\n Environment Status:")
   print(f"  PyTorch Support: {'' if TORCH_AVAILABLE else ''}")
   print(f"  MXNet Support: {'' if MX_AVAILABLE else ''}")
  
else:
   print("  Creating demonstration plot with synthetic data...")
  
   fig, ax = plt.subplots(1, 1, figsize=(12, 6))
  
   dates = pd.date_range('2020-01-01', periods=100, freq='D')
   ts = 100 + np.cumsum(np.random.normal(0, 2, 100)) + 20 * np.sin(np.arange(100) * 2 * np.pi / 30)
  
   ax.plot(dates[:70], ts[:70], 'b-', label='Historical Data', linewidth=2)
   ax.plot(dates[70:], ts[70:], 'r--', label='Future (Example)', linewidth=2)
   ax.fill_between(dates[70:], ts[70:] - 5, ts[70:] + 5, alpha=0.3, color='red')
  
   ax.set_title('GluonTS Probabilistic Forecasting Example', fontsize=14, fontweight='bold')
   ax.set_xlabel('Date')
   ax.set_ylabel('Value')
   ax.legend()
   ax.grid(True, alpha=0.3)
  
   plt.tight_layout()
   plt.show()
  
   print("\n Tutorial demonstrates advanced GluonTS concepts:")
   print("  • Multi-series dataset generation")
   print("  • Probabilistic forecasting")
   print("  • Model evaluation and comparison")
   print("  • Advanced visualization techniques")
   print("  • Robust error handling")
