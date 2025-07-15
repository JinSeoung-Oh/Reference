### From https://medium.com/chat-gpt-now-writes-all-my-articles/easier-bayesian-inference-with-neural-networks-using-bayesflow-code-included-792c2e8bf177

"""
BayesFlow is an open-source Python library designed to accelerate and scale Bayesian inference using amortized neural networks. By training neural networks to “learn” the inverse problem (inferring parameters from data) or the forward model (generating data from parameters), 
BayesFlow enables near-instantaneous inference after initial training — often in milliseconds.
"""

import numpy as np
from pathlib import Path
import keras
import bayesflow as bf

# Set precision for outputs
np.set_printoptions(suppress=True)

####################### Defining the Generative Model
def likelihood(beta, sigma, N):
    x = np.random.normal(0, 1, size=N)
    y = np.random.normal(beta[0] + beta[1] * x, sigma, size=N)
    return dict(y=y, x=x)

def prior():
    beta = np.random.normal([2, 0], [3, 1])
    sigma = np.random.gamma(1, 1)
    return dict(beta=beta, sigma=sigma)

def meta():
    N = np.random.randint(5, 15)
    return dict(N=N)

simulator = bf.simulators.make_simulator([prior, likelihood], meta_fn=meta)
sim_draws = simulator.sample(500)

####################### Data Preparation via Adapters
adapter = (
    bf.Adapter()
    .broadcast("N", to="x")
    .as_set(["x", "y"])
    .constrain("sigma", lower=0)
    .standardize(exclude=["N"])
    .sqrt("N")
    .convert_dtype("float64", "float32")
    .concatenate(["beta", "sigma"], into="inference_variables")
    .concatenate(["x", "y"], into="summary_variables")
    .rename("N", "inference_conditions")
)

processed_draws = adapter(sim_draws)

####################### Building Neural Networks
summary_net = bf.networks.DeepSet(input_shape=(None, 2), output_dim=64)
inference_net = bf.networks.InvertibleNetwork(n_params=3, num_coupling_layers=6)


amortizer = bf.amortizers.AmortizedPosterior(
    summary_net=summary_net,
    inference_net=inference_net
)

amortizer.compile(optimizer="adam")
amortizer.train(processed_draws, epochs=30, batch_size=64)

test_data = adapter(simulator.sample(1))
posterior_samples = amortizer.sample(test_data["summary_variables"], 
                                     conditions=test_data["inference_conditions"],
                                     n_samples=1000)

bf.diagnostics.plots.pairs_samples(
    samples=posterior_samples,
    variable_names=[r"$\beta_0$", r"$\beta_1$", r"$\sigma$"]
)







