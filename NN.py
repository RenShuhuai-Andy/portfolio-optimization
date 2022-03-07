from deepdow.benchmarks import Benchmark, OneOverN, Random
from deepdow.callbacks import EarlyStoppingCallback
from deepdow.data import InRAMDataset, RigidDataLoader, prepare_standard_scaler, Scale
from deepdow.data.synthetic import sin_single
from deepdow.experiments import Run
from deepdow.layers import SoftmaxAllocator
from deepdow.losses import MeanReturns, SharpeRatio, MaximumDrawdown
from deepdow.visualize import generate_metrics_table, generate_weights_table, plot_metrics, plot_weight_heatmap
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
from pypfopt import expected_returns

df = pd.read_csv("data/2021Q1_Top 50_of_Mutual_Funds_20160819_to_20200115.csv", index_col=0)
returns = expected_returns.returns_from_prices(df).to_numpy()
print(returns.shape)

n_timesteps, n_assets = returns.shape[0], returns.shape[1]
lookback_timesteps, gap, lookahead_timesteps = 365, 20, 365
n_samples = n_timesteps - lookback_timesteps - lookahead_timesteps - gap + 1

split_ix = int(n_samples * 0.8)
indices_train = list(range(split_ix))
# indices_test = list(range(split_ix + lookback_timesteps + lookahead_timesteps, n_samples))
indices_test = list(range(split_ix, n_samples))

print('Train range: {}:{}\nTest range: {}:{}'.format(indices_train[0], indices_train[-1],
                                                     indices_test[0], indices_test[-1]))

X_list, y_list = [], []

for i in range(lookback_timesteps, n_timesteps - lookahead_timesteps - gap + 1):
    X_list.append(returns[i - lookback_timesteps: i, :])
    y_list.append(returns[i + gap: i + gap + lookahead_timesteps, :])

X = np.stack(X_list, axis=0)[:, None, ...]
y = np.stack(y_list, axis=0)[:, None, ...]

print('X: {}, y: {}'.format(X.shape, y.shape))

means, stds = prepare_standard_scaler(X, indices=indices_train)
print('mean: {}, std: {}'.format(means, stds))

dataset = InRAMDataset(X, y, transform=Scale(means, stds))
# dataset = InRAMDataset(X, y)
dataloader_train = RigidDataLoader(dataset,
                                   indices=indices_train,
                                   batch_size=32)
dataloader_dev = RigidDataLoader(dataset,
                                 indices=indices_test,
                                 batch_size=32)


class NeuralNetwork(torch.nn.Module, Benchmark):
    def __init__(self, n_assets, lookback, p=0.5):
        super().__init__()

        n_features = n_assets * lookback

        self.dropout_layer = torch.nn.Dropout(p=p)
        self.dense_layer = torch.nn.Linear(n_features, n_assets, bias=True)
        self.allocate_layer = SoftmaxAllocator(temperature=None)
        self.temperature = torch.nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x):
        """Perform forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Of shape (n_samples, 1, lookback, n_assets).

        Returns
        -------
        weights : torch.Torch
            Tensor of shape (n_samples, n_assets).

        """
        n_samples, _, _, _ = x.shape
        x = x.view(n_samples, -1)  # flatten features
        x = self.dropout_layer(x)
        x = self.dense_layer(x)

        temperatures = torch.ones(n_samples).to(device=x.device, dtype=x.dtype) * self.temperature
        weights = self.allocate_layer(x, temperatures)

        return weights


network = NeuralNetwork(n_assets, lookback_timesteps)
print(network)

network = network.train()  # it is the default, however, just to make the distinction clear

loss = MaximumDrawdown() + 2 * MeanReturns() + SharpeRatio()
run = Run(network,
          loss,
          dataloader_train,
          val_dataloaders={'val': dataloader_dev},
          optimizer=torch.optim.Adam(network.parameters(), amsgrad=True),
          callbacks=[EarlyStoppingCallback(metric_name='loss',
                                           dataloader_name='val',
                                           patience=15)])
history = run.launch(100)
per_epoch_results = history.metrics.groupby(['dataloader', 'metric', 'model', 'epoch'])['value']

print(per_epoch_results.count())  # double check number of samples each epoch
print(per_epoch_results.mean())  # mean loss per epoch
per_epoch_results.mean()['val']['loss']['network'].plot()
network = network.eval()
benchmarks = {
    'Uniformity': OneOverN(),  # each asset has weight 1 / n_assets
    'random': Random(),  # random allocation that is however close 1OverN
    'NeuralNetwork': network
}
metrics = {
    'MaximumDrawdown': MaximumDrawdown(),
    'Sharpe': SharpeRatio(),
    'MeanReturn': MeanReturns()
}

metrics_table = generate_metrics_table(benchmarks,
                                       dataloader_dev,
                                       metrics)
plot_metrics(metrics_table)
plt.savefig('figures/NN_metrics.pdf', bbox_inches='tight')


# evaluation
X_for_test = returns[-lookback_timesteps:, :][None, None, :, ...]
means = X_for_test.mean(axis=(0, 2, 3))
stds = X_for_test.std(axis=(0, 2, 3))
print('mean: {}, std: {}'.format(means, stds))
X_for_test = torch.tensor(X_for_test, dtype=torch.float)
X_for_test, _, _, _ = Scale(means, stds)(X_for_test, None, None, None)
predicted_weight = network(X_for_test).detach().numpy().tolist()[0]
predicted_weight = np.round(predicted_weight, 5)
predicted_weight = pd.DataFrame(predicted_weight, index=list(df.columns))
predicted_weight.to_csv("results/weights_NN.csv", header=False)
print(predicted_weight)

weight_table = generate_weights_table(network, dataloader_dev)
plot_weight_heatmap(weight_table,
                    add_sum_column=True,
                    time_format=None,
                    time_skips=25)
plt.savefig('figures/NN_weight_heatmap.pdf', bbox_inches='tight')


# sharpe_ratio: 1.62
# 2016-8-19 2020-1-15
# stock_yield: 162.17%
# volatility: 33.35%
# drawdown: 32.61%