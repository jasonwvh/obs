import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from influxdb_client import InfluxDBClient

# InfluxDB configuration
INFLUXDB_URL = os.getenv("INFLUXDB_URL", "http://localhost:8086")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN", "my-super-secret-auth-token")
INFLUXDB_ORG = os.getenv("INFLUXDB_ORG", "obs")
INFLUXDB_BUCKET = os.getenv("INFLUXDB_BUCKET", "metrics")


class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_std=1.0):
        super().__init__()
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_rho = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.bias_mu = nn.Parameter(torch.randn(out_features) * 0.1)
        self.bias_rho = nn.Parameter(torch.randn(out_features) * 0.1)
        self.prior_std = prior_std

    def forward(self, x, sample=True):
        weight = self.weight_mu
        bias = self.bias_mu
        if sample:
            weight_std = F.softplus(self.weight_rho) + 1e-6
            bias_std = F.softplus(self.bias_rho) + 1e-6
            weight = self.weight_mu + weight_std * torch.randn_like(self.weight_mu)
            bias = self.bias_mu + bias_std * torch.randn_like(self.bias_mu)
        return F.linear(x, weight, bias)

    def kl_divergence(self):
        weight_std = F.softplus(self.weight_rho) + 1e-6
        bias_std = F.softplus(self.bias_rho) + 1e-6
        prior_var = self.prior_std ** 2
        kl_weight = 0.5 * (torch.log(prior_var / weight_std ** 2) - 1 +
                           (weight_std ** 2 + self.weight_mu ** 2) / prior_var).sum()
        kl_bias = 0.5 * (torch.log(prior_var / bias_std ** 2) - 1 +
                         (bias_std ** 2 + self.bias_mu ** 2) / prior_var).sum()
        return kl_weight + kl_bias


class BayesianAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super().__init__()
        self.encoder = nn.ModuleList()
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            self.encoder.append(BayesianLinear(dims[i], dims[i + 1]))
        self.mu_layer = BayesianLinear(hidden_dims[-1], latent_dim)
        self.logvar_layer = BayesianLinear(hidden_dims[-1], latent_dim)

        self.decoder = nn.ModuleList()
        dims = [latent_dim] + hidden_dims[::-1] + [input_dim]
        for i in range(len(dims) - 1):
            self.decoder.append(BayesianLinear(dims[i], dims[i + 1]))

        self.log_sigma = nn.Parameter(torch.tensor(-1.0))
        self.activation = nn.ReLU()

    def encode(self, x, sample=True):
        for layer in self.encoder:
            x = self.activation(layer(x, sample))
        mu = self.mu_layer(x, sample)
        logvar = self.logvar_layer(x, sample)
        return mu, logvar

    def decode(self, z, sample=True):
        x = z
        for i, layer in enumerate(self.decoder[:-1]):
            x = self.activation(layer(x, sample))
        return self.decoder[-1](x, sample)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x, sample=True):
        mu, logvar = self.encode(x, sample)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, sample)
        return x_recon, mu, logvar

    def kl_divergence(self):
        kl = sum(layer.kl_divergence() for layer in self.encoder + [self.mu_layer, self.logvar_layer])
        kl += sum(layer.kl_divergence() for layer in self.decoder)
        return kl

    def compute_loss(self, x, x_recon, mu, logvar, beta_kl=0.1, beta_weights=0.01):
        mse_loss = F.mse_loss(x_recon, x, reduction='mean')
        kl_latent = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        kl_weights = self.kl_divergence() / x.size(0)
        total_loss = mse_loss + beta_kl * kl_latent + beta_weights * kl_weights
        return {'total_loss': total_loss, 'mse_loss': mse_loss, 'kl_latent': kl_latent, 'kl_weights': kl_weights}

    def predict(self, x, num_samples=50):
        self.eval()
        with torch.no_grad():
            reconstructions = torch.stack([self.forward(x, sample=True)[0] for _ in range(num_samples)])
            return reconstructions.mean(dim=0), reconstructions.std(dim=0)


def get_data(range_start="-1h", window_size=10, noise_std=0.05):
    try:
        client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
        query_api = client.query_api()
        flux_query = f'''
            from(bucket: "{INFLUXDB_BUCKET}")
                |> range(start: {range_start})
                |> filter(fn: (r) => r["_measurement"] == "system.network.io")
                |> filter(fn: (r) => r["_field"] == "gauge")
                |> filter(fn: (r) => r["device"] == "eth0")
                |> filter(fn: (r) => r["direction"] == "transmit")
                |> sort(columns: ["_time"])
        '''
        tables = query_api.query(query=flux_query)
        values = torch.tensor([record.get_value() for table in tables for record in table.records
                               if record.get_value() is not None], dtype=torch.float32)

        if len(values) < window_size:
            print(f"Insufficient data: got {len(values)} points")
            return None, None, None

        values = values[~torch.isnan(values) & ~torch.isinf(values)]
        if len(values) < window_size:
            return None, None, None

        min_val, max_val = values.min(), values.max()
        if max_val - min_val < 1e-8:
            values = values + 1e-6 * torch.randn_like(values)
            min_val, max_val = values.min(), values.max()

        values_normalized = (values - min_val) / (max_val - min_val)
        windows = torch.stack([values_normalized[i:i + window_size] for i in range(len(values) - window_size + 1)])
        windows_noisy = torch.clamp(windows + noise_std * torch.randn_like(windows), 0, 1)

        dataset = TensorDataset(windows_noisy, windows)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        client.close()
        return loader, windows_noisy, (min_val.item(), max_val.item())
    except Exception as e:
        print(f"Error querying InfluxDB: {e}")
        return None, None, None


def train_model(model, dataloader, num_epochs=1000, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for data_noisy, data_clean in dataloader:
            optimizer.zero_grad()
            x_recon, mu, logvar = model(data_noisy)
            loss_dict = model.compute_loss(data_clean, x_recon, mu, logvar)
            loss = loss_dict['total_loss']

            if not torch.isnan(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()

        losses.append(epoch_loss / len(dataloader))
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss {losses[-1]:.4f}")

    return losses


def main():
    dataloader, x_noisy, norm_info = get_data(window_size=20)
    if dataloader is None:
        return

    model = BayesianAutoencoder(input_dim=20, hidden_dims=[64, 32], latent_dim=8)
    losses = train_model(model, dataloader)
    torch.save({'model_state_dict': model.state_dict(), 'normalization_stats': norm_info}, 'model.pth')
    print(f"Final Loss: {losses[-1]:.4f}")


if __name__ == "__main__":
    main()