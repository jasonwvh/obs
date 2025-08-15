import os
import torch
import torch.nn as nn
import torch.distributions as dist
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from matplotlib import pyplot as plt
from influxdb_client import InfluxDBClient
import logging

# --- InfluxDB Configuration ---
INFLUXDB_URL = os.getenv("INFLUXDB_URL", "http://localhost:8086")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN", "my-super-secret-auth-token")
INFLUXDB_ORG = os.getenv("INFLUXDB_ORG", "obs")
INFLUXDB_BUCKET = os.getenv("INFLUXDB_BUCKET", "metrics")

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_mean=0.0, prior_std=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Parameterize weight distribution (e.g., mean-field Gaussian variational posterior)
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.randn(out_features, in_features))  # For softplus to get std

        self.bias_mu = nn.Parameter(torch.randn(out_features))
        self.bias_rho = nn.Parameter(torch.randn(out_features))

        # Priors (fixed Gaussians)
        self.prior_mean = prior_mean
        self.prior_std = prior_std

    def forward(self, x, sample=True):
        # Sample weights from variational posterior if sampling, else use mean
        if sample:
            # Sample epsilon from standard normal
            epsilon_weight = torch.randn_like(self.weight_mu)
            epsilon_bias = torch.randn_like(self.bias_mu)

            # Compute std from rho (reparameterization: std = softplus(rho))
            weight_std = torch.nn.functional.softplus(self.weight_rho)
            bias_std = torch.nn.functional.softplus(self.bias_rho)

            # Reparameterization trick: weight = mu + std * epsilon
            weight = self.weight_mu + weight_std * epsilon_weight
            bias = self.bias_mu + bias_std * epsilon_bias
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return torch.matmul(x, weight.t()) + bias

    def kl_divergence(self):
        # Compute KL divergence between variational posterior and prior for weights and biases
        # For Gaussian: KL = 0.5 * (log(prior_var / q_var) - 1 + (q_var + (q_mu - prior_mu)^2) / prior_var)

        weight_std = torch.nn.functional.softplus(self.weight_rho)
        bias_std = torch.nn.functional.softplus(self.bias_rho)

        prior_var = self.prior_std ** 2
        weight_var = weight_std ** 2
        bias_var = bias_std ** 2

        kl_weight = 0.5 * (torch.log(prior_var / weight_var) - 1 + (
                    weight_var + (self.weight_mu - self.prior_mean) ** 2) / prior_var)
        kl_bias = 0.5 * (torch.log(prior_var / bias_var) - 1 + (
                    bias_var + (self.bias_mu - self.prior_mean) ** 2) / prior_var)

        return kl_weight.sum() + kl_bias.sum()


class BayesianNN(nn.Module):
    def __init__(self, window_size, num_features, hidden_dim, output_dim=1):
        super().__init__()
        input_dim = window_size * num_features  # Flatten window
        self.layer1 = BayesianLinear(input_dim, hidden_dim)
        self.layer2 = BayesianLinear(hidden_dim, output_dim)
        self.activation = nn.ReLU()
        self.log_sigma = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, sample=True):
        # x shape: (batch_size, window_size, num_features)
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, window_size * num_features)
        x = self.activation(self.layer1(x, sample))
        x = self.layer2(x, sample)  # Output: (batch_size, output_dim)
        return x

    def kl_divergence(self):
        return self.layer1.kl_divergence() + self.layer2.kl_divergence()

    def predict(self, x_test, num_samples=100):
        with torch.no_grad():
            preds = [self.forward(x_test, sample=True) for _ in range(num_samples)]
            preds = torch.stack(preds)
            mean_pred = preds.mean(dim=0)
            epistemic_std = preds.std(dim=0)
            aleatoric_std = torch.exp(0.5 * self.log_sigma)
            return mean_pred, epistemic_std, aleatoric_std


def elbo_loss(model, x, y, num_samples=1, num_data_points=1):
    # For variational inference: ELBO = E[log p(y|x,w)] - KL(q(w)||p(w))
    # Approximate expectation with Monte Carlo sampling

    log_likelihood = torch.tensor(0.0, dtype=x.dtype, device=x.device)
    for _ in range(num_samples):
        preds = model(x, sample=True)
        # Assume Gaussian likelihood for regression; adjust for other tasks
        likelihood = dist.Normal(preds, scale=torch.exp(0.5 * model.log_sigma)).log_prob(y).sum()
        log_likelihood += likelihood / num_samples

    kl = model.kl_divergence()

    # Scale KL by 1/N for mini-batches (amortized over dataset size)
    elbo = log_likelihood - (kl / num_data_points)
    return -elbo  # Minimize negative ELBO


def train_bnn(model, train_loader, epochs=100, lr=0.01, num_samples=3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    num_data_points = len(train_loader.dataset)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = elbo_loss(model, x, y, num_samples, num_data_points)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")

def get_training_data(range_start="-1h", window_size=10):
    client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
    query_api = client.query_api()
    flux_query = f'''
        from(bucket: "{INFLUXDB_BUCKET}")
            |> range(start: {range_start})
            |> filter(fn: (r) => r["_measurement"] == "system.network.io")
            |> filter(fn: (r) => r["_field"] == "gauge")
            |> filter(fn: (r) => r["device"] == "eth0")
            |> filter(fn: (r) => r["direction"] == "transmit")
    '''
    tables = query_api.query(query=flux_query)
    values = [record.get_value() for table in tables for record in table.records]
    if not values:
        return None, None, None

    # Create sliding windows
    x, y = [], []
    for i in range(len(values) - window_size):
        x.append(values[i:i + window_size])
        y.append(values[i + window_size])
    x = torch.tensor(x, dtype=torch.float32)  # Shape: (samples, window_size, 1)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # Shape: (samples, 1)
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    return loader, x, y


def generate_sample_dataset(num_samples=200, noise_std=0.2):
    # Inputs: uniformly distributed in [-1, 1]
    x = torch.linspace(-1, 1, num_samples).unsqueeze(1)  # Shape: [num_samples, input_dim=1]

    # Targets: sinusoidal with Gaussian noise
    y = torch.sin(2 * torch.pi * x) + noise_std * torch.randn_like(x)  # Shape: [num_samples, 1]

    # Create TensorDataset and DataLoader
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    return loader, x, y  # Return loader and raw data for plotting if needed

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    window_size = 10
    num_features = 1
    hidden_dim = 50
    output_dim = 1
    model = BayesianNN(window_size, num_features, hidden_dim, output_dim)

    train_loader, x, y = get_training_data()
    train_bnn(model, train_loader, epochs=1000, lr=0.5, num_samples=100)
    torch.save(model.state_dict(), 'model.pth')
    logging.info("Model training complete. Model saved to 'model.pth'")

    # --- Prediction and Plotting ---
    # if x is not None:
    #     x_test = torch.linspace(-1, 1, 100).unsqueeze(1)
    #     mean_pred, std_pred = model.predict(x_test)
    #
    #     plt.figure(figsize=(10, 6))
    #     plt.scatter(x.numpy(), y.numpy(), alpha=0.5, label="Training Data")
    #     plt.plot(x_test.numpy(), mean_pred.numpy(), color='red', label="Mean Prediction")
    #     plt.fill_between(x_test.numpy().flatten(),
    #                      (mean_pred - 2 * std_pred).numpy().flatten(),
    #                      (mean_pred + 2 * std_pred).numpy().flatten(),
    #                      color='red', alpha=0.2, label="Uncertainty (2 std)")
    #     plt.title("Bayesian Neural Network Predictions")
    #     plt.xlabel("Normalized Time")
    #     plt.ylabel("Value")
    #     plt.legend()
    #     plt.show()
    # --- End Prediction and Plotting ---
