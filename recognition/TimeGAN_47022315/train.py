"""Three-stage TimeGAN training; scaling and minibatches use training rows only."""
from pathlib import Path

import torch
import torch.nn.functional as F
from itertools import chain
import numpy as np

from torch.utils.data import DataLoader
from dataset import LOBSTERLevel10Dataset, WindowDataset
from modules import LATENT_DIMS, TimeGAN


def soft_histogram(values: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """Soft bin probabilities; sigmoid boundaries give gradients and open tails."""
    cdf = torch.sigmoid((edges[1:-1] - values.reshape(-1, 1)) / (1e-2 * (edges[1] - edges[0])))
    cdf = torch.cat((torch.zeros_like(cdf[:, :1]), cdf, torch.ones_like(cdf[:, :1])), dim=1)
    probabilities = cdf.diff(dim=1).mean(0) + 1e-8
    return probabilities / probabilities.sum()


def pre_train(
    dataset: WindowDataset,
    *,
    epochs: int = 20,
    device: torch.types.Device = "cuda",
    dtype: torch.dtype = torch.float32,
    filename: str = "pretrained.pth",
    seed: int = 0,
) -> TimeGAN:
    if filename:
        path = Path(filename)
        if path.is_file():
            print("Loading pre-trained model from fs cache")
            with open(path, "rb") as f:
                model = torch.load(f, map_location=device, weights_only=False)
            return model

    torch.manual_seed(seed)

    data_loader = DataLoader(dataset, batch_size=128, shuffle=True)

    model = TimeGAN().to(device, dtype)

    recon_optim = torch.optim.Adam(chain(model.encoder.parameters(), model.decoder.parameters()), lr=3e-4)
    supervisor_optim = torch.optim.Adam(model.supervisor.parameters(), lr=3e-4)

    batch_losses: list[float]

    batch: torch.Tensor
    labels: torch.Tensor

    print("Beginning auto-encoder pre-training")
    for epoch in range(epochs):
        batch_losses = []
        for batch, labels in data_loader:
            batch = batch.to(device, dtype)
            labels = labels.to(device, dtype)

            recon = model.decode(model.encode(batch))
            recon_loss = F.mse_loss(recon, batch)
            recon_optim.zero_grad()
            recon_loss.backward()
            recon_optim.step()

            batch_losses.append((10 * recon_loss.sqrt()).item())

        print(f"Finished training epoch {epoch + 1}; recon. loss (batch mean): {np.mean(batch_losses):.4f}")

    print("Begin supervisor pre-training")
    for epoch in range(epochs):
        batch_losses = []
        for batch, labels in data_loader:
            batch = batch.to(device, dtype)
            labels = labels.to(device, dtype)

            both_windows = torch.cat((batch, labels[:, None, :]), dim=1)
            with torch.no_grad():
                h = model.encode(both_windows)
            h_current = h[:, :-1, :]
            h_next = h[:, 1:, :]
            pred = model.supervisor(h_current)
            pred_loss = F.mse_loss(pred, h_next)
            supervisor_optim.zero_grad()
            pred_loss.backward()
            supervisor_optim.step()

            batch_losses.append((10 * pred_loss.sqrt()).item())
        print(f"Finished training epoch {epoch + 1}; supervised loss (batch mean): {np.mean(batch_losses):.4f}")

    with open(filename, "wb") as f:
        torch.save(model, f)
    print(f"Saved model to {filename}")
    print(f"Done pre-training")

    return model

def train(
    dataset: WindowDataset,
    *,
    epochs: int = 20,
    filename: str = "timegan.pth",
    balance: float | None = None,
    smoothing: float = 0.0,
    balance_total: bool = False,
    max_streak: int | None = None,
    pretrain_filename: str = "pretrained.pth",
    pretrain_epochs: int = 20,
    kl_weight: float = 1.0,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    model = pre_train(
        dataset,
        device=device,
        dtype=dtype,
        epochs=pretrain_epochs,
        filename=pretrain_filename,
    )

    torch.manual_seed(0)

    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    # Fix the soft reference from training rows once, without overlapping windows.
    training_returns = torch.stack([row[0] for row in dataset.data]).detach().to(device, dtype)
    low, high = training_returns.min(), training_returns.max()
    kl_edges = torch.linspace(low, low + (high - low).clamp_min(1e-6), 51, device=device, dtype=dtype)
    kl_target = soft_histogram(training_returns, kl_edges).detach()

    batch: torch.Tensor
    labels: torch.Tensor

    gen_optim = torch.optim.Adam(chain(model.generator.parameters(), model.supervisor.parameters()), lr=3e-4)
    recon_optim = torch.optim.Adam(chain(model.encoder.parameters(), model.decoder.parameters()), lr=3e-4)
    dis_optim = torch.optim.Adam(model.discriminator.parameters(), lr=1e-4)
    previous, rates, streak = None, np.ones(2), 0

    for epoch in range(epochs):
        dis_batch_losses: list[float] = []
        gen_batch_losses: list[float] = []
        gen_components: list[tuple[float, float, float, float]] = []
        updates = [0, 0]
        for batch, labels in data_loader:
            batch = batch.to(device, dtype)
            labels = labels.to(device, dtype)
            both_windows = torch.cat((batch, labels[:, None, :]), dim=1)
            update_generator = balance is None or rates[1] <= balance * rates[0] or (previous is not None and previous[1] <= 0.15)
            if balance is not None and max_streak is not None and abs(streak) >= max_streak:
                update_generator = streak < 0 or (previous is not None and previous[1] <= 0.15)  # Respect D pauses.
            streak = max(streak, 0) + 1 if update_generator else min(streak, 0) - 1

            # Keep two autoencoder updates; use G when TimeGAN already pauses D below 0.15.
            for _ in range(2):
                with torch.no_grad():
                    h = model.encode(both_windows)

                # Both generated latent sequences should look real to the discriminator.
                e_fake = model.generator(torch.randn(*batch.shape[:2], LATENT_DIMS, device=device, dtype=dtype))
                h_fake = model.supervisor(e_fake)
                adv_gen_loss = F.binary_cross_entropy(
                    model.discriminate(h_fake),
                    torch.ones(*batch.shape[:2], 1, device=device, dtype=dtype),
                )
                adv_gen_loss_e = F.binary_cross_entropy(
                    model.discriminate(e_fake),
                    torch.ones(*batch.shape[:2], 1, device=device, dtype=dtype),
                )
                adv_gen_loss = adv_gen_loss + adv_gen_loss_e
                supervised_loss = F.mse_loss(model.supervisor(h[:, :-1]), h[:, 1:])
                synthetic = model.decode(h_fake)
                # Match batch means and standard deviations at each timestep.
                moment_loss = (synthetic.mean(0) - batch.mean(0)).abs().mean()
                real_std = (batch.var(0, unbiased=False) + 1e-6).sqrt()
                fake_std = (synthetic.var(0, unbiased=False) + 1e-6).sqrt()
                moment_loss = moment_loss + (fake_std - real_std).abs().mean()
                kl_loss = F.kl_div(soft_histogram(synthetic[..., 0], kl_edges).log(), kl_target, reduction="sum")
                gen_loss = adv_gen_loss + 100 * (supervised_loss + 1e-8).sqrt() + 100 * moment_loss + kl_weight * kl_loss
                model.zero_grad()
                if update_generator:
                    gen_loss.backward()
                    gen_optim.step()
                    updates[0] += 1
                gen_batch_losses.append(gen_loss.item())
                # Log weighted TimeGAN terms and the raw soft midpoint-return KL.
                gen_components.append((adv_gen_loss.item(), 100 * (supervised_loss.item() + 1e-8) ** 0.5,
                                       100 * moment_loss.item(), kl_loss.item()))

                # Recompute embeddings after the supervisor update, retaining encoder gradients.
                h = model.encode(both_windows)
                recon_loss = F.mse_loss(model.decode(h[:, :-1]), batch)
                supervised_loss = F.mse_loss(model.supervisor(h[:, :-1]), h[:, 1:])
                model.zero_grad()
                (10 * (recon_loss + 1e-8).sqrt() + 0.1 * supervised_loss).backward()
                recon_optim.step()

            with torch.no_grad():
                h_real = model.encode(batch)
            adv_dis_loss_real = F.binary_cross_entropy(
                model.discriminate(h_real),
                torch.ones(*batch.shape[:2], 1, device=device, dtype=dtype),
            )

            with torch.no_grad():
                e_fake = model.generator(torch.randn(*batch.shape[:2], LATENT_DIMS, device=device, dtype=dtype))
                h_fake = model.supervisor(e_fake)
            fake_probability = model.discriminate(h_fake)
            adv_dis_loss_fake = F.binary_cross_entropy(
                fake_probability,
                torch.zeros(*batch.shape[:2], 1, device=device, dtype=dtype),
            )

            noise_probability = model.discriminate(e_fake)
            adv_dis_loss_fake_e = F.binary_cross_entropy(
                noise_probability,
                torch.zeros(*batch.shape[:2], 1, device=device, dtype=dtype),
            )

            adv_dis_loss = adv_dis_loss_real + adv_dis_loss_fake + adv_dis_loss_fake_e
            dis_optim.zero_grad()
            if adv_dis_loss.item() > 0.15 and (balance is None or not update_generator):
                adv_dis_loss.backward()
                dis_optim.step()
                updates[1] += 1

            dis_batch_losses.append(adv_dis_loss.item() / 3)
            if balance is not None:
                # BCE adaptation of arXiv:2008.12463: D updates when r_D > lambda * r_G.
                paired_adversarial = (F.binary_cross_entropy(fake_probability.detach(), torch.ones_like(fake_probability))
                                      + F.binary_cross_entropy(noise_probability.detach(), torch.ones_like(noise_probability)))
                current = np.array((gen_loss.item() if balance_total else paired_adversarial.item(), adv_dis_loss.item()))
                if previous is not None:
                    change = np.abs(current - previous) / np.maximum(np.abs(previous), 1e-8)
                    rates = smoothing * rates + (1 - smoothing) * change
                previous = current

        adversarial, supervised, moment, midpoint_kl = np.mean(gen_components, axis=0)
        print(f"Finished training epoch {epoch + 1}; dis. loss (batch mean): {np.mean(dis_batch_losses):.4f}; "
              f"gen. loss (batch mean): {np.mean(gen_batch_losses):.4f}; adversarial: {adversarial:.4f}; "
              f"supervised (weighted): {supervised:.4f}; moment (weighted): {moment:.4f}; midpoint KL (soft): {midpoint_kl:.4f}; G/D updates: {updates[0]}/{updates[1]}", flush=True)

    torch.save(model, filename)

if __name__ == "__main__":

    torch.set_num_threads(4)
    data = LOBSTERLevel10Dataset(train=True).data
    dataset = WindowDataset(data, window_size=512, stride=32)

    # The displayed validation plots used joint epoch 4 after both pretraining stages.
    train(dataset, epochs=20, pretrain_epochs=20, filename="timegan.pth", pretrain_filename="pretrained.pth", balance=0.5)
