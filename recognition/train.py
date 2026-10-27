"""Three-stage TimeGAN training; scaling and minibatches use training rows only."""
from pathlib import Path

import torch
import torch.nn.functional as F
from itertools import chain
import numpy as np

from torch.utils.data import DataLoader
from dataset import LOBSTERLevel10Dataset, WindowDataset
from modules import LATENT_DIMS
from timegan import TimeGAN


def fit(data, steps=(1000, 1000, 5000), window=64, batch_size=64, seed=20261025):
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    features = data.clone()
    features[:, 0] = torch.diff(data[:, 0], prepend=data[:1, 0])
    center, scale = features.mean(0), features.std(0).clamp_min(1e-8)
    values = ((features - center) / scale).float().to(device)
    model = TimeGAN().to(device)
    encoder, decoder, generator, supervisor, discriminator = (
        model.encoder, model.decoder, model.generator, model.supervisor, model.discriminator)
    modules = ((encoder, decoder), (supervisor,), (generator, supervisor), (discriminator,))
    optimizers = [torch.optim.Adam(chain(*(m.parameters() for m in group)), lr=1e-3) for group in modules]
    history = []

    def update(loss, optimizer):
        model.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_((p for group in optimizer.param_groups for p in group['params']), 1.0)
        optimizer.step()

    def losses(x):
        h, e = model.encode(x), generator(torch.randn(len(x), window, model._latent_dims, device=device))
        fake = supervisor(e)
        supervised = F.mse_loss(supervisor(h)[:, :-1], h[:, 1:])
        reconstruction = F.mse_loss(model.decode(h), x)
        synthetic = model.decode(fake)
        moments = (synthetic.mean((0, 1)) - x.mean((0, 1))).abs().mean()
        moments = moments + ((synthetic.var((0, 1), unbiased=False) + 1e-6).sqrt() - (x.var((0, 1), unbiased=False) + 1e-6).sqrt()).abs().mean()
        real_logits, fake_logits, noise_logits = (discriminator(z) for z in (h, fake, e))
        bce = F.binary_cross_entropy_with_logits
        adversarial = bce(fake_logits, torch.ones_like(fake_logits)) + bce(noise_logits, torch.ones_like(noise_logits))
        discrimination = bce(real_logits, torch.ones_like(real_logits)) + bce(fake_logits, torch.zeros_like(fake_logits)) + bce(noise_logits, torch.zeros_like(noise_logits))
        return reconstruction, supervised, adversarial + 100 * (supervised + 1e-8).sqrt() + 100 * moments, discrimination

    # Minibatch windows train the networks; prediction later embeds the entire history.
    for stage, count in enumerate(steps):
        for step in range(count):
            starts = torch.randint(len(values) - window + 1, (batch_size,), device=device)
            x = values[starts[:, None] + torch.arange(window, device=device)]
            if stage == 0:
                loss = 10 * (F.mse_loss(model.decode(model.encode(x)), x) + 1e-8).sqrt()
                update(loss, optimizers[0])
            elif stage == 1:
                h = model.encode(x).detach()
                loss = F.mse_loss(supervisor(h)[:, :-1], h[:, 1:])
                update(loss, optimizers[1])
            else:
                reconstruction, supervised, loss, _ = losses(x)
                update(loss, optimizers[2])
                reconstruction, supervised, _, _ = losses(x)
                update(10 * (reconstruction + 1e-8).sqrt() + 0.1 * supervised, optimizers[0])
                _, _, _, discrimination = losses(x)
                if discrimination.item() > 0.15:
                    update(discrimination, optimizers[3])
            if step % 100 == 0 or step == count - 1:
                history.append({"stage": stage, "step": step + 1, "loss": loss.item()})
                print(history[-1], flush=True)
    return model.eval(), values, center, scale, history


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
                return torch.load(f, weights_only=False)

    torch.manual_seed(seed)

    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

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

        print(f"Finished training epoch {epoch + 1}; recon. loss (batch mean): {np.mean(batch_losses)}")

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
        print(f"Finished training epoch {epoch + 1}; supervised loss (batch mean): {np.mean(batch_losses)}")

    with open(filename, "wb") as f:
        torch.save(model, f)
    print(f"Saved model to {filename}")
    print(f"Done pre-training")

    return model

def train(dataset: WindowDataset) -> None:
    epochs = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    model = pre_train(dataset, device=device, dtype=dtype)

    torch.manual_seed(0)

    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    batch: torch.Tensor
    labels: torch.Tensor

    gen_optim = torch.optim.Adam(chain(model.generator.parameters(), model.supervisor.parameters()), lr=3e-4)
    recon_optim = torch.optim.Adam(chain(model.encoder.parameters(), model.decoder.parameters()), lr=3e-4)
    dis_optim = torch.optim.Adam(model.discriminator.parameters(), lr=3e-4)

    for epoch in range(epochs):
        dis_batch_losses: list[float] = []
        gen_batch_losses: list[float] = []
        for batch, labels in data_loader:
            batch = batch.to(device, dtype)
            labels = labels.to(device, dtype)
            both_windows = torch.cat((batch, labels[:, None, :]), dim=1)

            # Two generator/autoencoder update pairs precede each discriminator update.
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
                gen_loss = adv_gen_loss + 100 * (supervised_loss + 1e-8).sqrt() + 100 * moment_loss
                model.zero_grad()
                gen_loss.backward()
                gen_optim.step()
                gen_batch_losses.append(gen_loss.item())

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
            adv_dis_loss_fake = F.binary_cross_entropy(
                model.discriminate(h_fake),
                torch.zeros(*batch.shape[:2], 1, device=device, dtype=dtype),
            )

            adv_dis_loss_fake_e = F.binary_cross_entropy(
                model.discriminate(e_fake),
                torch.zeros(*batch.shape[:2], 1, device=device, dtype=dtype),
            )

            adv_dis_loss = adv_dis_loss_real + adv_dis_loss_fake + adv_dis_loss_fake_e
            dis_optim.zero_grad()
            if adv_dis_loss.item() > 0.15:
                adv_dis_loss.backward()
                dis_optim.step()

            dis_batch_losses.append(adv_dis_loss.item() / 3)

        print(f"Finished training epoch {epoch + 1}; dis. loss (batch mean): {np.mean(dis_batch_losses)}; gen. loss (batch mean): {np.mean(gen_batch_losses)}")

    torch.save(model, "timegan.pth")

if __name__ == "__main__":
    # First standardize dataset
    data = LOBSTERLevel10Dataset(train=True).data
    mean, std = data.mean(0), data.std(0).clamp_min(1e-3)
    data = (data.clone() - mean[None, ...]) / std[None, ...]

    dataset = WindowDataset(data)
    train(dataset)
