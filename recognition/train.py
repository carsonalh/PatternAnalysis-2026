"""Three-stage TimeGAN training; scaling and minibatches use training rows only."""
import torch
import torch.nn.functional as F
from itertools import chain
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
        model._encoder, model._decoder, model._generator, model._supervisor, model._discriminator)
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
