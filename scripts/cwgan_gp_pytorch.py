"""
Conditional Wasserstein GAN with Gradient Penalty for credit-scoring tabular data.

This implementation trains on the full training set with class conditioning.
After training, call generate_samples(class_label=1, n_samples=...) to create
synthetic minority/default samples.
"""

import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


class CrossLayer(nn.Module):
    """Deep-and-cross style interaction layer."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input_dim, 1) * 0.02)
        self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return x0 * (x @ self.weight) + self.bias + x


class Generator(nn.Module):
    def __init__(
        self,
        numerical_dim: int,
        categorical_dims: List[int],
        latent_dim: int,
        n_classes: int,
        label_embed_dim: int = 16,
        hidden_dim: int = 256,
        cross_layers: int = 2,
        use_woe: bool = False,
    ):
        super().__init__()
        self.numerical_dim = numerical_dim
        self.categorical_dims = categorical_dims
        self.latent_dim = latent_dim
        self.use_woe = use_woe
        self.label_embedding = nn.Embedding(n_classes, label_embed_dim)

        input_dim = latent_dim + label_embed_dim
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
        )
        self.cross_layers = nn.ModuleList([CrossLayer(hidden_dim) for _ in range(cross_layers)])
        self.deep = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
        )
        self.numerical_head = nn.Linear(hidden_dim, numerical_dim) if numerical_dim > 0 else None
        self.categorical_heads = nn.ModuleList([nn.Linear(hidden_dim, dim) for dim in categorical_dims])

    def forward(
        self,
        z: torch.Tensor,
        labels: torch.Tensor,
        temperature: float = 0.5,
        hard: bool = False,
    ):
        label_emb = self.label_embedding(labels.long())
        x0 = self.input_layer(torch.cat([z, label_emb], dim=1))
        x = x0
        for layer in self.cross_layers:
            x = layer(x0, x)
        h = self.deep(x)

        numerical_out = None
        if self.numerical_head is not None:
            numerical_out = self.numerical_head(h)
            if not self.use_woe:
                numerical_out = torch.sigmoid(numerical_out)

        categorical_outs = []
        for head in self.categorical_heads:
            logits = head(h)
            categorical_outs.append(F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=1))
        return numerical_out, categorical_outs


class Critic(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        label_embed_dim: int = 16,
        hidden_dim: int = 256,
        cross_layers: int = 2,
    ):
        super().__init__()
        self.label_embedding = nn.Embedding(n_classes, label_embed_dim)
        model_input_dim = input_dim + label_embed_dim
        self.input_layer = nn.Sequential(
            nn.Linear(model_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
        )
        self.cross_layers = nn.ModuleList([CrossLayer(hidden_dim) for _ in range(cross_layers)])
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
        )
        self.score_head = nn.Linear(hidden_dim // 2, 1)
        self.classifier_head = nn.Linear(hidden_dim // 2, n_classes)

    def forward(self, x: torch.Tensor, labels: torch.Tensor):
        label_emb = self.label_embedding(labels.long())
        x0 = self.input_layer(torch.cat([x, label_emb], dim=1))
        h = x0
        for layer in self.cross_layers:
            h = layer(x0, h)
        h = self.shared(h)
        return self.score_head(h).view(-1), self.classifier_head(h)


@dataclass
class TrainingHistory:
    generator_loss: List[float]
    critic_loss: List[float]
    wasserstein_distance: List[float]
    gradient_penalty: List[float]
    auxiliary_loss: List[float]


class ConditionalWGAN_GP:
    def __init__(
        self,
        numerical_dim: int,
        categorical_dims: Optional[List[int]] = None,
        latent_dim: int = 64,
        n_classes: int = 2,
        label_embed_dim: int = 16,
        hidden_dim: int = 256,
        cross_layers: int = 2,
        lambda_gp: float = 10.0,
        aux_weight: float = 1.0,
        critic_aux_weight: float = 0.5,
        lr: float = 1e-4,
        betas=(0.5, 0.9),
        n_critic: int = 5,
        use_woe: bool = False,
        seed: int = 42,
        device: Optional[str] = None,
    ):
        set_seed(seed)
        self.numerical_dim = numerical_dim
        self.categorical_dims = categorical_dims or []
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.lambda_gp = lambda_gp
        self.aux_weight = aux_weight
        self.critic_aux_weight = critic_aux_weight
        self.n_critic = n_critic
        self.use_woe = use_woe
        self.seed = seed
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.output_dim = numerical_dim + sum(self.categorical_dims)

        self.generator = Generator(
            numerical_dim=numerical_dim,
            categorical_dims=self.categorical_dims,
            latent_dim=latent_dim,
            n_classes=n_classes,
            label_embed_dim=label_embed_dim,
            hidden_dim=hidden_dim,
            cross_layers=cross_layers,
            use_woe=use_woe,
        ).to(self.device)
        self.critic = Critic(
            input_dim=self.output_dim,
            n_classes=n_classes,
            label_embed_dim=label_embed_dim,
            hidden_dim=hidden_dim,
            cross_layers=cross_layers,
        ).to(self.device)

        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=betas)
        self.c_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr, betas=betas)
        self.history = TrainingHistory([], [], [], [], [])
        self.config = {
            "numerical_dim": numerical_dim,
            "categorical_dims": self.categorical_dims,
            "latent_dim": latent_dim,
            "n_classes": n_classes,
            "label_embed_dim": label_embed_dim,
            "hidden_dim": hidden_dim,
            "cross_layers": cross_layers,
            "lambda_gp": lambda_gp,
            "aux_weight": aux_weight,
            "critic_aux_weight": critic_aux_weight,
            "lr": lr,
            "n_critic": n_critic,
            "use_woe": use_woe,
            "seed": seed,
        }

    def _concat_outputs(self, numerical_out, categorical_outs) -> torch.Tensor:
        parts = []
        if numerical_out is not None:
            parts.append(numerical_out)
        parts.extend(categorical_outs)
        return torch.cat(parts, dim=1)

    def make_dataloader(self, X_train, y_train, batch_size: int = 128, shuffle: bool = True):
        X_tensor = torch.tensor(np.asarray(X_train), dtype=torch.float32)
        y_tensor = torch.tensor(np.asarray(y_train), dtype=torch.long)
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        return DataLoader(
            TensorDataset(X_tensor, y_tensor),
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=True,
            generator=generator,
        )

    def _gradient_penalty(self, real: torch.Tensor, fake: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        batch_size = real.size(0)
        alpha = torch.rand(batch_size, 1, device=self.device).expand_as(real)
        interpolated = (alpha * real + (1.0 - alpha) * fake).requires_grad_(True)
        scores, _ = self.critic(interpolated, labels)
        gradients = torch.autograd.grad(
            outputs=scores,
            inputs=interpolated,
            grad_outputs=torch.ones_like(scores),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(batch_size, -1)
        return ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()

    def train(
        self,
        X_train=None,
        y_train=None,
        train_loader=None,
        epochs: int = 300,
        batch_size: int = 128,
        n_critic: Optional[int] = None,
        lr_g: Optional[float] = None,
        lr_d: Optional[float] = None,
        temperature_start: float = 1.0,
        temperature_end: float = 0.5,
        temperature_anneal_epochs: int = 100,
        early_stopping_patience: int = 50,
        min_delta: float = 1e-4,
        verbose: bool = True,
        random_state: int = 42,
    ) -> Dict[str, List[float]]:
        set_seed(random_state)
        if train_loader is None:
            if X_train is None or y_train is None:
                raise ValueError("Provide either train_loader or X_train and y_train.")
            train_loader = self.make_dataloader(X_train, y_train, batch_size=batch_size)

        if n_critic is None:
            n_critic = self.n_critic
        if lr_g is not None:
            for group in self.g_optimizer.param_groups:
                group["lr"] = lr_g
        if lr_d is not None:
            for group in self.c_optimizer.param_groups:
                group["lr"] = lr_d

        best_plateau_value = None
        plateau_count = 0

        print(f"Using device: {self.device}")
        print(f"Training cWGAN-GP on full class-conditioned data for up to {epochs} epochs")

        for epoch in range(epochs):
            if epoch < temperature_anneal_epochs:
                temperature = temperature_start - (temperature_start - temperature_end) * (
                    epoch / max(temperature_anneal_epochs, 1)
                )
            else:
                temperature = temperature_end

            epoch_g, epoch_c, epoch_w, epoch_gp, epoch_aux = [], [], [], [], []
            for real, labels in train_loader:
                real = real.to(self.device)
                labels = labels.to(self.device)
                current_batch = real.size(0)

                for _ in range(n_critic):
                    z = torch.randn(current_batch, self.latent_dim, device=self.device)
                    numerical_fake, categorical_fake = self.generator(
                        z, labels, temperature=temperature, hard=False
                    )
                    fake = self._concat_outputs(numerical_fake, categorical_fake).detach()

                    real_score, real_cls = self.critic(real, labels)
                    fake_score, _ = self.critic(fake, labels)
                    gp = self._gradient_penalty(real, fake, labels)
                    aux_real = F.cross_entropy(real_cls, labels)
                    wasserstein_distance = real_score.mean() - fake_score.mean()
                    critic_loss = (
                        -wasserstein_distance
                        + self.lambda_gp * gp
                        + self.critic_aux_weight * aux_real
                    )

                    self.c_optimizer.zero_grad(set_to_none=True)
                    critic_loss.backward()
                    self.c_optimizer.step()

                z = torch.randn(current_batch, self.latent_dim, device=self.device)
                numerical_fake, categorical_fake = self.generator(z, labels, temperature=temperature, hard=False)
                fake = self._concat_outputs(numerical_fake, categorical_fake)
                fake_score, fake_cls = self.critic(fake, labels)
                aux_fake = F.cross_entropy(fake_cls, labels)
                generator_loss = -fake_score.mean() + self.aux_weight * aux_fake

                self.g_optimizer.zero_grad(set_to_none=True)
                generator_loss.backward()
                self.g_optimizer.step()

                epoch_g.append(float(generator_loss.detach().cpu()))
                epoch_c.append(float(critic_loss.detach().cpu()))
                epoch_w.append(float(wasserstein_distance.detach().cpu()))
                epoch_gp.append(float(gp.detach().cpu()))
                epoch_aux.append(float(aux_fake.detach().cpu()))

            mean_g = float(np.mean(epoch_g))
            mean_c = float(np.mean(epoch_c))
            mean_w = float(np.mean(epoch_w))
            mean_gp = float(np.mean(epoch_gp))
            mean_aux = float(np.mean(epoch_aux))
            self.history.generator_loss.append(mean_g)
            self.history.critic_loss.append(mean_c)
            self.history.wasserstein_distance.append(mean_w)
            self.history.gradient_penalty.append(mean_gp)
            self.history.auxiliary_loss.append(mean_aux)

            if verbose and (epoch == 0 or (epoch + 1) % 10 == 0):
                print(
                    f"Epoch {epoch + 1:04d}/{epochs} | "
                    f"G: {mean_g:.4f} | C: {mean_c:.4f} | "
                    f"W-dist: {mean_w:.4f} | GP: {mean_gp:.4f} | Aux: {mean_aux:.4f}"
                )

            # Plateau detection uses the absolute Wasserstein distance change, not direction.
            if best_plateau_value is None or abs(mean_w - best_plateau_value) > min_delta:
                best_plateau_value = mean_w
                plateau_count = 0
                self.best_state = {
                    "generator": self.generator.state_dict(),
                    "critic": self.critic.state_dict(),
                    "epoch": epoch,
                }
            else:
                plateau_count += 1

            if plateau_count >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}; Wasserstein distance plateaued.")
                break

        if hasattr(self, "best_state"):
            self.generator.load_state_dict(self.best_state["generator"])
            self.critic.load_state_dict(self.best_state["critic"])
        return self.history.__dict__

    @torch.no_grad()
    def generate_samples(
        self,
        class_label: int,
        n_samples: int,
        temperature: float = 0.3,
        batch_size: int = 512,
    ) -> np.ndarray:
        self.generator.eval()
        outputs = []
        remaining = n_samples
        while remaining > 0:
            current = min(batch_size, remaining)
            z = torch.randn(current, self.latent_dim, device=self.device)
            labels = torch.full((current,), class_label, dtype=torch.long, device=self.device)
            numerical_out, categorical_outs = self.generator(
                z, labels, temperature=temperature, hard=True
            )
            outputs.append(self._concat_outputs(numerical_out, categorical_outs).cpu().numpy())
            remaining -= current
        self.generator.train()
        return np.vstack(outputs)

    def save(self, save_path: str) -> None:
        os.makedirs(save_path, exist_ok=True)
        torch.save(
            {
                "generator_state_dict": self.generator.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "g_optimizer_state_dict": self.g_optimizer.state_dict(),
                "c_optimizer_state_dict": self.c_optimizer.state_dict(),
                "history": self.history.__dict__,
                "config": self.config,
            },
            os.path.join(save_path, "cwgan_model.pt"),
        )
        joblib.dump(self.config, os.path.join(save_path, "metadata.pkl"))
        print(f"Model saved to {save_path}")

    @classmethod
    def load(cls, save_path: str, device: Optional[str] = None):
        config = joblib.load(os.path.join(save_path, "metadata.pkl"))
        if device is not None:
            config["device"] = device
        gan = cls(**config)
        checkpoint = torch.load(os.path.join(save_path, "cwgan_model.pt"), map_location=gan.device)
        gan.generator.load_state_dict(checkpoint["generator_state_dict"])
        gan.critic.load_state_dict(checkpoint["critic_state_dict"])
        gan.g_optimizer.load_state_dict(checkpoint["g_optimizer_state_dict"])
        gan.c_optimizer.load_state_dict(checkpoint["c_optimizer_state_dict"])
        history = checkpoint.get("history", {})
        gan.history = TrainingHistory(
            history.get("generator_loss", []),
            history.get("critic_loss", []),
            history.get("wasserstein_distance", []),
            history.get("gradient_penalty", []),
            history.get("auxiliary_loss", []),
        )
        print(f"Model loaded from {save_path}")
        return gan

    def plot_training_history(self, save_path: Optional[str] = None):
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes[0, 0].plot(self.history.generator_loss)
        axes[0, 0].set_title("Generator Loss")
        axes[0, 1].plot(self.history.critic_loss)
        axes[0, 1].set_title("Critic Loss")
        axes[1, 0].plot(self.history.wasserstein_distance)
        axes[1, 0].set_title("Wasserstein Distance")
        axes[1, 1].plot(self.history.gradient_penalty)
        axes[1, 1].axhline(0.05, color="red", linestyle="--", linewidth=1)
        axes[1, 1].set_title("Gradient Penalty")
        for ax in axes.ravel():
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("Epoch")
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
