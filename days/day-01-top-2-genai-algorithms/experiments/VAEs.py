import os, math
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, utils as vutils

LATENT_DIM = 20
HIDDEN = 400
BATCH_SIZE = 128
EPOCHS = 1
LR = 1e-3
DATA_DIR = "./data"
OUT_DIR = "samples"
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

transform = transforms.ToTensor()  # [0,1]
train_ds = datasets.MNIST(root=DATA_DIR, train=True, transform=transform, download=True)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)

class VAE(nn.Module):
    def __init__(self, in_dim=784, h=HIDDEN, z=LATENT_DIM):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, h), nn.ReLU(),
        )
        self.mu = nn.Linear(h, z)
        self.logvar = nn.Linear(h, z)
        self.dec = nn.Sequential(
            nn.Linear(z, h), nn.ReLU(),
            nn.Linear(h, in_dim), nn.Sigmoid()  
        )

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
bce = nn.BCELoss(reduction="sum")  

def vae_loss(recon, x, mu, logvar):
    recon_loss = bce(recon, x)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl, recon_loss, kl

model.train()
for epoch in range(1, EPOCHS + 1):
    total, total_recon, total_kl = 0.0, 0.0, 0.0
    for imgs, _ in train_loader:
        imgs = imgs.view(imgs.size(0), -1).to(device)  
        optimizer.zero_grad()
        recon, mu, logvar = model(imgs)
        loss, r, k = vae_loss(recon, imgs, mu, logvar)
        loss.backward()
        optimizer.step()
        total += loss.item()
        total_recon += r.item()
        total_kl += k.item()
    n = len(train_loader.dataset if not isinstance(train_loader.dataset, Subset) else train_loader.dataset.indices)
    print(f"Epoch {epoch}/{EPOCHS} | ELBO loss: {total/n:.3f}  recon: {total_recon/n:.3f}  KL: {total_kl/n:.3f}")

model.eval()
with torch.no_grad():
    imgs, _ = next(iter(train_loader))
    imgs = imgs.to(device)
    recon, _, _ = model(imgs.view(imgs.size(0), -1))
    recon = recon.view(-1, 1, 28, 28)
    grid = vutils.make_grid(torch.cat([imgs[:8], recon[:8]], dim=0), nrow=8)
    vutils.save_image(grid, os.path.join(OUT_DIR, "vae_recon.png"))

    z = torch.randn(64, LATENT_DIM, device=device)
    samples = model.decode(z).view(-1, 1, 28, 28)
    vutils.save_image(vutils.make_grid(samples, nrow=8), os.path.join(OUT_DIR, "vae_samples.png"))

print("Saved:", os.path.join(OUT_DIR, "vae_recon.png"), "and", os.path.join(OUT_DIR, "vae_samples.png"))
