import os, math
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, utils as vutils

Z_DIM = 64
G_HIDDEN = 256
D_HIDDEN = 256
BATCH_SIZE = 128
EPOCHS = 1
LR = 2e-4
BETA1 = 0.5
DATA_DIR = "./data"
OUT_DIR = "samples"
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])
train_ds = datasets.MNIST(root=DATA_DIR, train=True, transform=transform, download=True)
train_ds = Subset(train_ds, range(1000))  

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)

class Generator(nn.Module):
    def __init__(self, z=Z_DIM, out_dim=784, h=G_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z, h), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(h, h*2), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(h*2, out_dim), nn.Tanh(),  
        )
    def forward(self, z):
        x = self.net(z)
        return x

class Discriminator(nn.Module):
    def __init__(self, in_dim=784, h=D_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h*2), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(h*2, h), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(h, 1), nn.Sigmoid(),  
        )
    def forward(self, x):
        return self.net(x).view(-1)

G = Generator().to(device)
D = Discriminator().to(device)

g_opt = optim.Adam(G.parameters(), lr=LR, betas=(BETA1, 0.999))
d_opt = optim.Adam(D.parameters(), lr=LR, betas=(BETA1, 0.999))
criterion = nn.BCEWithLogitsLoss()

fixed_noise = torch.randn(64, Z_DIM, device=device)

for epoch in range(EPOCHS):
    for i, (real, _) in enumerate(train_loader, 1):
        real = real.view(real.size(0), -1).to(device)
        b = real.size(0)
        ones = torch.ones(b, device=device)
        zeros = torch.zeros(b, device=device)

        D.train(); G.train()
        d_opt.zero_grad()

        real_logits = D(real)
        d_real = criterion(real_logits, ones)

        z = torch.randn(b, Z_DIM, device=device)
        fake = G(z).detach()  
        fake_logits = D(fake)
        d_fake = criterion(fake_logits, zeros)

        d_loss = d_real + d_fake
        d_loss.backward()
        d_opt.step()

        g_opt.zero_grad()
        z = torch.randn(b, Z_DIM, device=device)
        fake = G(z)
        fake_logits = D(fake)
        g_loss = criterion(fake_logits, ones)
        g_loss.backward()
        g_opt.step()

        if i % 100 == 0:
            print(f"Epoch {epoch}/{EPOCHS}  Step {i}/{len(train_loader)}  "
                  f"D_loss: {d_loss.item():.3f}  G_loss: {g_loss.item():.3f}")

    with torch.no_grad():
        fake = G(fixed_noise).view(-1, 1, 28, 28)
        img = (fake + 1) / 2
        vutils.save_image(vutils.make_grid(img, nrow=8), os.path.join(OUT_DIR, f"gan_fake_epoch_{epoch}.png"))

print("Saved:", os.path.join(OUT_DIR, f"gan_fake_epoch_{EPOCHS}.png"))
