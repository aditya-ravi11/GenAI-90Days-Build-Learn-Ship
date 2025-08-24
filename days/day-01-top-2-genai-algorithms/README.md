# Day 01 — Top 2 GenAI Algorithms (VAE & GAN)

> Folder: `days/day-01-top-2-genai-algorithms/`

## What I set out to do
- Understand what GenAI is, and how it is used in modern engineering.
- Understand 2 GenAI algorithms- GANs and VAEs
- Code basic working implementations of both algorithms.

## What I built today
- **VAE (MLP, MNIST)** — trains for 1 epoch and saves reconstructions + prior samples.
- **GAN (MLP, MNIST)** — trains for 1 epoch and saves generated digit grids.
- **Executable notes** in `notes.ipynb` (Typed out all theory i learnt and understood today.)

## Results

**VAE reconstructions (top = input, bottom = recon):**  
![VAE Recon](samples/vae_recon.png)

**VAE samples (z ~ N(0, I)):**  
![VAE Samples](samples/vae_samples.png)

**GAN fake samples (epoch 1):**  
![GAN Samples](samples/gan_fake_epoch_1.png)

**Metrics:**
ELBO loss: 165.484  recon: 150.218  KL: 15.267

## How to run
```bash
# from this folder
cd experiments

# VAE
python vae_basic.py

# GAN
python gan_basic.py

