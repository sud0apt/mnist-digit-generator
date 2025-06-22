import torch

def generate_digit_images(model, digit, n_samples=5):
    model.eval()
    with torch.no_grad():
        device = next(model.parameters()).device
        latent_dim = model.latent_dim 
        y = torch.eye(10)[[digit] * n_samples].to(device)
        z = torch.randn(n_samples, latent_dim).to(device)
        generated = model.decode(z, y).cpu()
        return generated.view(-1, 28, 28)
