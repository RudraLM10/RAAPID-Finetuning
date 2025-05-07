import torch
import torch.nn as nn
import torch.optim as optim
import math


class GRITAdapter(nn.Module):
    def __init__(self, base_layer: nn.Linear, rank: int = 4, epsilon: float = 1e-3, lr: float = 1e-2):
        super().__init__()
        # frozen pre-trained layer
        self.base = base_layer  
        d_out, d_in = base_layer.weight.size()
        self.rank = rank
        # LoRA adapters 
        self.A = nn.Parameter(torch.zeros(d_out, rank))
        self.B = nn.Parameter(torch.zeros(rank, d_in))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)
        # GRIT hyperparams
        self.epsilon = epsilon  
        self.lr = lr

    def forward(self, x):
        base_out = nn.functional.linear(x, self.base.weight)
        lora_out = x @ self.B.t() @ self.A.t()
        return base_out + lora_out

    def compute_kfac(self, activations, backprops):
        batch = activations.size(0)
        A_cov = activations.t() @ activations / batch + 1e-5 * torch.eye(activations.size(1))
        B_cov = backprops.t() @ backprops / batch + 1e-5 * torch.eye(backprops.size(1))
        return A_cov, B_cov

    def grit_step(self, x, y, loss_fn):
        out = self.forward(x)
        loss = loss_fn(out, y)
        loss.backward(retain_graph=True)

        G_A = self.A.grad.detach().clone()
        G_B = self.B.grad.detach().clone()

        backprop = torch.autograd.grad(loss, out, retain_graph=True)[0]
        A_kfac, B_kfac = self.compute_kfac(x.detach(), backprop.detach())

        fisher_B = torch.trace(G_B @ A_kfac @ G_B.t())
        fisher_A = torch.trace(G_A.t() @ B_kfac @ G_A)
        fisher_norm_sq = fisher_A + fisher_B
        
        scale = 1.0
        if fisher_norm_sq > self.epsilon:
            scale = torch.sqrt(self.epsilon / (fisher_norm_sq + 1e-12))
        with torch.no_grad():
            self.A -= self.lr * scale * G_A
            self.B -= self.lr * scale * G_B
            self.A.grad.zero_()
            self.B.grad.zero_()
        return loss.item()


if __name__ == '__main__':
    d_in, d_out = 16, 16
    W_true = torch.randn(d_out, d_in)
    base_layer = nn.Linear(d_in, d_out, bias=False)
    base_layer.weight.data = W_true.clone()
    base_layer.weight.requires_grad = False

    adapter = GRITAdapter(base_layer, rank=2, epsilon=1e-2, lr=1e-1)
    loss_fn = nn.MSELoss()
    # Test data
    X = torch.randn(64, d_in)
    Y = X @ (W_true * 1.1).t()

    for epoch in range(20):
        loss = adapter.grit_step(X, Y, loss_fn)
        print(f"Epoch {epoch+1:02d}, Loss={loss:.4f}")
