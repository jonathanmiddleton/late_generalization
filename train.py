import argparse
import time
from dataclasses import dataclass
import wandb
import math

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

@dataclass(frozen=True)
class ModOpSpec:
    p: int = 97
    op_token: int = 97
    eq_token: int = 98

    @property
    def vocab_size(self) -> int:
        return self.p + 2

    @property
    def seq_len(self) -> int:
        return 4

    def label(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a + b) % self.p


class ModAddTokensDataset(Dataset):
    def __init__(self, tokens: torch.Tensor, labels: torch.Tensor):
        self.tokens = tokens.long()
        self.labels = labels.long()

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, idx: int):
        return self.tokens[idx], self.labels[idx]


def make_mod_add_split(p: int, train_frac: float, seed: int) -> tuple[Dataset, Dataset, ModOpSpec]:
    spec = ModOpSpec(p=p, op_token=p, eq_token=p + 1)

    a = torch.arange(p, dtype=torch.long)
    b = torch.arange(p, dtype=torch.long)
    aa, bb = torch.meshgrid(a, b, indexing="ij")
    aa = aa.reshape(-1)
    bb = bb.reshape(-1)

    labels = spec.label(aa, bb)

    op_tok = torch.full_like(aa, fill_value=spec.op_token)
    eq_tok = torch.full_like(aa, fill_value=spec.eq_token)
    tokens = torch.stack([aa, op_tok, bb, eq_tok], dim=1)

    n = tokens.shape[0]
    n_train = int(train_frac * n)
    n_train = max(1, min(n - 1, n_train))

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    train_ds = ModAddTokensDataset(tokens[train_idx], labels[train_idx])
    val_ds = ModAddTokensDataset(tokens[val_idx], labels[val_idx])
    return train_ds, val_ds, spec


class CausalTransformer1L(nn.Module):
    def __init__(self, vocab_size: int, p_out: int, seq_len: int, d_model: int, nhead: int, d_ff: int):
        super().__init__()
        self.seq_len = seq_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1, norm=nn.LayerNorm(d_model))
        self.out = nn.Linear(d_model, p_out)

        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        bsz, seqlen = tokens.shape
        if seqlen != self.seq_len:
            raise ValueError(f"expected seq_len={self.seq_len}, got {seqlen}")

        pos = torch.arange(seqlen, device=tokens.device)
        x = self.tok_emb(tokens) + self.pos_emb(pos)[None, :, :]
        x = self.encoder(x, mask=self.causal_mask)
        h = x[:, -1, :]
        return self.out(h)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = torch.tensor(0.0, device=device)
    total_correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = loss_fn(logits, y)

        bs = y.shape[0]
        total_loss += loss.detach() * bs
        # noinspection PyUnresolvedReferences
        total_correct += (logits.argmax(dim=-1) == y).sum()
        total += bs

    return total_loss.item() / max(1, total), total_correct / max(1, total)

def cosine_decay_schedule(s: float, cooldown_frac: float = 1.0) -> float:
    """Constant 1.0 until cooldown, then cosine decay to 0.0 over cooldown span."""
    if cooldown_frac == 0.0:
        return 1.0
    x = 0.0 if s < 0.0 else (1.0 if s > 1.0 else s)
    c = 0.0 if cooldown_frac < 0.0 else (1.0 if cooldown_frac > 1.0 else cooldown_frac)
    if c <= 0.0:
        return 1.0
    if x < 1.0 - c:
        return 1.0
    t = (x - (1.0 - c)) / max(c, 1e-8)  # normalized to [0,1]
    return 0.5 * (1.0 + math.cos(math.pi * t))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=97, help="(prime) modulus of the operation")
    ap.add_argument("--train_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--batch_size", type=int, default=1024)

    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--d_ff", type=int, default=512)

    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--cooldown_frac", type=float, default=0.0)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "mps"
        if torch.backends.mps.is_available() else "cpu")

    args = ap.parse_args()

    # ------------ Determinism and Precision ------------- #
    # disallow TF32 for matmul and cuDNN convs
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # favor highest-precision FP32 matmul kernels
    torch.set_float32_matmul_precision("highest")
    torch.set_default_dtype(torch.float32)
    torch.use_deterministic_algorithms(True)
    # despite above, below may be required to increase likelihood of deterministic results from cuDNN
    torch.backends.cudnn.benchmark = False

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    # ---------------------------------------------------- #

    _wandb = wandb.init(project="late_generalization", config=vars(args))

    device = torch.device(args.device)

    train_ds, val_ds, spec = make_mod_add_split(p=args.p, train_frac=args.train_frac, seed=args.seed)

    #noinspection PyTypeChecker
    train_bs = min(args.batch_size, len(train_ds))
    train_loader = DataLoader(train_ds, batch_size=train_bs, shuffle=True, num_workers=0, pin_memory=True)
    train_eval_loader = DataLoader(train_ds, batch_size=2048, shuffle=False, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=2048, shuffle=False, num_workers=0, pin_memory=True)

    model = CausalTransformer1L(
        vocab_size=spec.vocab_size,
        p_out=spec.p,
        seq_len=spec.seq_len,
        d_model=args.d_model,
        nhead=args.nhead,
        d_ff=args.d_ff,
    ).to(device)
    model = torch.compile(model)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    loss_fn = nn.CrossEntropyLoss()

    #noinspection PyTypeChecker
    print(f"p={spec.p}  train={len(train_ds)}  val={len(val_ds)}  chance_acc={1.0/spec.p:.6f}")
    print(
        f"model: 1-layer causal Transformer  d_model={args.d_model} nhead={args.nhead} d_ff={args.d_ff}  "
        f"opt=AdamW lr={args.lr} wd={args.weight_decay}"
    )

    t0 = time.time()
    train_iter = iter(train_loader)
    wandb_log_stepsize = max(1, args.eval_every // 100)
    tokens_per_step = args.batch_size * args.seq_len
    cum_tokens = 0

    for step in range(1, args.steps + 1):
        cum_tokens += tokens_per_step
        model.train()

        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = loss_fn(logits, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        lr = cosine_decay_schedule(step / args.steps, args.cooldown_frac) * args.lr
        for pg in opt.param_groups: pg["lr"] = lr
        opt.step()

        if step == 1 or step % args.eval_every == 0:
            train_loss, train_acc = evaluate(model, train_eval_loader, device)
            val_loss, val_acc = evaluate(model, val_loader, device)
            dt = time.time() - t0
            print(
                f"step={step:>7d}  "
                f"train/loss={train_loss:.6f} train/acc={train_acc:.6f}  "
                f"val/loss={val_loss:.6f} val/acc={val_acc:.6f}  "
                f"elapsed_s={dt:.1f}"
            )
            _wandb.log({"train/loss": train_loss, "train/acc": train_acc, "val/loss": val_loss, "val/acc": val_acc, "lr":lr, "cum_tokens":cum_tokens})
        elif step % wandb_log_stepsize == 0:
            train_loss = loss.detach().item()
            _wandb.log({"step": step, "train/loss": train_loss, "cum_tokens":cum_tokens})

    train_loss, train_acc = evaluate(model, train_eval_loader, device)
    val_loss, val_acc = evaluate(model, val_loader, device)
    print(f"final  train/loss={train_loss:.6f} train/acc={train_acc:.6f}  val/loss={val_loss:.6f} val/acc={val_acc:.6f}")


if __name__ == "__main__":
    main()
