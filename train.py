import argparse
import time
import math
import gc
import statistics
from collections import defaultdict
from dataclasses import dataclass
from contextlib import nullcontext
from typing import Callable, Optional

import wandb as _wandb

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.profiler import profile, ProfilerActivity

@dataclass(frozen=True)
class EvalEvent:
    step: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float


EvalCallback = Callable[[EvalEvent], None]

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
    def __init__(self, tokens: torch.Tensor, labels: torch.Tensor, device: torch.device):
        if tokens.shape[0] != labels.shape[0]:
            raise ValueError(f"tokens and labels must have same length: {tokens.shape[0]} vs {labels.shape[0]}")
        self.tokens = tokens.long().to(device, non_blocking=True)
        self.labels = labels.long().to(device, non_blocking=True)

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, idx: int):
        return self.tokens[idx], self.labels[idx]


def make_mod_add_split(p: int, train_frac: float, seed: int, device: torch.device) -> tuple[Dataset, Dataset, ModOpSpec]:
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

    train_ds = ModAddTokensDataset(tokens[train_idx], labels[train_idx], device=device)
    val_ds = ModAddTokensDataset(tokens[val_idx], labels[val_idx], device=device)
    return train_ds, val_ds, spec


class SimpleCausalTransformer(nn.Module):
    def __init__(self, vocab_size: int, p_out: int, seq_len: int, d_model: int, nhead: int, d_ff: int, num_layers: int, dropout: float):
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
            dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, norm=nn.LayerNorm(d_model))
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
    total_correct = torch.zeros((), device=device, dtype=torch.long)
    total = torch.zeros((), device=device, dtype=torch.long)

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

    avg_loss = total_loss / total.clamp_min(1)
    acc = total_correct.to(torch.float32) / total.clamp_min(1).to(torch.float32)

    return avg_loss.item(), acc.item()


def cosine_decay_schedule(s: float, cooldown_frac: float = 1.0) -> float:
    if cooldown_frac == 0.0:
        return 1.0
    x = 0.0 if s < 0.0 else (1.0 if s > 1.0 else s)
    c = 0.0 if cooldown_frac < 0.0 else (1.0 if cooldown_frac > 1.0 else cooldown_frac)
    if c <= 0.0:
        return 1.0
    if x < 1.0 - c:
        return 1.0
    t = (x - (1.0 - c)) / max(c, 1e-8)
    return 0.5 * (1.0 + math.cos(math.pi * t))


def _clone_namespace(args: argparse.Namespace, **updates) -> argparse.Namespace:
    d = vars(args).copy()
    d.update(updates)
    return argparse.Namespace(**d)


def _maybe_reset_compile_cache() -> None:
    dyn = getattr(torch, "_dynamo", None)
    if dyn is not None and hasattr(dyn, "reset"):
        dyn.reset()


_PRUNE_HISTORY = defaultdict(list)


def _median_prune(step: int, val_loss: float, args: argparse.Namespace) -> bool:
    if args.optuna_prune_median_off:
        return False
    if step < args.optuna_prune_warmup_steps:
        return False
    past = _PRUNE_HISTORY.get(step, [])
    if len(past) < args.optuna_prune_min_trials:
        return False
    med = statistics.median(past)
    return val_loss > med * (1.0 + args.optuna_prune_margin)


def train_and_eval(args: argparse.Namespace, trial=None, *, on_eval: Optional[list[EvalCallback]] = None) -> dict[str, float]:
    if args.d_model % args.nhead != 0:
        raise ValueError(f"d_model must be divisible by nhead (got d_model={args.d_model}, nhead={args.nhead})")

    if args.high_precision:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        torch.set_default_dtype(torch.float32)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)

    run = None
    if not args.no_wandb:
        run = _wandb.init(
            project=args.wandb_project,
            config=vars(args),
            group=args.wandb_group if args.wandb_group else None,
            name=args.wandb_name if args.wandb_name else None,
            reinit=True,
        )

    callbacks: list[EvalCallback] = [] if on_eval is None else list(on_eval)

    try:
        train_ds, val_ds, spec = make_mod_add_split(p=args.p, train_frac=args.train_frac, seed=args.seed, device=device)

        if args.p >= 23:
            #noinspection PyTypeChecker
            train_bs = int(min(args.batch_size, len(train_ds)))
        else:
            train_bs = int(min(max(1, int(args.p * args.p * args.train_frac)), args.batch_size))

        eval_bs = 2048
        train_loader = DataLoader(train_ds, batch_size=train_bs, shuffle=True, num_workers=0)
        train_eval_loader = DataLoader(train_ds, batch_size=eval_bs, shuffle=False, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=eval_bs, shuffle=False, num_workers=0)

        model = SimpleCausalTransformer(
            vocab_size=spec.vocab_size,
            p_out=spec.p,
            seq_len=spec.seq_len,
            d_model=args.d_model,
            nhead=args.nhead,
            d_ff=args.d_ff,
            num_layers=args.num_layers,
            dropout=args.dropout,
        ).to(device)

        if not args.no_compile and hasattr(torch, "compile"):
            model = torch.compile(model)

        embed_ids = {id(p) for p in model.tok_emb.parameters()} | {id(p) for p in model.pos_emb.parameters()}
        non_embedding_params = [p for p in model.parameters() if p.requires_grad and id(p) not in embed_ids]
        print(f"non-embedding trainable params: {sum(p.numel() for p in non_embedding_params)}")

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98))
        loss_fn = nn.CrossEntropyLoss()

        chance_acc = 1.0 / spec.p
        chance_loss = -torch.log(torch.tensor(chance_acc, device="cpu")).item()

        #noinspection PyTypeChecker
        print(f"p={spec.p}  train={len(train_ds)}  val={len(val_ds)}  chance_acc={chance_acc:.6f}")
        print(
            f"model: num_layers={args.num_layers}  d_model={args.d_model}  nhead={args.nhead}  d_ff={args.d_ff}  "
            f"opt=AdamW  lr={args.lr}  wd={args.weight_decay}"
        )

        t0 = time.time()
        train_iter = iter(train_loader)
        warmup_steps = min(max(int(args.lr_warmup_steps), 0), int(args.steps))

        total_labels = 0
        log_train_every = int(args.wandb_log_every)

        ctx = nullcontext() if device.type == "cpu" or args.high_precision else torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16)

        x, y = next(train_iter)

        profiler_skip_steps = 2

        best_val_loss = float("inf")
        best_val_acc = 0.0
        steps_to_target = None
        steps_ran = int(args.steps)

        for step in range(1, int(args.steps) + 1):
            model.train()

            B, _ = x.shape
            total_labels += B

            do_profile = (
                args.profile_steps > 0
                and (step <= args.profile_steps + profiler_skip_steps)
                and step > profiler_skip_steps
            )
            prof_ctx = (
                profile(
                    activities=[
                        ProfilerActivity.CPU,
                        ProfilerActivity.CUDA if device.type == "cuda" else ProfilerActivity.CPU,
                    ],
                    record_shapes=True,
                )
                if do_profile
                else nullcontext()
            )

            with ctx:
                with prof_ctx as prof:
                    logits = model(x)

            try:
                _x, _y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                _x, _y = next(train_iter)

            if do_profile:
                print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=40))

            loss = loss_fn(logits, y)
            train_loss_step = float(loss.detach().item())

            opt.zero_grad(set_to_none=True)
            loss.backward()

            if warmup_steps > 0 and step <= warmup_steps:
                lr_scale = step / warmup_steps
            else:
                lr_scale = cosine_decay_schedule(step / args.steps, args.cooldown_frac)

            lr = args.lr * lr_scale
            for pg in opt.param_groups:
                pg["lr"] = lr
            opt.step()

            if step == 1 or step % args.eval_every == 0:
                train_loss, train_acc = evaluate(model, train_eval_loader, device)
                val_loss, val_acc = evaluate(model, val_loader, device)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_acc = val_acc

                if steps_to_target is None and val_loss <= args.optuna_target_val_loss:
                    steps_to_target = step

                dt = time.time() - t0
                print(
                    f"step={step:>7d}  "
                    f"train/loss={train_loss:.6f} train/acc={train_acc:.6f}  "
                    f"val/loss={val_loss:.6f} val/acc={val_acc:.6f}  "
                    f"elapsed_s={dt:.1f}  "
                    f"total_labels:{total_labels:,}  "
                    f"lr={lr:.6f}  "
                    f"lr_scale={lr_scale:.6f}"
                )

                if not args.no_wandb:
                    _wandb.log(
                        {
                            "step": step,
                            "train/loss": train_loss,
                            "labels": total_labels,
                            "train/acc": train_acc,
                            "val/loss": val_loss,
                            "val/acc": val_acc,
                            "lr": lr,
                            "lr_scale": lr_scale,
                            "chance_loss": chance_loss,
                            "best/val_loss": best_val_loss,
                            "best/val_acc": best_val_acc,
                            "target/val_loss": args.optuna_target_val_loss,
                            "steps_to_target": -1 if steps_to_target is None else int(steps_to_target),
                        }
                    )

                if callbacks:
                    event = EvalEvent(
                        step=int(step),
                        train_loss=float(train_loss),
                        train_acc=float(train_acc),
                        val_loss=float(val_loss),
                        val_acc=float(val_acc),
                    )
                    for cb in callbacks:
                        cb(event)

                if trial is not None:
                    do_prune = _median_prune(step, float(val_loss), args)
                    _PRUNE_HISTORY[step].append(float(val_loss))

                    if do_prune:
                        if args.optuna_prune_action == "prune":
                            import optuna
                            trial.set_user_attr("pruned_at_step", int(step))
                            trial.set_user_attr("pruned_val_loss", float(val_loss))
                            raise optuna.TrialPruned()
                        else:
                            steps_ran = step
                            break
            elif step % log_train_every == 0 and not args.no_wandb:
                _wandb.log(
                    {
                        "step": step,
                        "train/loss": train_loss_step,
                        "labels": total_labels,
                        "lr": lr,
                        "lr_scale": lr_scale,
                        "chance_loss": chance_loss,
                    }
                )

            x, y = _x, _y

        train_loss, train_acc = evaluate(model, train_eval_loader, device)
        val_loss, val_acc = evaluate(model, val_loader, device)

        if trial is not None:
            trial.set_user_attr("steps_ran", int(steps_ran))
            trial.set_user_attr("final_val_loss", float(val_loss))
            trial.set_user_attr("final_val_acc", float(val_acc))
            trial.set_user_attr("best_val_loss", float(best_val_loss))
            trial.set_user_attr("best_val_acc", float(best_val_acc))
            trial.set_user_attr("steps_to_target", -1 if steps_to_target is None else int(steps_to_target))

        print(f"final  train/loss={train_loss:.6f} train/acc={train_acc:.6f}  val/loss={val_loss:.6f} val/acc={val_acc:.6f}")

        return {
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "best_val_loss": float(best_val_loss),
            "best_val_acc": float(best_val_acc),
            "steps_ran": float(steps_ran),
            "steps_to_target": float(-1 if steps_to_target is None else steps_to_target),
        }
    finally:
        if run is not None:
            _wandb.finish()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        _maybe_reset_compile_cache()


def run_optuna(args: argparse.Namespace) -> None:
    try:
        import optuna
    except Exception as e:
        raise RuntimeError("Optuna is not installed. pip install optuna") from e

    storage = args.optuna_storage if args.optuna_storage else None

    if args.optuna_sampler == "nsga2":
        sampler = optuna.samplers.NSGAIISampler(seed=args.optuna_seed)
    elif args.optuna_sampler == "tpe":
        sampler = optuna.samplers.TPESampler(seed=args.optuna_seed)
    else:
        sampler = optuna.samplers.RandomSampler(seed=args.optuna_seed)

    study = optuna.create_study(
        study_name=args.optuna_study_name,
        storage=storage,
        load_if_exists=bool(storage),
        direction="minimize",
        sampler=sampler,
    )

    best_steps_to_100: int | None = None

    def objective(trial: optuna.Trial):
        nonlocal best_steps_to_100

        nhead = trial.suggest_categorical("nhead", [4, 8])
        head_dim = trial.suggest_categorical("head_dim", [32, 64, 128])
        d_model = int(nhead * head_dim)
        d_ff_mult = trial.suggest_categorical("d_ff_mult", [2, 4, 8])
        d_ff = int(d_model * d_ff_mult)

        num_layers = 2
        dropout = trial.suggest_float("dropout", 0.0, 0.3)
        lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 0.5, 2.0)
        lr_warmup_steps = 100
        batch_size = trial.suggest_int("batch_size", 64, 1024)

        # Prune if we are "too late" compared to the best known solution (20% slack by default).
        prune_slack = 0.20
        acc_target = 1.0
        acc_tol = 1e-8  # epsilon

        def optuna_eval_hook(ev: EvalEvent) -> None:
            nonlocal best_steps_to_100

            trial.report(ev.val_acc, step=ev.step)

            hit_target = ev.val_acc >= (acc_target - acc_tol)
            if hit_target:
                trial.set_user_attr("steps_to_100_val_acc", int(ev.step))
                if best_steps_to_100 is None or ev.step < best_steps_to_100:
                    best_steps_to_100 = int(ev.step)
                return

            if best_steps_to_100 is not None:
                latest_allowed = int(math.floor(best_steps_to_100 * (1.0 + prune_slack)))
                if ev.step >= latest_allowed:
                    raise optuna.TrialPruned(
                        f"too slow: step={ev.step} best={best_steps_to_100} slack={prune_slack}"
                    )

        trial_args = _clone_namespace(
            args,
            eval_every=args.optuna_eval_every,
            nhead=nhead,
            d_model=d_model,
            d_ff=d_ff,
            num_layers=num_layers,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            lr_warmup_steps=lr_warmup_steps,
            batch_size=batch_size,
            wandb_group=args.wandb_group if args.wandb_group else args.optuna_study_name,
            wandb_name=args.wandb_name if args.wandb_name else f"trial_{trial.number}",
        )

        _ = train_and_eval(trial_args, trial=trial, on_eval=[optuna_eval_hook])

        steps_to_100 = trial.user_attrs.get("steps_to_100_val_acc", None)
        if steps_to_100 is None:
            steps_to_100 = int(args.steps + args.eval_every)

        return int(steps_to_100)


    timeout = None if args.optuna_timeout_s <= 0 else float(args.optuna_timeout_s)
    study.optimize(objective, n_trials=args.optuna_n_trials, timeout=timeout, gc_after_trial=True)

    best = study.best_trial
    print(f"best_trial={best.number}  steps_to_100={int(best.value)}  params={best.params}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=97)
    ap.add_argument("--train_frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--wandb_log_every", type=int, default=250)
    ap.add_argument("--batch_size", type=int, default=512)

    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--d_ff", type=int, default=512)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr_warmup_steps", type=int, default=10)
    ap.add_argument("--cooldown_frac", type=float, default=0.0)
    ap.add_argument("--weight_decay", type=float, default=1.0)

    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--profile_steps", type=int, default=0)
    ap.add_argument("--high_precision", action="store_true")

    ap.add_argument("--no_compile", action="store_true")
    ap.add_argument("--no_wandb", action="store_true")
    ap.add_argument("--wandb_project", type=str, default="late_generalization")
    ap.add_argument("--wandb_group", type=str, default="")
    ap.add_argument("--wandb_name", type=str, default="")

    ap.add_argument("--optuna", action="store_true")
    ap.add_argument("--optuna_n_trials", type=int, default=50)
    ap.add_argument("--optuna_timeout_s", type=int, default=0)
    ap.add_argument("--optuna_eval_every", type=int, default=50)
    ap.add_argument("--optuna_storage", type=str, default="")
    ap.add_argument("--optuna_study_name", type=str, default="late_generalization_mo")
    ap.add_argument("--optuna_sampler", type=str, default="tpe", choices=["nsga2", "tpe", "random"])
    ap.add_argument("--optuna_seed", type=int, default=1337)

    ap.add_argument("--optuna_target_val_loss", type=float, default=1e-3)

    ap.add_argument("--optuna_prune_median_off", action="store_true", default=False)
    ap.add_argument("--optuna_prune_action", type=str, default="prune", choices=["prune", "stop"])
    ap.add_argument("--optuna_prune_warmup_steps", type=int, default=0)
    ap.add_argument("--optuna_prune_min_trials", type=int, default=10)
    ap.add_argument("--optuna_prune_margin", type=float, default=0.02, help="Scalar applied to median for prune margin.")

    args = ap.parse_args()

    if args.optuna:
        run_optuna(args)
    else:
        train_and_eval(args)


if __name__ == "__main__":
    main()
