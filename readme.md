# Grokking on Modular Addition 

This is a minimal, single-file reproduction of the **grokking / delayed generalization** phenomenon described in:

- Alethea Power, Yuri Burda, Harri Edwards, Igor Babuschkin, Vedant Misra. *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets*. arXiv:2201.02177 (2022).

The core observation in this setting is that a model can reach **near-perfect training accuracy** quickly while **validation accuracy remains near chance for a long time**, and then—after many additional optimization steps—validation accuracy transitions sharply to **near-perfect generalization**.

---

## What this implementation does

**Task (modular addition)**  
For a prime (or any integer) modulus `p`, define the dataset of all ordered pairs:

- inputs: `(a, b)` with `a, b ∈ {0, …, p-1}`
- label: `(a + b) mod p`

**Tokenization**  
Each example is represented as a short token sequence:

```
[a, "+", b, "="]
```

with:
- numbers: token IDs `0 … p-1`
- `"+"`: token ID `p`
- `"="`: token ID `p+1`

The model is trained to predict the label (a class in `0 … p-1`) from the sequence.

**Train/validation split**  
A random split is taken over the full set of `p²` pairs:
- `train_frac` controls what fraction goes into training
- the remainder is used for validation

**Model**  
A small causal Transformer (token embeddings + positional embeddings + `nn.TransformerEncoder` with a causal mask) reads the 4-token sequence and predicts the result using the hidden state at the final position.

**Optimization**  
Training uses AdamW with explicit weight decay. 

---

## Requirements

- PyTorch
- `wandb` 
- `optuna` (optional, only if you use `--optuna`)


---

## Quickstart


```bash
python train_mod_add.py
```

A more explicit run (defaults shown):

```bash
python train_mod_add.py   --p 97   --train_frac 0.5   --steps 100000   --eval_every 250   --batch_size 512   --lr 1e-3   --weight_decay 1.0
```

Force CPU:

```bash
python train_mod_add.py --device cpu
```

---

## What to look for (the “grokking” signature)

The script prints metrics periodically:

- `train/acc` should typically rise toward **1.0** relatively early.
- `val/acc` will often hover near **chance** (`1/p`) for a long time.
- Later in training, `val/acc` can jump rapidly from ~chance to **near 1.0**.

For `p=97`, chance accuracy is:

- `chance_acc = 1/97 ≈ 0.0103`

The printout includes `chance_acc` to sanity-check that you are seeing the right regime.

If you log to W&B, the qualitative pattern to watch is:

- `train/acc`: rises early and stays high
- `val/acc`: flat near chance, then a sharp transition upward
- `train/loss` vs `val/loss`: can show long periods of decoupled behavior

---

## Reproducing grokking more reliably

Grokking is sensitive to hyperparameters. If you do **not** see delayed generalization, try these levers:

1. **Increase training duration**
   - grokking often requires a very large number of optimization steps / effective epochs
   - start by increasing `--steps` (e.g. `300000` or more)

2. **Adjust weight decay (often critical)**
   - try `--weight_decay 0.5`, `1.0`, `2.0`

3. **Reduce training set fraction**
   - smaller `--train_frac` generally makes generalization harder and can make the delay more pronounced  
   - try `--train_frac 0.3` or `0.2`

4. **Tune learning rate and warmup**
   - if unstable: lower `--lr` and/or increase `--lr_warmup_steps`
   - if nothing happens: sometimes slightly higher `--lr` helps escape plateaus

Example run that often makes the delayed transition easier to see:

```bash
python train_mod_add.py   --p 97   --train_frac 0.3   --steps 300000   --eval_every 250   --batch_size 512   --lr 5e-4   --lr_warmup_steps 100   --weight_decay 1.0   --dropout 0.0
```

### Interpreting “steps” vs “epochs”

This script trains by iterating batches indefinitely (it restarts the DataLoader iterator when an epoch ends), so the meaningful quantity is often **effective epochs**:

```
effective_epochs ≈ steps * batch_size / |train_set|
```

For example, with `p=97`, `train_frac=0.5`, `|train| ≈ 4704`, `batch_size=512`:

- `100000` steps corresponds to roughly `100000 * 512 / 4704 ≈ 10884` effective epochs

---

## Command-line options

### Core experiment

- `--p`: modulus (default: `97`)
- `--train_frac`: fraction of `p²` pairs used for training (default: `0.5`)
- `--seed`: controls the random train/val split and initialization (default: `1337`)
- `--steps`: number of optimizer steps (default: `100000`)
- `--eval_every`: evaluation frequency in steps (default: `250`)
- `--batch_size`: training batch size cap (default: `512`)

### Model

- `--d_model`: embedding/hidden width (default: `128`)
- `--nhead`: attention heads (default: `4`)  
  (`d_model` must be divisible by `nhead`)
- `--d_ff`: feedforward width (default: `512`)
- `--num_layers`: Transformer encoder layers (default: `2`)
- `--dropout`: dropout probability (default: `0.1`)

### Optimization

- `--lr`: AdamW learning rate (default: `1e-3`)
- `--lr_warmup_steps`: linear LR warmup steps (default: `10`)
- `--cooldown_frac`: end-of-training cosine LR cooldown fraction (default: `0.0`, i.e. constant LR)
- `--weight_decay`: AdamW weight decay (default: `1.0`)

### Performance / runtime

- `--device`: `cuda`, `cpu`, or `mps` (default auto-detect)
- `--no_compile`: disables `torch.compile` (possibly required for your setup)
- `--high_precision`: disables autocast and TF32; runs in float32
- `--profile_steps`: prints a short profiler report for a few steps

### Weights & Biases

- `--no_wandb`: disable logging
- `--wandb_project`, `--wandb_group`, `--wandb_name`: W&B metadata

---

## Optuna mode (optional hyperparameter search)

The script includes an Optuna driver intended to find configurations that reach a target validation accuracy in as few steps as possible.

Run:

```bash
python train_mod_add.py   --optuna   --optuna_n_trials 100   --optuna_steps 2000   --optuna_eval_every 25   --optuna_target_val_acc 1.0
```

Notes:
- `optuna` is imported only when `--optuna` is set.
- The objective uses the number of steps required to hit `--optuna_target_val_acc`.
- Trials can be pruned early using heuristic pruning rules; disable the median-based rule with `--optuna_prune_median_off` if it is not helping.

---

## Implementation notes

### Model definition

`SimpleCausalTransformer`:
- learned token embedding + learned positional embedding
- `nn.TransformerEncoder` with:
  - GELU activation
  - pre-norm (`norm_first=True`)
  - a fixed upper-triangular causal mask
- output head:
  - takes the representation at the final position
  - maps to `p` logits via a linear layer

### Evaluation

`evaluate()` computes:
- average cross-entropy loss
- accuracy
over an entire DataLoader (train set or validation set).

---

## Reference (BibTeX)

```bibtex
@article{power2022grokking,
  title   = {Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets},
  author  = {Power, Alethea and Burda, Yuri and Edwards, Harri and Babuschkin, Igor and Misra, Vedant},
  journal = {arXiv preprint arXiv:2201.02177},
  year    = {2022}
}
```
