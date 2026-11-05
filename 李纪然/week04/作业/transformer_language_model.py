

import argparse
import glob
import json
import math
import os
import random
import tempfile
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ─────────────────────────────── 数据 ───────────────────────────────

def load_corpus(pattern):
    """pattern 可以是单个文件，也可以是 glob（如 '*.txt'）。"""
    paths = [pattern] if os.path.isfile(pattern) else sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"未找到语料：{pattern}")
    texts = []
    for path in paths:
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    return "".join(texts)


def build_vocab(text, max_vocab=None):
    """按字频取前 max_vocab 个字建字表；三个模型用同一字表，PPL 才可比。"""
    from collections import Counter

    counter = Counter(text)
    if max_vocab:
        chars = [c for c, _ in counter.most_common(max_vocab)]
    else:
        chars = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


class CharDataset(Dataset):
    """
    输入 x = 连续 seq_len 个字符，目标 y = 整体右移一位。
    stride=seq_len 时是不重叠切块（标准做法，一趟正好扫过全部语料）；
    stride=1 时每个位置都切一条样本，样本数被放大 seq_len 倍，训练极慢。
    """

    def __init__(self, text, char2idx, seq_len, stride):
        self.seq_len = seq_len
        self.stride = max(1, stride)
        ids = [char2idx[c] for c in text if c in char2idx]
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return max(0, (len(self.data) - self.seq_len - 1) // self.stride + 1)

    def __getitem__(self, idx):
        start = idx * self.stride
        x = self.data[start: start + self.seq_len]
        y = self.data[start + 1: start + self.seq_len + 1]
        return x, y


# ─────────────────────────────── 模型 ───────────────────────────────

class RNNLM(nn.Module):
    """RNN / LSTM 基线：Embedding → 循环层 → Dropout → 线性输出。"""

    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2,
                 dropout=0.3, cell="lstm"):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        rnn_cls = nn.LSTM if cell == "lstm" else nn.RNN
        self.rnn = rnn_cls(embed_dim, hidden_dim, num_layers=num_layers,
                           batch_first=True,
                           dropout=dropout if num_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, idx, targets=None):
        x = self.drop(self.embed(idx))
        out, _ = self.rnn(x)
        logits = self.fc(self.drop(out))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=40, temperature=1.0, top_k=None):
        self.eval()
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k:
                thresh = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                logits = logits.masked_fill(logits < thresh, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, num_samples=1)], dim=1)
        return idx


class CausalSelfAttention(nn.Module):

    def __init__(self, d_model, n_head, dropout, max_len, use_sdpa=True):
        super().__init__()
        assert d_model % n_head == 0, "d_model 必须能被 n_head 整除"
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.dropout = dropout
        self.use_sdpa = use_sdpa

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(max_len, max_len, dtype=torch.bool))
        self.register_buffer("mask", mask.view(1, 1, max_len, max_len), persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.use_sdpa:
            y = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            att = att.masked_fill(~self.mask[:, :, :T, :T], float("-inf"))
            att = self.attn_drop(F.softmax(att, dim=-1))
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class FFN(nn.Module):
    """Position-wise Feed-Forward：d → 4d → d，GELU。"""

    def __init__(self, d_model, mult=4, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, mult * d_model, bias=False)
        self.fc2 = nn.Linear(mult * d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):


    def __init__(self, d_model, n_head, mult, dropout, max_len, use_sdpa):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, dropout, max_len, use_sdpa)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, mult, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    """Decoder-only Transformer 语言模型。"""

    def __init__(self, vocab_size, d_model=256, num_layers=4, num_heads=8,
                 max_len=512, ff_mult=4, dropout=0.1, use_sdpa=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)   # 可学习位置编码
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            Block(d_model, num_heads, ff_mult, dropout, max_len, use_sdpa)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight     # 权重共享：省参数、小语料更稳

        self.apply(self._init_weights)
        # 残差分支的输出投影按层数缩小初始化，深层训练更稳（GPT-2 的做法）
        for name, p in self.named_parameters():
            if name.endswith(("proj.weight", "fc2.weight")):
                p.data.normal_(0.0, 0.02 / math.sqrt(2 * num_layers))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.max_len, f"序列长度 {T} 超过位置编码上限 {self.max_len}"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.token_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=40, temperature=1.0, top_k=None):
        self.eval()
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.max_len:])
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k:
                thresh = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                logits = logits.masked_fill(logits < thresh, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, num_samples=1)], dim=1)
        return idx


def build_model(args, vocab_size):
    if args.model in ("rnn", "lstm"):
        return RNNLM(vocab_size, embed_dim=args.embed_dim, hidden_dim=args.hidden_dim,
                     num_layers=args.num_layers, dropout=args.dropout, cell=args.model)
    return TransformerLM(vocab_size, d_model=args.d_model, num_layers=args.num_layers,
                         num_heads=args.num_heads, max_len=max(args.max_len, args.seq_len),
                         ff_mult=args.ff_mult, dropout=args.dropout,
                         use_sdpa=not args.no_sdpa)


# ─────────────────────────── 训练 / 评估 ───────────────────────────

def run_epoch(model, loader, device, optimizer=None, clip=1.0, lr_fn=None, lr_base=0.0):

    train = optimizer is not None
    model.train(train)
    total_loss, total_tokens, total_correct = 0.0, 0, 0
    step = 0

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        if train:
            if lr_fn is not None:
                lr = lr_fn(lr_base)
                for group in optimizer.param_groups:
                    group["lr"] = lr
            logits, loss = model(x, targets=y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            step += 1
        else:
            with torch.no_grad():
                logits, loss = model(x, targets=y)

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()
        total_correct += (logits.argmax(-1) == y).sum().item()

    avg_loss = total_loss / max(1, total_tokens)
    return {
        "loss": avg_loss,
        "ppl": math.exp(min(avg_loss, 50.0)),
        "acc": total_correct / max(1, total_tokens),
        "tokens": total_tokens,
    }


def make_lr_fn(total_steps, warmup_steps, min_ratio=0.1):
    """线性 warmup + 余弦退火，返回 f(base_lr) -> lr。"""
    state = {"step": 0}

    def lr_fn(base_lr):
        s = state["step"]
        state["step"] += 1
        if s < warmup_steps:
            return base_lr * (s + 1) / max(1, warmup_steps)
        progress = (s - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return base_lr * (min_ratio + (1.0 - min_ratio) * cosine)

    return lr_fn


def build_optimizer(model, lr, weight_decay):
    """权重矩阵做 weight decay，LayerNorm / 偏置不衰减。"""
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)


# ─────────────────────────── 画图 ───────────────────────────

def plot_history(history, path, model_name):
    """三联图：loss / PPL(对数轴) / 学习率。"""
    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp())
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "Heiti TC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    ep = [e["epoch"] for e in history["epochs"]]
    tr_loss = [e["train_loss"] for e in history["epochs"]]
    va_loss = [e["val_loss"] for e in history["epochs"]]
    tr_ppl = [e["train_ppl"] for e in history["epochs"]]
    va_ppl = [e["val_ppl"] for e in history["epochs"]]
    lrs = [e["lr"] for e in history["epochs"]]
    best_i = min(range(len(va_ppl)), key=lambda i: va_ppl[i])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    fig.suptitle(
        f"{model_name}  |  语料 {history['corpus_chars']:,} 字  |  词表 {history['vocab_size']}  |  "
        f"参数量 {history['params']:,}  |  seq_len={history['args']['seq_len']} stride={history['args']['stride']}  |  "
        f"最佳验证 PPL {va_ppl[best_i]:.2f} (epoch {ep[best_i]})",
        fontsize=12, y=1.02)

    ax = axes[0]
    ax.plot(ep, tr_loss, "o-", ms=3.5, label="训练集 loss")
    ax.plot(ep, va_loss, "s-", ms=3.5, label="验证集 loss")
    ax.set_xlabel("epoch"); ax.set_ylabel("交叉熵 loss"); ax.set_title("Loss 曲线")
    ax.grid(alpha=0.3); ax.legend()

    ax = axes[1]
    ax.plot(ep, tr_ppl, "o-", ms=3.5, label="训练集 PPL")
    ax.plot(ep, va_ppl, "s-", ms=3.5, label="验证集 PPL")
    ax.set_yscale("log")
    ax.set_xlabel("epoch"); ax.set_ylabel("困惑度 PPL（对数轴）"); ax.set_title("PPL 曲线")
    ax.annotate(f"best {va_ppl[best_i]:.2f}", (ep[best_i], va_ppl[best_i]),
                textcoords="offset points", xytext=(8, 10), fontsize=9, color="tab:red",
                arrowprops=dict(arrowstyle="->", color="tab:red", lw=0.8))
    ax.grid(alpha=0.3, which="both"); ax.legend()

    ax = axes[2]
    ax.plot(ep, lrs, "-", color="tab:green", label="学习率（epoch 末）")
    ax.set_xlabel("epoch"); ax.set_ylabel("lr"); ax.set_title("学习率：warmup + 余弦退火")
    ax.grid(alpha=0.3); ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────── 主函数 ───────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="transformer", choices=["rnn", "lstm", "transformer"])
    parser.add_argument("--corpus", default="corpus_general.txt", help="语料文件或 glob")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--stride", type=int, default=64, help="切块间隔，64=不重叠（与基线同口径）")
    parser.add_argument("--eval_stride", type=int, default=0, help="验证集切块间隔，0=同 stride")

    parser.add_argument("--batch_size", type=int, default=128)
    # RNN / LSTM 超参
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    # Transformer 超参
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--ff_mult", type=int, default=4)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--no_sdpa", action="store_true", help="用显式 L×L 掩码（可导出注意力）")

    parser.add_argument("--num_layers", type=int, default=4, help="transformer=块数，rnn/lstm=层数")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--clip", type=float, default=1.0)

    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--max_vocab", type=int, default=4000, help="字表最大字数，0=不限")
    parser.add_argument("--max_chars", type=int, default=0, help="只用语料前 N 字（0=全部）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--out_dir", default=".")
    parser.add_argument("--sample_prompts", default="今天天气,我们明天,中国的首都,我正在,怎么才能,为什么")
    parser.add_argument("--no_plot", action="store_true")
    parser.add_argument("--plot_only", action="store_true",
                        help="不训练，只从已有的 {model}-stride{stride}.json 重画曲线")
    parser.add_argument("--tag", default="", help="文件名后缀，便于区分不同实验")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)


    if args.plot_only:
        stem = f"{args.model}-stride{args.stride}{args.tag}"
        history_path = os.path.join(args.out_dir, f"{stem}.json")
        with open(history_path, encoding="utf-8") as f:
            history = json.load(f)
        plot_history(history, os.path.join(args.out_dir, f"{stem}.png"), args.model.upper())
        print(f"已从 {history_path} 重画曲线 → {stem}.png")
        return

    # ── 设备 ──
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"device={device}  model={args.model}  sdpa={not args.no_sdpa and args.model == 'transformer'}")


    text = load_corpus(args.corpus)
    if args.max_chars:
        text = text[: args.max_chars]
    if not text:
        raise FileNotFoundError("语料为空，请检查 --corpus。")
    print(f"语料字符数: {len(text):,}")

    char2idx, idx2char = build_vocab(text, args.max_vocab or None)
    vocab_size = len(char2idx)
    print(f"字表大小: {vocab_size}")

    lines = text.splitlines()
    random.shuffle(lines)
    split = int(len(lines) * (1 - args.val_ratio))
    train_text, val_text = "\n".join(lines[:split]), "\n".join(lines[split:])

    eval_stride = args.eval_stride or args.stride
    train_ds = CharDataset(train_text, char2idx, args.seq_len, args.stride)
    val_ds = CharDataset(val_text, char2idx, args.seq_len, eval_stride)
    loader_kw = dict(num_workers=args.num_workers,
                     pin_memory=device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            drop_last=True, **loader_kw)
    print(f"训练集 {len(train_text):,} 字 / {len(train_ds):,} 个样本 / {len(train_loader):,} 步每轮")
    print(f"验证集 {len(val_text):,} 字 / {len(val_ds):,} 个样本")

    # ── 模型 ──
    model = build_model(args, vocab_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params:,}")

    optimizer = build_optimizer(model, args.lr, args.weight_decay)
    total_steps = max(1, len(train_loader) * args.epochs)
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    lr_fn = make_lr_fn(total_steps, warmup_steps, args.min_lr_ratio)

    stem = f"{args.model}-stride{args.stride}{args.tag}"
    ckpt_path = os.path.join(args.out_dir, f"best_{args.model}{args.tag}.pt")
    history_path = os.path.join(args.out_dir, f"{stem}.json")
    fig_path = os.path.join(args.out_dir, f"{stem}.png")

    history = {
        "args": vars(args), "model": args.model, "params": n_params,
        "vocab_size": vocab_size, "corpus_chars": len(text),
        "train_chars": len(train_text), "val_chars": len(val_text),
        "train_samples": len(train_ds), "steps_per_epoch": len(train_loader),
        "device": str(device), "epochs": [],
    }

    best_val_ppl = float("inf")
    print(f"\n{'epoch':>5}  {'train_loss':>10}  {'train_ppl':>10}  {'val_loss':>10}  "
          f"{'val_ppl':>10}  {'val_acc':>8}  {'lr':>9}  {'sec':>6}")
    print("-" * 86)

    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, train_loader, device, optimizer=optimizer,
                       clip=args.clip, lr_fn=lr_fn, lr_base=args.lr)
        va = run_epoch(model, val_loader, device)
        dt = time.time() - t0
        cur_lr = optimizer.param_groups[0]["lr"]

        marker = ""
        if va["ppl"] < best_val_ppl:
            best_val_ppl = va["ppl"]
            marker = "  *best"
            torch.save({
                "model_state": model.state_dict(),
                "char2idx": char2idx, "idx2char": idx2char,
                "args": vars(args), "vocab_size": vocab_size,
                "val_ppl": va["ppl"], "epoch": epoch, "model_type": args.model,
            }, ckpt_path)

        history["epochs"].append({
            "epoch": epoch,
            "train_loss": tr["loss"], "train_ppl": tr["ppl"], "train_acc": tr["acc"],
            "val_loss": va["loss"], "val_ppl": va["ppl"], "val_acc": va["acc"],
            "lr": cur_lr, "seconds": dt,
        })
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        print(f"{epoch:>5}  {tr['loss']:>10.4f}  {tr['ppl']:>10.2f}  {va['loss']:>10.4f}  "
              f"{va['ppl']:>10.2f}  {va['acc']:>7.2%}  {cur_lr:>9.2e}  {dt:>5.1f}s{marker}")

    total_min = (time.time() - t_start) / 60
    print(f"\n训练完成，用时 {total_min:.1f} 分钟；最佳验证 PPL = {best_val_ppl:.2f} → {ckpt_path}")

    # ── 曲线 ──
    if not args.no_plot:
        plot_history(history, fig_path, args.model.upper())
        print(f"训练曲线已保存: {fig_path}")

    # ── 生成样例（用最优权重，而非最后一个 epoch）──
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    prompts = [p for p in args.sample_prompts.split(",") if p]
    lines_out = [f"# {args.model}  best epoch={ckpt['epoch']}  val_ppl={ckpt['val_ppl']:.2f}",
                 "# 格式: 提示 → 续写（temperature=0.8, top_k=20）"]
    print("\n生成样例（temperature=0.8, top_k=20）:")
    for prompt in prompts:
        ids = [char2idx[c] for c in prompt if c in char2idx] or [0]
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(idx, max_new_tokens=30, temperature=0.8, top_k=20)
        reply = "".join(idx2char.get(i, "?") for i in out[0].tolist())
        print(f"  {prompt} → {reply}")
        lines_out.append(f"{prompt} → {reply}")

    with open(os.path.join(args.out_dir, f"samples_{args.model}{args.tag}.txt"),
              "w", encoding="utf-8") as f:
        f.write("\n".join(lines_out) + "\n")

    # ── 一行汇总，方便填报告表格 ──
    print(f"\nSUMMARY  model={args.model}  params={n_params:,}  "
          f"best_val_ppl={best_val_ppl:.2f}  minutes={total_min:.1f}  ckpt={ckpt_path}")


if __name__ == "__main__":
    main()
