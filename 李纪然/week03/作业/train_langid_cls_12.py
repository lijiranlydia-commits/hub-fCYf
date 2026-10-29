# -*- coding: utf-8 -*-

import os
import csv
import time
import random
import argparse
import unicodedata
from collections import Counter, OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


from train_langid_cls import (LangIDClassifier, normalize, encode,
                              macro_f1, confusion, print_confusion,
                              MAX_LEN, EMBED_DIM, HIDDEN_DIM, DROPOUT,
                              BATCH_SIZE, LR)

SEED = 42
EPOCHS = 8
NUM_THREADS = 1

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data", "langid", "v12")
VOCAB_PATH = os.path.join(HERE, "data", "langid", "vocab", "vocab_v12.txt")

# 12 个语种：课堂 8 个 + ko(韩) th(泰) vi(越) tr(土)
LABELS = ["ar", "en", "es", "hi", "ja", "ko", "pt", "ru", "th", "tr", "vi", "zh"]


def build_char_vocab(sentences, min_freq=1):

    freq = Counter()
    for s in sentences:
        freq.update(normalize(s))
    chars = sorted(((c, n) for c, n in freq.items() if n >= min_freq),
                   key=lambda kv: (-kv[1], kv[0]))
    vocab = OrderedDict([("<PAD>", 0), ("<UNK>", 1)])
    for ch, _ in chars:
        vocab[ch] = len(vocab)
    return vocab, freq


def save_vocab(vocab, freq, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 字符\tid\t出现次数\n")
        for ch, idx in vocab.items():
            f.write(f"{ch}\t{idx}\t{freq.get(ch, 0)}\n")


def load_vocab_file(path):
    vocab = OrderedDict()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            ch, idx, _freq = line.rstrip("\n").split("\t")
            vocab[ch] = int(idx)
    return vocab


def load_split(data_dir, name):
    path = os.path.join(data_dir, f"{name}.csv")
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((r["text"], r["label"]))
    return rows


class TextDataset(Dataset):
    def __init__(self, rows, vocab, label2id):
        self.X = [encode(t, vocab) for t, _ in rows]
        self.y = [label2id[l] for _, l in rows]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return torch.tensor(self.X[i], dtype=torch.long), self.y[i]


def predict(model, loader):
    model.eval()
    preds, golds = [], []
    with torch.no_grad():
        for X, y in loader:
            preds += model(X).argmax(dim=1).tolist()
            golds += y.tolist()
    return preds, golds


def coverage(vocab, freq, ratio=0.95):
    """前多少个字符覆盖 ratio 的字符出现次数 —— 和 DATA_CARD 里的数字对一下。"""
    total = sum(freq.values())
    acc, n = 0, 0
    for ch in vocab:
        if ch in ("<PAD>", "<UNK>"):
            continue
        n += 1
        acc += freq.get(ch, 0)
        if acc >= ratio * total:
            return n
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--vocab", default=VOCAB_PATH)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--threads", type=int, default=NUM_THREADS)
    ap.add_argument("--rebuild-vocab", action="store_true",
                    help="强制用当前训练集重新生成字符表（默认有文件就复用）")
    ap.add_argument("--save", default=os.path.join(HERE, "langid_rnn_v12.pt"))
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    random.seed(SEED)
    torch.manual_seed(SEED)

    train_rows = load_split(args.data_dir, "train")
    dev_rows = load_split(args.data_dir, "dev")
    test_rows = load_split(args.data_dir, "test")

    # 数据里的标签必须正好是这 12 个：多一个少一个，模型输出的格子就对不上
    seen = sorted({l for _, l in train_rows})
    if seen != sorted(LABELS):
        raise SystemExit(
            f"数据标签和 LABELS 不一致\n"
            f"  数据里有（{len(seen)}）：{' '.join(seen)}\n"
            f"  LABELS（{len(LABELS)}）：{' '.join(LABELS)}"
        )
    label2id = {l: i for i, l in enumerate(LABELS)}
    num_classes = len(LABELS)

    # ── 字符表：没有就现建，有就复用 ──────────────────────────────
    if args.rebuild_vocab or not os.path.exists(args.vocab):
        vocab, freq = build_char_vocab([t for t, _ in train_rows])
        save_vocab(vocab, freq, args.vocab)
        src = "刚生成"
    else:
        vocab = load_vocab_file(args.vocab)
        freq = Counter()
        with open(args.vocab, encoding="utf-8") as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                ch, _i, c = line.rstrip("\n").split("\t")
                freq[ch] = int(c)
        src = "复用已有文件"

    print(f"数据目录：{args.data_dir}")
    print(f"  训练 {len(train_rows)} 条 / 验证 {len(dev_rows)} 条 / 测试 {len(test_rows)} 条")
    print(f"  类别 {num_classes} 个：{' '.join(LABELS)}")
    print(f"  字符表 {len(vocab)} 个字符（{src}，{os.path.basename(args.vocab)}）"
          f"，前 {coverage(vocab, freq)} 个覆盖 95% 出现次数")
    print(f"  随机猜的准确率基准：{1/num_classes:.1%}")

    train_loader = DataLoader(TextDataset(train_rows, vocab, label2id),
                              batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(TextDataset(dev_rows, vocab, label2id), batch_size=128)
    test_loader = DataLoader(TextDataset(test_rows, vocab, label2id), batch_size=128)

    model = LangIDClassifier(len(vocab), num_classes)
    print(f"  模型参数量：{sum(p.numel() for p in model.parameters()):,}\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    t0 = time.time()
    best_acc, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for X, y in train_loader:
            loss = criterion(model(X), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        preds, golds = predict(model, dev_loader)
        acc = sum(p == g for p, g in zip(preds, golds)) / len(golds)
        # 留一份 dev 上最好的权重，避免最后一个 epoch 恰好抖下去
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"Epoch {epoch:2d}/{args.epochs}  loss={total/len(train_loader):.4f}  "
              f"dev_acc={acc:.4f}")
    print(f"\n训练耗时：{time.time() - t0:.0f} 秒（{args.epochs} 个 epoch，{args.threads} 线程）")

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"取 dev 最好的一轮：dev_acc={best_acc:.4f}")

    preds, golds = predict(model, test_loader)
    acc = sum(p == g for p, g in zip(preds, golds)) / len(golds)
    f1 = macro_f1(preds, golds, num_classes)
    print(f"\n测试集：accuracy = {acc:.4f}   macro-F1 = {f1:.4f}")
    print_confusion(confusion(preds, golds, LABELS), LABELS, top=5)

    print("\n  各语种准确率：")
    for c, lab in enumerate(LABELS):
        idx = [i for i, g in enumerate(golds) if g == c]
        hit = sum(preds[i] == c for i in idx)
        print(f"    {lab}: {hit}/{len(idx)} = {hit/max(1,len(idx)):.3f}")

    print("\n--- 推理示例 ---")
    model.eval()
    with torch.no_grad():
        for text in [t for t, _ in test_rows[:6]]:
            logits = model(torch.tensor([encode(text, vocab)]))
            prob = torch.softmax(logits, dim=1)[0]
            pid = int(prob.argmax())
            print(f"  [{LABELS[pid]} {prob[pid]:.2f}]  {text[:50]}")

    if args.save:
        torch.save({"state_dict": model.state_dict(), "vocab": dict(vocab),
                    "labels": LABELS, "maxlen": MAX_LEN,
                    "embed_dim": EMBED_DIM, "hidden_dim": HIDDEN_DIM,
                    "dropout": DROPOUT}, args.save)
        print(f"\n模型已保存：{args.save}")
        print(f"评测：python eval_langid.py --model {os.path.basename(args.save)} "
              f"--data-dir data/langid/v12")


if __name__ == "__main__":
    main()
