import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from tqdm.auto import tqdm

from helper import VocabMapper, prepare_dataloaders
from model import TokenEncoder, HybridDecoder, SequenceTranslator

def compute_exact_accuracy(model, loader, tgt_vocab, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for src, src_lens, tgt in loader:
            src, src_lens, tgt = (x.to(device) for x in (src, src_lens, tgt))
            pred = model.infer_greedy(tgt_vocab, src_lens, src, max_length=tgt.size(1))
            for b in range(src.size(0)):
                pred_str = tgt_vocab.convert_to_text(pred[b].cpu().tolist())
                gold_str = tgt_vocab.convert_to_text(tgt[b, 1:].cpu().tolist())
                correct += (pred_str == gold_str)
            total += src.size(0)
    return correct / total if total else 0.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_size", type=int, default=256)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--enc_layers", type=int, default=2)
    parser.add_argument("--cell", type=str, default="LSTM", choices=["LSTM", "GRU"])
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--attention", action="store_true", help="Enable Bahdanau Attention")

    args = parser.parse_args()
    print("Hyper-parameters:", vars(args))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaders, src_vocab, tgt_vocab = prepare_dataloaders(
        language_code="hi",
        batch_sz=args.batch_size,
        device=device
    )

    enc = TokenEncoder(
        src_vocab.total_vocab_size(),
        args.emb_size,
        args.hidden_size,
        args.enc_layers,
        args.cell,
        args.dropout
    ).to(device)

    dec = HybridDecoder(
        tgt_vocab.total_vocab_size(),
        args.emb_size,
        args.hidden_size,
        args.hidden_size,
        args.enc_layers,
        args.cell,
        args.dropout,
        attention_enabled=args.attention
    ).to(device)

    model = SequenceTranslator(enc, dec, pad_token_id=src_vocab.get_pad_index(), device=device).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.get_pad_index())
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    save_path = "attn_best_model.pth" if args.attention else "noattn_best_model.pth"

    for epoch in tqdm(range(1, args.epochs + 1), desc="Epochs"):
        model.train(); total_loss = 0
        for src, src_lens, tgt in tqdm(loaders["train"], leave=True):
            src, src_lens, tgt = src.to(device), src_lens.to(device), tgt.to(device)
            optimizer.zero_grad()
            out = model(src, src_lens, tgt)
            loss = criterion(out.view(-1, out.size(-1)), tgt[:, 1:].reshape(-1))
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); total_loss += loss.item()

        train_loss = total_loss / len(loaders["train"])

        model.eval(); val_loss = 0
        with torch.no_grad():
            for src, src_lens, tgt in loaders["dev"]:
                src, src_lens, tgt = src.to(device), src_lens.to(device), tgt.to(device)
                out = model(src, src_lens, tgt)
                val_loss += criterion(out.view(-1, out.size(-1)), tgt[:, 1:].reshape(-1)).item()
        val_loss /= len(loaders["dev"])

        train_acc = compute_exact_accuracy(model, loaders["train"], tgt_vocab, device)
        val_acc = compute_exact_accuracy(model, loaders["dev"], tgt_vocab, device)

        print(f"Epoch {epoch:2d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
              f"train_acc {train_acc:.4f} | val_acc {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"[epoch {epoch}] new best val_acc={val_acc:.4f} → saved {save_path}")

    # Rebuild test loader with batch_size = 1 for accurate evaluation
    from helper import prepare_dataloaders as build_loader_1batch
    test_loader_1, _, _ = build_loader_1batch(
        language_code="hi",
        batch_sz=1,
        device=device
    )
    test_acc = compute_exact_accuracy(model, loaders["test"], tgt_vocab, device)
    print(f"\n★ Test accuracy (exact match): {test_acc:.4f}")
