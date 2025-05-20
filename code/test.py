import torch
from helper import prepare_dataloaders
from model import TokenEncoder, HybridDecoder, SequenceTranslator
if __name__ == "__main__":
    # ---------------------- Checkpoint & Config ----------------------
    MODEL_PATH = "attn_best_model.pth"  # adjust path if needed
    LANGUAGE = "hi"

    BEST_CONFIG = dict(
        emb_size     = 256,
        hidden_size  = 512,
        enc_layers   = 2,
        cell         = "LSTM",
        dropout      = 0.5,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------------- Load Test Data with batch_size=1 ----------------------
    loaders, src_vocab, tgt_vocab = prepare_dataloaders(
        language_code=LANGUAGE,
        batch_sz=1,            # Important for per-sample decoding
        device=device
    )

    # ---------------------- Rebuild Model ----------------------
    encoder = TokenEncoder(
        src_vocab.total_vocab_size(),
        BEST_CONFIG["emb_size"],
        BEST_CONFIG["hidden_size"],
        BEST_CONFIG["enc_layers"],
        BEST_CONFIG["cell"],
        BEST_CONFIG["dropout"]
    ).to(device)

    decoder = HybridDecoder(
        tgt_vocab.total_vocab_size(),
        BEST_CONFIG["emb_size"],
        BEST_CONFIG["hidden_size"],
        BEST_CONFIG["hidden_size"],
        BEST_CONFIG["enc_layers"],
        BEST_CONFIG["cell"],
        BEST_CONFIG["dropout"],
        attention_enabled=True
    ).to(device)

    model = SequenceTranslator(
        encoder, decoder,
        pad_token_id=src_vocab.get_pad_index(),
        device=device
    ).to(device)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # ---------------------- Evaluate Accuracy ----------------------
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

    # ---------------------- Run Test ----------------------
    acc = compute_exact_accuracy(model, loaders["test"], tgt_vocab, device)
    print(f"✅ Exact-match TEST accuracy (batch=1): {acc * 100:.2f}%")
