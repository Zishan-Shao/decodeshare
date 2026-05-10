#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--calib_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--k", type=int, default=130)
    ap.add_argument("--max_samples", type=int, default=512)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, device_map=None).to(args.device)
    model.eval()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # register hooks on each transformer layer to grab last-token hidden state
    layers = model.model.layers
    H = model.config.hidden_size
    feats = [[] for _ in range(len(layers))]

    handles = []
    def make_hook(i):
        def hook(module, inp, out):
            hs = out[0] if isinstance(out, (tuple, list)) else out
            v = hs[:, -1, :].detach().float().cpu()  # (1,H)
            feats[i].append(v[0])
        return hook

    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_hook(i)))

    # read prompts
    n = 0
    with open(args.calib_jsonl, "r") as f:
        for line in f:
            if n >= args.max_samples: break
            obj = json.loads(line)
            text = obj["text"]
            ids = tok(text, return_tensors="pt", truncation=True, max_length=args.seq_len).to(args.device)
            with torch.no_grad():
                _ = model(**ids, use_cache=False)
            n += 1

    for h in handles: h.remove()

    # build PCA basis per layer
    for i in range(len(layers)):
        X = torch.stack(feats[i], dim=0)  # (N,H)
        X = X - X.mean(dim=0, keepdim=True)
        # SVD: X = U S V^T, take V[:, :k]
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        Q = Vh.T[:, :args.k].contiguous()  # (H,k)
        torch.save(Q, out_dir / f"layer_{i}.pt")

    print(f"[OK] wrote {len(layers)} basis files to {out_dir}")

if __name__ == "__main__":
    main()
