import torch
from collections.abc import Mapping, Sequence

path = "./results_sink_only/freesolv3_1_3/NN_UQ/trial_0.pt"  # 変更
ckpt = torch.load(path, map_location="cpu", weights_only=False)

print("type:", type(ckpt))
print("keys:", list(ckpt.keys()))

def summarize(name, x, max_items=1000):
    if torch.is_tensor(x):
        print(f"{name}: Tensor shape={tuple(x.shape)} dtype={x.dtype}")
        return
    if isinstance(x, (int, float, str, bool, type(None))):
        print(f"{name}: {type(x).__name__} = {x}")
        return
    if isinstance(x, Mapping):
        print(f"{name}: dict len={len(x)} keys(sample)={list(x.keys())[:10]}")
        return
    if isinstance(x, Sequence) and not isinstance(x, (str, bytes, bytearray)):
        print(f"{name}: list/seq len={len(x)}")
        for i, item in enumerate(x[:max_items]):
            if torch.is_tensor(item):
                print(f"  [{i}] Tensor shape={tuple(item.shape)} dtype={item.dtype}")
            else:
                t = type(item).__name__
                preview = str(item)
                if len(preview) > 120:
                    preview = preview[:117] + "..."
                print(f"  [{i}] {t}: {preview}")
        return
    print(f"{name}: {type(x).__name__}")

for k in ckpt:
    summarize(k, ckpt[k])

# 追加で系列の整合性チェック
lens = {}
for k in ["cumulative_costs","test_losses","selected_inputs","selected_scores","observed_sink_vals"]:
    v = ckpt.get(k, None)
    if isinstance(v, list):
        lens[k] = len(v)
print("lengths:", lens)