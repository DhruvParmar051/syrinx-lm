import time, sys, torch
from transformers import LlamaConfig, LlamaForCausalLM

dev = "mps"
shapes = {"50m": (512, 10, 1536, 8), "125m": (768, 12, 2560, 12)}


def run(name, bs, amp, steps=15, warm=4):
    d, L, ff, h = shapes[name]
    cfg = LlamaConfig(
        vocab_size=32768,
        hidden_size=d,
        num_hidden_layers=L,
        intermediate_size=ff,
        num_attention_heads=h,
        num_key_value_heads=h,
        max_position_embeddings=1024,
        tie_word_embeddings=True,
        attn_implementation="sdpa",
    )
    m = LlamaForCausalLM(cfg).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, fused=False)
    n = sum(p.numel() for p in m.parameters())
    x = torch.randint(0, 32768, (bs, 1024), device=dev)
    torch.mps.empty_cache()

    def step():
        with torch.autocast("mps", dtype=torch.bfloat16, enabled=amp):
            out = m(input_ids=x, labels=x)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        return out.loss

    for _ in range(warm):
        step()
    torch.mps.synchronize()
    t = time.perf_counter()
    for _ in range(steps):
        l = step()
    torch.mps.synchronize()
    dt = time.perf_counter() - t
    tps = bs * 1024 * steps / dt
    print(
        f"{name} params={n:,} bs={bs} amp={amp} tok/s={tps:,.0f} ms/step={dt / steps * 1000:.0f} "
        f"driver_mem={torch.mps.driver_allocated_memory() / 2**30:.1f}GiB loss={l.item():.2f}",
        flush=True,
    )
    del m, opt
    torch.mps.empty_cache()


for name, bs, amp in [("125m", 8, False), ("125m", 8, True), ("125m", 16, True), ("50m", 16, True)]:
    run(name, bs, amp)
