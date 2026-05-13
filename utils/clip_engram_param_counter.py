from dataclasses import dataclass
from typing import Optional
from model.model_configs import clip_config, clip_config_set

 
def count_params(cfg: clip_config):
    def L(i, o, bias=True): return i*o + (o if bias else 0)
    def LN(d): return 2*d
    def attn(d): return 4 * L(d, d)
    def dense_ffn(d, mul): return L(d, d*mul) + L(d*mul, d)
    def moe_ffn(d, mul, n): return L(d, n, bias=False) + n * dense_ffn(d, mul)
    def moe_ffn_active(d, mul, n, k): return L(d, n, bias=False) + k * dense_ffn(d, mul)
    def schedule(depth, use_moe, every_2):
        s = [use_moe and (not every_2 or i%2==1) for i in range(depth)]
        return s.count(False), s.count(True)
 
    v, t = cfg.vit_config, cfg.tet_config
    vd, vm = schedule(v.depth, v.use_moe, v.every_2)
    td, tm = schedule(t.depth, t.use_moe, t.every_2)
 
    # ── ViT (patch embed + cls + pos + LN + attention layers only)
    n_patches = (v.img_size // v.patch_size) ** 2
    vit_embed   = L(v.in_channels * v.patch_size**2, v.emb_dim) + v.emb_dim + (n_patches+1)*v.emb_dim + LN(v.emb_dim)
    vit_attn    = v.depth * (attn(v.emb_dim) + 2*LN(v.emb_dim))
    vit_dense   = vd * dense_ffn(v.emb_dim, v.ffn_mul)
    vit_moe     = vm * moe_ffn(v.emb_dim, v.ffn_mul, v.n_experts)
    vit_total   = vit_embed + vit_attn + vit_dense + vit_moe
    vit_active  = vit_embed + vit_attn + vit_dense + vm * moe_ffn_active(v.emb_dim, v.ffn_mul, v.n_experts, v.k)
 
    # ── Text encoder (token emb + pos emb + LN + attention layers only)
    tok_emb     = t.vocab_size * t.emb_dim
    pos_emb     = t.max_ctx_len * t.emb_dim
    tet_embed   = tok_emb + pos_emb + LN(t.emb_dim)
    tet_attn    = t.depth * (attn(t.emb_dim) + 2*LN(t.emb_dim))
    tet_dense   = td * dense_ffn(t.emb_dim, t.ffn_mul)
    tet_moe     = tm * moe_ffn(t.emb_dim, t.ffn_mul, t.n_experts)
    tet_total   = tet_attn + tet_dense + tet_moe
    tet_active  = tet_attn + tet_dense + tm * moe_ffn_active(t.emb_dim, t.ffn_mul, t.n_experts, t.k)
 
    # ── Embedding tables (tok + pos, separated out)
    emb_total   = tok_emb + pos_emb + vit_embed  # vit patch/cls/pos counts here too
 
    # ── MoE (router + experts across both encoders)
    moe_total   = vit_moe + tet_moe
 
    # ── Dense FFN
    dense_total = vit_dense + tet_dense
 
    # ── CLIP projection
    proj_dim    = min(v.emb_dim, t.emb_dim)
    clip_proj   = L(v.emb_dim, proj_dim, bias=False) + L(t.emb_dim, proj_dim, bias=False) + 1
 
    # ── Engram
    def engram(ve, te, vc, tc): return te.max_ngram-1 * te.engram_vocab_size * te.engram_embd_d + (L(te.engram_embd_d, te.embd_d) + L(te.engram_embd_d, te.embd_d) * tc.hc_mult) + (L(ve.engram_embd_d, ve.embd_d) + L(ve.engram_embd_d, ve.embd_d) * vc.hc_mult)
    engram_total = engram(cfg.tet_engram_config, cfg.vit_engram_config, cfg.vit_config, cfg.tet_config) if cfg.tet_engram_config is not None else 0

    grand = vit_total + tet_total + clip_proj + engram_total
 
    def gb(n, b=2): return n*b/1024**3
    def row(label, n, note=""): 
        print(f"  {label:<22} {n/1e9:>8.4f}B   {gb(n):>6.2f} GB bf16  {note}")
    def row_active(label, total, active, note=""):
        print(f"  {label:<22} {total/1e9:>8.4f}B   {gb(total):>6.2f} GB bf16  (active: {active/1e9:.4f}B / {gb(active):.2f} GB bf16)  {note}")
 
    S = "─"*80
    print(S)
    print(f"  {'Component':<22} {'Params':>9}    {'Memory (bf16)':>12}")
    print(S)
 
    print("[ViT]")
    row("  patch/cls/pos emb",  vit_embed,  f"patches={n_patches}")
    row("  attention ×"+str(v.depth),       vit_attn)
    row("  dense FFN ×"+str(vd),            vit_dense)
    row("  MoE FFN ×"+str(vm),              vit_moe,   f"n_experts={v.n_experts}")
    row_active("  TOTAL", vit_total, vit_active, f"k={v.k}")
    print()
 
    print("[Text Encoder]")
    row("  tok emb",            tok_emb,    f"vocab={t.vocab_size}")
    row("  pos emb",            pos_emb,    f"ctx={t.max_ctx_len}")
    row("  attention ×"+str(t.depth),       tet_attn)
    row("  dense FFN ×"+str(td),            tet_dense)
    row("  MoE FFN ×"+str(tm),              tet_moe,   f"n_experts={t.n_experts}")
    row_active("  TOTAL", tet_total, tet_active, f"k={t.k}")
    print()
 
    print("[Embedding Tables]  (subset, not additive)")
    row("  ViT patch/cls/pos",  vit_embed)
    row("  TET tok+pos",        tok_emb + pos_emb)
    row("  TOTAL",              vit_embed + tok_emb + pos_emb)
    print()
 
    print("[MoE]  (subset, not additive)")
    row("  ViT MoE",            vit_moe,   f"×{vm} layers")
    row("  TET MoE",            tet_moe,   f"×{tm} layers")
    print()
 
    if engram_total > 0:
        print("[Engram]  (host DRAM, offloadable)")
        row("  TOTAL",              engram_total, f"vocab={cfg.vit_engram_config.engram_vocab_size} d={cfg.vit_engram_config.engram_embd_d, cfg.tet_engram_config.engram_embd_d}")
        print()
 
    print("[CLIP Projection]")
    row("  visual+text proj",   clip_proj,  f"proj_dim={proj_dim}")
    print()
    
    print(f"{'─'*80}")
    row("ACTIVATE", vit_active + tet_active)
    row("MOE",              moe_total)  
    if engram_total > 0 :
        row("ENGRAM   (DRAM)",   engram_total)
    print()

    print(f"{'─'*80}")
    row("BACKBONE (GPU)",       vit_total + tet_total + clip_proj)
    row("GRAND TOTAL(w/o token embed)",          grand)
    print(f"\n  fp32: {gb(grand,4):.2f} GB  |  bf16: {gb(grand,2):.2f} GB  |  int8: {gb(grand,1):.2f} GB")
    print(f"{'─'*80}")



if __name__ == "__main__" :
    count_params(clip_config_set.clip_1_5B_engram)
    print("-" * 32) 