import torch
import math
from attention_cutlass_fp8 import flash_attention_v2_cutlass
import math
import time
# offical flash attention implement
# from vllm_flash_attn import flash_attn_func as flash_attn_func_offical

'''
simple attention implement without multi head
'''
torch.manual_seed(180)
def get_tensors(BS, HEAD, SEQLEN, DIM, dtype=torch.float16):
    q = (torch.empty((BS, HEAD, SEQLEN, DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=1).requires_grad_())
    k = (torch.empty((BS, HEAD, SEQLEN, DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=1).requires_grad_())
    v = (torch.empty((BS, HEAD, SEQLEN, DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=1).requires_grad_())
    # q = q.clamp(-0.1, 0.1).to(torch.float8_e5m2)
    # k = k.clamp(-0.1, 0.1).to(torch.float8_e5m2)
    # v = v.clamp(-0.1, 0.1).to(torch.float8_e5m2)
    return q, k, v

def self_attention(q, k, v, causal=True, sm_scale=1):
    SEQLEN = q.shape[-2]
    M = torch.tril(torch.ones((SEQLEN, SEQLEN), device="cuda"))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    if causal:
        p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).half()
    ref_out = torch.matmul(p, v)
    return ref_out


def run_benchmark(epoch, warmup, func, *args, **kwargs):
    # warmup phase
    for _ in range(warmup):
        _ = func(*args, **kwargs)
    torch.cuda.synchronize()
    time_s = time.time()
    for _ in range(epoch):
        _ = func(*args, **kwargs)
        torch.cuda.synchronize()
    time_e = time.time() - time_s
    return time_e


def main(bs=1, head=32, seq_len=4096, head_dim=128):
    BS, HEAD, SEQLEN, DIM = bs, head, seq_len, head_dim
    q,k,v = get_tensors(BS, HEAD, SEQLEN, DIM, dtype=torch.float16)

    warmup = 5
    epoch = 20
    
    is_causal = True
    sm_scale = 1.0 / math.sqrt(SEQLEN)
    
    q_d16 = q.to(torch.float16)
    k_d16 = k.to(torch.float16)
    v_d16 = v.to(torch.float16)
    base_time = run_benchmark(epoch, warmup, self_attention, q_d16, k_d16, v_d16, causal=is_causal, sm_scale=sm_scale)
    baseline = self_attention(q_d16, k_d16, v_d16, causal=is_causal, sm_scale=sm_scale)

    
    q = q.to(torch.float8_e4m3fn)
    k = k.to(torch.float8_e4m3fn)
    v = v.to(torch.float8_e4m3fn)
    v = v.transpose(2, 3).contiguous()
    flash2_time = run_benchmark(epoch, warmup, flash_attention_v2_cutlass, q, k, v, is_causal, sm_scale)
    flash2_cutlass_ref = flash_attention_v2_cutlass(q, k, v, is_causal, sm_scale)[0]
    print(f"flash2_cutlass_fp8: {flash2_time * 1000 / epoch}ms\n")

    # print(f"difference:\n", (baseline - flash2_cutlass_ref.to(torch.float16)), flush=True)
    print(f"max diff: {torch.max(torch.abs(baseline - flash2_cutlass_ref.to(torch.float16)))}", flush=True)

    assert torch.allclose(baseline, flash2_cutlass_ref.to(torch.float16), rtol=0.2, atol=0.2)
    
    print(f"head:{head}, seq_len:{seq_len}, head_dim:{head_dim}, baseline: {base_time * 1000 / epoch}ms, flash2_cutlass_fp8: {flash2_time * 1000 / epoch}ms")

if __name__ == "__main__":
    epoch = 1
    for _ in range(epoch):
        for bs in [1]:
            for head in [16, 32, 64]:
                for seq_len in [1024]:
                    for head_dim in [64, 128]:
                        main(bs, head, seq_len, head_dim)

# if __name__ == "__main__":
#     epoch = 1
#     for _ in range(epoch):
#         for bs in [1]:
#             for head in [64]:
#                 for seq_len in [1024]:
#                     for head_dim in [128]:
#                         main(bs, head, seq_len, head_dim)


