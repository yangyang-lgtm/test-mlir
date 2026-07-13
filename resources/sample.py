import torch
import triton
import triton.language as tl


@triton.jit
def kernel(a, b, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * num

    value = tl.full([BLOCK], 0, dtype=tl.float32)
    for offset in range(base, num, BLOCK):
        mask_offset = tl.arange(0, BLOCK) + offset
        mask = mask_offset < num
        value += tl.load(a + mask_offset, mask, other=0)

    res = tl.sum(value)
    tl.store(b, res)


a = torch.randn(size=(1024, ), device="cuda")
b = torch.randn(size=(1, ), device="cuda")
kernel[(1,)](a, b, a.numel(), 128)

print(b - a.sum())

# a = torch.randn(size=(1024,), device='cuda')
# b = torch.randn(size=(1024,), device='cuda')
# c = torch.randn(size=(1024,), device='cuda')
# d = torch.randn(size=(1024,), device='cuda')
#
# kernel[(1,)](a,b,c,d, d.numel(), 128)
#
# print(d - (a + b - c))
