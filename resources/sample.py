import torch
import triton
import triton.language as tl


@triton.jit
def kernel(a, b, c, num, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * num
    for offset in range(base, num, BLOCK):
        mask_offset = tl.arange(0, BLOCK) + offset
        mask = mask_offset < num
        ta = tl.load(a + mask_offset, mask)
        tb = tl.load(b + mask_offset, mask)
        out = ta + tb
        tl.store(c + mask_offset, out, mask)



a = torch.randn(size=(1024,), device='cuda')
b = torch.randn(size=(1024,), device='cuda')
c = torch.randn(size=(1024,), device='cuda')

kernel[(1,)](a,b,c, c.numel(), 128)

print(c - a - b)
