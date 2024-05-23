
import torch
from torch.nn import functional as F
import mixlib
# this is a test for TFLOPS of int8 GEMM with different shapes
# please compile the MIXQ lib: mixlib by
# cd quantkernel
# python setup.py install
test_batch_pool = [32, 64, 128, 256, 512, 1024, 2048]
GEMM_shapes = [(4096, 11008) , (11008, 4096)]

for shape in GEMM_shapes:

    in_features, out_features = shape
    @torch.no_grad()
    def test_quant_linear_a16_w16(M, N ,K) -> float:
        weight = torch.rand(N, K).to(torch.int8).cuda()
        x = torch.rand(M, K).to(torch.int8).cuda()

        elapsed_time_ms = 0
        iterations = 100

        for _ in range(10):
            y = mixlib.gemm(x, weight, M, N, K)
        
        for _ in range(iterations):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()
            y = mixlib.gemm(x, weight, M, N, K)
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time_ms += start_event.elapsed_time(end_event)

        total_ops = M * N * K * 2 * iterations
        tflops = total_ops / elapsed_time_ms / 10**9
        return tflops, elapsed_time_ms

    print("0: INT8 GEMM with shape N=%d K=%d \n"%(out_features, in_features))

    a16w16_flops = []
    for m in test_batch_pool:
        tflops, elapsed_time_ms = test_quant_linear_a16_w16(m, out_features, in_features)
        a16w16_flops.append(tflops)
        print(tflops,end=",")
    print("\n")
