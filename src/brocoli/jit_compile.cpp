#include <asmjit/asmjit.h>
#include <iostream>

using namespace asmjit;

// Define the function signature: void(float* A, float* B, float* C, int K)
typedef void (*MatMulKernel)(float*, float*, float*, int);

MatMulKernel generate_kernel() {
    JitRuntime rt;
    CodeHolder code;
    code.init(rt.environment());

    // 1. SELECT ASSEMBLER BASED ON CPU
    #if defined(__x86_64__) || defined(_M_X64)
        x86::Compiler cc(&code);
        cc.addFunc(FuncSignatureT<void, float*, float*, float*, int>());

        // Register Allocation
        x86::Gp a_ptr = cc.newIntPtr("a_ptr");
        x86::Gp b_ptr = cc.newIntPtr("b_ptr");
        x86::Gp c_ptr = cc.newIntPtr("c_ptr");
        x86::Gp k_val = cc.newInt32("k_val");

        cc.setArg(0, a_ptr); cc.setArg(1, b_ptr); 
        cc.setArg(2, c_ptr); cc.setArg(3, k_val);

        // JIT-Specialized Loop: Use AVX2 YMM registers
        x86::Ymm acc = cc.newYmmPs("acc");
        cc.vxorps(acc, acc, acc); // Zero out accumulator

        Label loop = cc.newLabel();
        cc.bind(loop);
        {
            x86::Ymm va = cc.newYmmPs();
            x86::Ymm vb = cc.newYmmPs();
            cc.vbroadcastss(va, x86::ptr(a_ptr));
            cc.vmovups(vb, x86::ptr(b_ptr));
            cc.vfmadd231ps(acc, va, vb); // Fused Multiply-Add

            cc.add(a_ptr, 4);
            cc.add(b_ptr, 32); // Step by 8 floats
            cc.dec(k_val);
            cc.jnz(loop);
        }
        cc.vmaxps(acc, acc, x86::ymm0); // Fused ReLU
        cc.vmovups(x86::ptr(c_ptr), acc);
        cc.endFunc();

    #elif defined(__aarch64__)
        a64::Compiler cc(&code);
        cc.addFunc(FuncSignatureT<void, float*, float*, float*, int>());
        
        // ARM NEON Logic: Use V registers (128-bit)
        // (Similar to x86, but using cc.fmla for ARM)
    #endif

    // 2. FINALIZE AND COMPILE
    MatMulKernel kernel;
    rt.add(&kernel, &code);
    return kernel;
}

int main() {
    MatMulKernel my_jit_kernel = generate_kernel();
    // Use the kernel...
    std::cout << "🚀 JIT Kernel Compiled and Ready." << std::endl;
    return 0;
}
