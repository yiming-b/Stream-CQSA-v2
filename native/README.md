# next/native -- the native multi-subproblem CQS kernel ("wave" kernel)

Separate from `next/kernel/v*` (one kernel launch per subproblem). Copied from v11
and extended; nothing here touches the previous version.

    csrc/flash_attn/            FlashAttention-2 + CQS sources (v11 copy + wave mode)
      src/flash.h               Flash_fwd_params: cqs_wave, cqs_blk_cu, cqs_bb_cu
      src/flash_fwd_kernel.h    per-subproblem CQS tables (Cqs_mode 3/4 = wave), fp32-only epilogue
      src/flash_bwd_kernel.h    per-subproblem CQS tables (runtime flag)
      src/flash_fwd_launch_template.h   CQS_MODE_SWITCH with the wave modes
      src/wave_kernels.cu       wave_merge (deterministic batched merge), wave_scatter_add
      flash_api.cpp             fwd_wave / bwd_wave entry points + bindings
    setup.py                    builds `cqsa_native` (CQSA_KERNEL_SET=native_common64: hdim64 fp16+bf16 forwards with the wave modes,
                            every backward; head_dim=128 forwards run on cqsa_cuda -- their wave instantiations do not come out of ptxas)
    build.sh / build.slurm      CPU-node build (~20-40 min)
    test_wave.py                exactness tests (kernel == v11 bit for bit at W=1, engine vs fp64, bwd vs fp64, host pool)
    bench_wave.py               vs previous engine / FA-2, timeline plot
    run_test.slurm              gpu-test job: tests then bench

Engine: `stream_cqsa/native_wave.py` (wave_forward, wave_backward, wave_attention, ChunkPool).

Status / how to continue:
    build:  sbatch native/build.slurm         -> native/cqsa_native*.so, log results/native/build_native_build_<job>.out
    test:   sbatch native/run_test.slurm      -> results/native/native_test_<job>.out, plots results/native/wave_*.png
