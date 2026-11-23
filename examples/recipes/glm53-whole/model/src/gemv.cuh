// Warp-cooperative GEMV, linear dispatcher, norms, gating, and residual add.

static __device__ __forceinline__ u32 warp_start(u64 cursor, u32 wid, u32 wstep) {
    u32 rem = static_cast<u32>(cursor % wstep);
    return wid >= rem ? (wid - rem) : (wid + wstep - rem);
}

static __device__ __forceinline__ void store_warp(float acc[8], u32 n, u32 r0, u32 o, float* y,
                                                  u32 ys, u32 lane) {
    for (u32 r = 0; r < n; r++) {
        float s = warp_sum(acc[r]);
        if (lane == 0u) y[u64(r0 + r) * ys + o] = s;
    }
}

template <bool IS_BF16>
static __device__ __forceinline__ float4 load_w4(const u8* row_ptr, u32 i) {
    if (IS_BF16) {
        uint2 h = *reinterpret_cast<const uint2*>(reinterpret_cast<const u16*>(row_ptr) + i);
        return make_float4(__uint_as_float(h.x << 16), __uint_as_float(h.x & 0xFFFF0000u),
                           __uint_as_float(h.y << 16), __uint_as_float(h.y & 0xFFFF0000u));
    }
    return *reinterpret_cast<const float4*>(reinterpret_cast<const float*>(row_ptr) + i);
}

template <bool IS_BF16>
static __device__ __forceinline__ float load_w1(const u8* row_ptr, u32 i) {
    if (IS_BF16) {
        return __uint_as_float(u32(reinterpret_cast<const u16*>(row_ptr)[i]) << 16);
    }
    return reinterpret_cast<const float*>(row_ptr)[i];
}

static __device__ __forceinline__ u32 sm_u32_raw(const u32* s, u32 byte_off) {
    u32 w = byte_off >> 2;
    u32 sh = (byte_off & 3u) << 3;
    if (sh == 0u) return s[w];
    u32 out;
    asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(out) : "r"(s[w]), "r"(s[w + 1u]), "r"(sh));
    return out;
}

static __device__ __forceinline__ u32 e4m3x2_to_f16x2(u32 packed) {
    u32 h;
    u16 q = static_cast<u16>(packed);
    asm("{.reg .b16 tmp; mov.b16 tmp, %1; cvt.rn.f16x2.e4m3x2 %0, tmp;}" : "=r"(h) : "h"(q));
    return h;
}

static __device__ __forceinline__ u32 int4x2_to_f16x2(u32 byte_pair) {
    u32 h = 0x64006400u | (byte_pair & 0x0Fu) | ((byte_pair & 0xF0u) << 12);
    u32 out;
    asm("sub.rn.f16x2 %0, %1, %2;" : "=r"(out) : "r"(h), "r"(0x64086408u));
    return out;
}

static __device__ __forceinline__ float dot4_f16x2(u32 hw0, u32 hw1, uint2 xh) {
    u32 prod;
    asm("mul.rn.f16x2 %0, %1, %2; fma.rn.f16x2 %0, %3, %4, %0;"
        : "=r"(prod)
        : "r"(hw0), "r"(xh.x), "r"(hw1), "r"(xh.y));
    return trellis::f16_to_f32(prod & 0xFFFFu) + trellis::f16_to_f32(prod >> 16);
}

static __device__ __forceinline__ void pipe_next(u32 cur_u, u32 cur_ch, u32 chunks, u32 step,
                                                 u32* next_u, u32* next_ch) {
    u32 nc = cur_ch + 1u;
    *next_u = nc == chunks ? (cur_u + step) : cur_u;
    *next_ch = nc == chunks ? 0u : nc;
}

template <typename Fetch, typename Store, typename Compute, typename Done>
static __device__ __forceinline__ void pipe_run(u32 total, u32 wid, u32 wstep, u32 chunks,
                                                Fetch fetch, Store store, Compute compute,
                                                Done done) {
    u32 n_items = total > wid ? ((total - 1u - wid) / wstep + 1u) : 0u;
    u32 total_iters = n_items * chunks;
    if (total_iters == 0u) return;
    u32 cur_u = wid, cur_ch = 0u;
    fetch(cur_u, cur_ch);
    for (u32 it = 0; it < total_iters; it++) {
        store();
        __syncwarp();
        u32 next_u, next_ch;
        pipe_next(cur_u, cur_ch, chunks, wstep, &next_u, &next_ch);
        if (it + 1u < total_iters) fetch(next_u, next_ch);
        compute(cur_ch);
        __syncwarp();
        if (cur_ch + 1u == chunks) done(cur_u);
        cur_u = next_u;
        cur_ch = next_ch;
    }
}

static __device__ __forceinline__ void stage_x_f16(u32* x_h, const float* x, u32 n) {
    for (u32 i = threadIdx.x << 2; i < n; i += blockDim.x << 2) {
        float4 xv = *reinterpret_cast<const float4*>(x + i);
        u32 p0 = trellis::f32x2_to_f16x2(xv.x, xv.y);
        u32 p1 = trellis::f32x2_to_f16x2(xv.z, xv.w);
        *reinterpret_cast<uint2*>(x_h + (i >> 1)) = make_uint2(p0, p1);
    }
    __syncthreads();
}

template <bool IS_BF16>
static __device__ __noinline__ void gemv_dense_1(const mat& w, const float* x, float* y, u32 out,
                                                 u64 cursor) {
    constexpr u32 elem_bytes = IS_BF16 ? 2u : 4u;
    constexpr u64 align_mask = IS_BF16 ? 7ull : 15ull;
    u32 wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    u32 wstep = (gridDim.x * blockDim.x) >> 5;
    u32 lane = threadIdx.x & 31u;
    u32 vec_end = w.in & ~3u;
    bool x_vec = (reinterpret_cast<u64>(x) & 15u) == 0u && vec_end > 0u;
    bool bf16_v4 = IS_BF16 && x_vec && (w.in & 7u) == 0u
                   && (reinterpret_cast<u64>(w.p) & 15ull) == 0ull;
    for (u32 o = warp_start(cursor, wid, wstep); o < out; o += wstep) {
        const u8* wr = w.p + u64(o) * w.in * elem_bytes;
        float acc = 0.0f;
        if (bf16_v4) {
            const u16* wp = reinterpret_cast<const u16*>(wr);
            for (u32 i = lane << 3; i < w.in; i += 256u) {
                uint4 h = ld_nc_u4(wp + i);
                float4 x0 = *reinterpret_cast<const float4*>(x + i);
                float4 x1 = *reinterpret_cast<const float4*>(x + i + 4u);
                acc = fmaf(__uint_as_float(h.x << 16), x0.x,
                           fmaf(__uint_as_float(h.x & 0xFFFF0000u), x0.y,
                                fmaf(__uint_as_float(h.y << 16), x0.z,
                                     fmaf(__uint_as_float(h.y & 0xFFFF0000u), x0.w, acc))));
                acc = fmaf(__uint_as_float(h.z << 16), x1.x,
                           fmaf(__uint_as_float(h.z & 0xFFFF0000u), x1.y,
                                fmaf(__uint_as_float(h.w << 16), x1.z,
                                     fmaf(__uint_as_float(h.w & 0xFFFF0000u), x1.w, acc))));
            }
        } else if (x_vec && (reinterpret_cast<u64>(wr) & align_mask) == 0ull) {
            for (u32 i = lane << 2; i < vec_end; i += 128u) {
                float4 wv = load_w4<IS_BF16>(wr, i);
                float4 xv = *reinterpret_cast<const float4*>(x + i);
                acc = fmaf(wv.x, xv.x, fmaf(wv.y, xv.y, fmaf(wv.z, xv.z, fmaf(wv.w, xv.w, acc))));
            }
            for (u32 i = vec_end + lane; i < w.in; i += 32u) acc = fmaf(load_w1<IS_BF16>(wr, i), x[i], acc);
        } else {
            for (u32 i = lane; i < w.in; i += 32u) acc = fmaf(load_w1<IS_BF16>(wr, i), x[i], acc);
        }
        float s = warp_sum(acc);
        if (lane == 0u) y[o] = s;
    }
}

template <bool IS_BF16>
static __device__ __noinline__ void gemv_dense_n(const mat& w, const float* x, u32 xs, float* y,
                                                 u32 ys, u32 rows, u32 out, u64 cursor) {
    constexpr u32 elem_bytes = IS_BF16 ? 2u : 4u;
    constexpr u64 align_mask = IS_BF16 ? 7ull : 15ull;
    u32 wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    u32 wstep = (gridDim.x * blockDim.x) >> 5;
    u32 lane = threadIdx.x & 31u;
    u32 chunks = (rows + 7u) >> 3;
    u32 vec_end = w.in & ~3u;
    bool x_vec = (reinterpret_cast<u64>(x) & 15u) == 0u && (xs & 3u) == 0u;
    for (u32 u = warp_start(cursor, wid, wstep); u < out * chunks; u += wstep) {
        u32 o = u % out, r0 = (u / out) * 8u, n = min(8u, rows - r0);
        const u8* wr = w.p + u64(o) * w.in * elem_bytes;
        bool use_vec = x_vec && (reinterpret_cast<u64>(wr) & align_mask) == 0ull && vec_end > 0u;
        float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        if (use_vec) {
            for (u32 i = lane << 2; i < vec_end; i += 128u) {
                float4 wv = load_w4<IS_BF16>(wr, i);
                for (u32 r = 0; r < n; r++) {
                    float4 xv = *reinterpret_cast<const float4*>(x + u64(r0 + r) * xs + i);
                    acc[r] = fmaf(wv.x, xv.x, acc[r]);
                    acc[r] = fmaf(wv.y, xv.y, acc[r]);
                    acc[r] = fmaf(wv.z, xv.z, acc[r]);
                    acc[r] = fmaf(wv.w, xv.w, acc[r]);
                }
            }
            for (u32 i = vec_end + lane; i < w.in; i += 32u) {
                float wv = load_w1<IS_BF16>(wr, i);
                for (u32 r = 0; r < n; r++) acc[r] = fmaf(wv, x[u64(r0 + r) * xs + i], acc[r]);
            }
        } else {
            for (u32 i = lane; i < w.in; i += 32u) {
                float wv = load_w1<IS_BF16>(wr, i);
                for (u32 r = 0; r < n; r++) acc[r] = fmaf(wv, x[u64(r0 + r) * xs + i], acc[r]);
            }
        }
        store_warp(acc, n, r0, o, y, ys, lane);
    }
}

// Two-row pipelined INT4_G32 GEMV for decode.
static __device__ __noinline__ void gemv_int4_2row(const ctx& c, const mat& w, const float* x,
                                                   float* y, u32 out, u16* y_bf16 = nullptr) {
    u32 wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    u32 wstep = (gridDim.x * blockDim.x) >> 5;
    u32 lane = threadIdx.x & 31u;
    u32 warp = threadIdx.x >> 5;
    u32 blocks = w.in >> 5;
    u32* x_h = c.sm.q32.x_h;
    stage_x_f16(x_h, x, w.in);
    u32* sm_u32 = c.sm.q32.buf[warp];
    u8* sm_u8 = reinterpret_cast<u8*>(sm_u32);
    u32 pairs = out >> 1;
    u32 chunks = blocks >> 5;
    uint4 cur0, cur1, cur2;
    float acc0 = 0.0f, acc1 = 0.0f;
    pipe_run(
        pairs, wid, wstep, chunks,
        [&](u32 pair_idx, u32 ch) {
            u64 rb0 = u64(pair_idx << 1) * blocks + (u64(ch) << 5);
            const u8* p0 = w.p + rb0 * 18ull;
            const u8* p1 = w.p + (rb0 + blocks) * 18ull;
            cur0 = ld_nc_u4(p0 + (lane << 4));
            cur1 = ld_nc_u4(p1 + (lane << 4));
            if (lane < 4u) cur2 = ld_nc_u4(p0 + 512u + (lane << 4));
            else if (lane < 8u) cur2 = ld_nc_u4(p1 + 512u + ((lane - 4u) << 4));
        },
        [&]() {
            *reinterpret_cast<uint4*>(sm_u32 + (lane << 2)) = cur0;
            *reinterpret_cast<uint4*>(sm_u32 + 144u + (lane << 2)) = cur1;
            if (lane < 4u) {
                *reinterpret_cast<uint4*>(sm_u32 + 128u + (lane << 2)) = cur2;
            } else if (lane < 8u) {
                *reinterpret_cast<uint4*>(sm_u32 + 272u + ((lane - 4u) << 2)) = cur2;
            }
        },
        [&](u32 ch) {
#pragma unroll
            for (u32 k = 0; k < 8u; k++) {
                u32 col = (ch << 10) + (k << 7) + (lane << 2);
                uint2 xh = *reinterpret_cast<const uint2*>(x_h + (col >> 1));
                u32 b0 = ((k << 2) + (lane >> 3)) * 18u;
                u32 b1 = 576u + b0;
                u32 q_add = 2u + ((lane & 7u) << 1);
                float sc0 = trellis::f16_to_f32(*reinterpret_cast<const u16*>(sm_u8 + b0));
                float sc1 = trellis::f16_to_f32(*reinterpret_cast<const u16*>(sm_u8 + b1));
                u32 q0 = *reinterpret_cast<const u16*>(sm_u8 + b0 + q_add);
                u32 q1 = *reinterpret_cast<const u16*>(sm_u8 + b1 + q_add);
                acc0 = fmaf(sc0, dot4_f16x2(int4x2_to_f16x2(q0 & 0xFFu), int4x2_to_f16x2(q0 >> 8), xh),
                            acc0);
                acc1 = fmaf(sc1, dot4_f16x2(int4x2_to_f16x2(q1 & 0xFFu), int4x2_to_f16x2(q1 >> 8), xh),
                            acc1);
            }
        },
        [&](u32 cur_p) {
            float s0 = warp_sum(acc0), s1 = warp_sum(acc1);
            if (lane == 0u) {
                if (y_bf16 != nullptr) {
                    u32 b0 = trellis::f32_to_bf16(s0), b1 = trellis::f32_to_bf16(s1);
                    *reinterpret_cast<u32*>(y_bf16 + (cur_p << 1)) = b0 | (b1 << 16);
                } else if ((reinterpret_cast<u64>(y) & 7ull) == 0ull) {
                    *reinterpret_cast<float2*>(y + (cur_p << 1)) = make_float2(s0, s1);
                } else {
                    y[cur_p << 1] = s0;
                    y[(cur_p << 1) + 1u] = s1;
                }
            }
            acc0 = 0.0f;
            acc1 = 0.0f;
        });
    __syncthreads();
}

template <bool IS_MXFP8>
static __device__ __noinline__ void gemv_q32_1(const ctx& c, const mat& w, const float* x, float* y,
                                               u32 out, u64 cursor) {
    constexpr u32 blk_bytes = IS_MXFP8 ? 33u : 18u;
    u32 wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    u32 wstep = (gridDim.x * blockDim.x) >> 5;
    u32 lane = threadIdx.x & 31u;
    u32 warp = threadIdx.x >> 5;
    u32 blocks = w.in >> 5;
    bool aligned = (w.in & 31u) == 0u;
    bool x_vec = (reinterpret_cast<u64>(x) & 15ull) == 0ull;
    u64 row_bytes = u64(blocks) * blk_bytes;
    bool row_v4 = aligned && x_vec && (reinterpret_cast<u64>(w.p) & 15ull) == 0ull
                  && (row_bytes & 15ull) == 0ull && w.in <= 6144u
                  && (IS_MXFP8 ? (blocks >= 16u && (blocks & 15u) == 0u)
                               : (blocks >= 32u && (blocks & 31u) == 0u));
    if (row_v4) {
        u32* x_h = c.sm.q32.x_h;
        stage_x_f16(x_h, x, w.in);
        u32* sm_u32 = c.sm.q32.buf[warp];
        u8* sm_u8 = reinterpret_cast<u8*>(sm_u32);
        constexpr u32 step = IS_MXFP8 ? 16u : 32u;
        constexpr u32 tail = IS_MXFP8 ? 1u : 4u;
        constexpr u32 k_end = IS_MXFP8 ? 4u : 8u;
        for (u32 o = warp_start(cursor, wid, wstep); o < out; o += wstep) {
            u64 row_blk = u64(o) * blocks;
            float acc = 0.0f;
            const u8* p = w.p + row_blk * blk_bytes;
            uint4 cur0 = ld_nc_u4(p + (lane << 4)), cur1;
            if (lane < tail) cur1 = ld_nc_u4(p + 512u + (lane << 4));
            for (u32 blk = 0u; blk < blocks; blk += step) {
                *reinterpret_cast<uint4*>(sm_u32 + (lane << 2)) = cur0;
                if (lane < tail) *reinterpret_cast<uint4*>(sm_u32 + 128u + (lane << 2)) = cur1;
                __syncwarp();
                if (blk + step < blocks) {
                    const u8* np = w.p + (row_blk + blk + step) * blk_bytes;
                    cur0 = ld_nc_u4(np + (lane << 4));
                    if (lane < tail) cur1 = ld_nc_u4(np + 512u + (lane << 4));
                }
#pragma unroll
                for (u32 k = 0; k < k_end; k++) {
                    u32 b_off = ((k << 2) + (lane >> 3)) * blk_bytes;
                    u32 col = (blk << 5) + (k << 7) + (lane << 2);
                    uint2 xh = *reinterpret_cast<const uint2*>(x_h + (col >> 1));
                    float sc;
                    u32 hw0, hw1;
                    if constexpr (IS_MXFP8) {
                        u32 eb = sm_u8[b_off];
                        sc = __uint_as_float(eb == 0u ? 0x00400000u : (eb << 23));
                        u32 qw = sm_u32_raw(sm_u32, b_off + 1u + ((lane & 7u) << 2));
                        hw0 = e4m3x2_to_f16x2(qw & 0xFFFFu);
                        hw1 = e4m3x2_to_f16x2(qw >> 16);
                    } else {
                        sc = trellis::f16_to_f32(*reinterpret_cast<const u16*>(sm_u8 + b_off));
                        u32 q16 = *reinterpret_cast<const u16*>(sm_u8 + b_off + 2u + ((lane & 7u) << 1));
                        hw0 = int4x2_to_f16x2(q16 & 0xFFu);
                        hw1 = int4x2_to_f16x2(q16 >> 8);
                    }
                    acc = fmaf(sc, dot4_f16x2(hw0, hw1, xh), acc);
                }
                __syncwarp();
            }
            float s = warp_sum(acc);
            if (lane == 0u) y[o] = s;
        }
        __syncthreads();
        return;
    }
    u32 nib_byte = 2u + (lane >> 1), nib_shift = (lane & 1u) << 2;
    for (u32 o = warp_start(cursor, wid, wstep); o < out; o += wstep) {
        float acc = 0.0f;
        if (aligned) {
            u64 row_blk = u64(o) * blocks;
            for (u32 blk = 0; blk < blocks; blk++) {
                const u8* b = w.p + (row_blk + blk) * blk_bytes;
                float sc = 0.0f;
                if (lane == 0u) {
                    sc = IS_MXFP8 ? mxfp8_scale(b[0]) : trellis::f16_to_f32(b[0] | (u32(b[1]) << 8));
                }
                sc = __shfl_sync(0xFFFFFFFFu, sc, 0);
                float q = IS_MXFP8 ? trellis::e4m3_to_f32(b[1u + lane])
                                   : (float((b[nib_byte] >> nib_shift) & 15u) - 8.0f);
                acc = fmaf(sc * q, x[(blk << 5) + lane], acc);
            }
        } else {
            for (u32 i = lane; i < w.in; i += 32u) acc = fmaf(elem(w, 0u, o, i), x[i], acc);
        }
        float s = warp_sum(acc);
        if (lane == 0u) y[o] = s;
    }
}

// Fused RMSNorm and multi-output GEMV for decode.
static __device__ __noinline__ void gemv_mxfp8_norm(const ctx& c, const mat& nw, const mat& w0,
                                                    const mat& w1, const mat& w2, float* y0,
                                                    u32 out0, float* y1, u32 out1, float* y2,
                                                    u32 out2, bool save_h) {
    u32 n = c.s.hidden;
    u32* x_h = c.sm.q32.x_h;
    float inv = rms_inv(c, c.p.x, n);
    bool nw_bf16 = nw.klass == K_BF16 && (reinterpret_cast<u64>(nw.p) & 7ull) == 0ull;
    bool write_h = save_h && blockIdx.x == 0u;
    for (u32 i = threadIdx.x << 2; i < n; i += blockDim.x << 2) {
        float4 v = *reinterpret_cast<const float4*>(c.p.x + i);
        float4 wv = norm_w4(nw, nw_bf16, i);
        float4 out = make_float4(v.x * (inv * wv.x), v.y * (inv * wv.y), v.z * (inv * wv.z),
                                 v.w * (inv * wv.w));
        if (write_h) *reinterpret_cast<float4*>(c.p.h + i) = out;
        u32 p0 = trellis::f32x2_to_f16x2(out.x, out.y);
        u32 p1 = trellis::f32x2_to_f16x2(out.z, out.w);
        *reinterpret_cast<uint2*>(x_h + (i >> 1)) = make_uint2(p0, p1);
    }
    __syncthreads();
    u32 wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    u32 wstep = (gridDim.x * blockDim.x) >> 5;
    u32 lane = threadIdx.x & 31u;
    u32 warp = threadIdx.x >> 5;
    u32 blocks = n >> 5;
    u32* sm_u32 = c.sm.q32.buf[warp];
    u8* sm_u8 = reinterpret_cast<u8*>(sm_u32);
    bool wk_mxfp8 = out2 > 0u && w2.klass == K_MXFP8;
    u32 cut1 = out0 + out1;
    u32 total = cut1 + (wk_mxfp8 ? out2 : 0u);
    u32 chunks = blocks >> 5;
    uint4 cur0, cur1, cur2;
    float acc = 0.0f;
    pipe_run(
        total, wid, wstep, chunks,
        [&](u32 u, u32 ch) {
            const u8* wp = u < out0 ? w0.p : (u < cut1 ? w1.p : w2.p);
            u32 o = u < out0 ? u : (u < cut1 ? (u - out0) : (u - cut1));
            const u8* p = wp + (u64(o) * blocks + (u64(ch) << 5)) * 33ull;
            cur0 = ld_nc_u4(p + (lane << 4));
            cur1 = ld_nc_u4(p + 512u + (lane << 4));
            if (lane < 2u) cur2 = ld_nc_u4(p + 1024u + (lane << 4));
        },
        [&]() {
            *reinterpret_cast<uint4*>(sm_u32 + (lane << 2)) = cur0;
            *reinterpret_cast<uint4*>(sm_u32 + 128u + (lane << 2)) = cur1;
            if (lane < 2u) *reinterpret_cast<uint4*>(sm_u32 + 256u + (lane << 2)) = cur2;
        },
        [&](u32 ch) {
#pragma unroll
            for (u32 k = 0; k < 8u; k++) {
                u32 b_off = ((k << 2) + (lane >> 3)) * 33u;
                u32 eb = sm_u8[b_off];
                float sc = __uint_as_float(eb == 0u ? 0x00400000u : (eb << 23));
                u32 qw = sm_u32_raw(sm_u32, b_off + 1u + ((lane & 7u) << 2));
                u32 col = (ch << 10) + (k << 7) + (lane << 2);
                uint2 xh = *reinterpret_cast<const uint2*>(x_h + (col >> 1));
                acc = fmaf(sc,
                           dot4_f16x2(e4m3x2_to_f16x2(qw & 0xFFFFu), e4m3x2_to_f16x2(qw >> 16), xh),
                           acc);
            }
        },
        [&](u32 cur_u) {
            float sum = warp_sum(acc);
            if (lane == 0u) {
                if (cur_u < out0) y0[cur_u] = sum;
                else if (cur_u < cut1) y1[cur_u - out0] = sum;
                else y2[cur_u - cut1] = sum;
            }
            acc = 0.0f;
        });
    if (out2 > 0u && !wk_mxfp8) {
        bool wk_bf16 = w2.klass == K_BF16;
        for (u32 o = warp_start(cut1, wid, wstep); o < out2; o += wstep) {
            float acc = 0.0f;
            if (wk_bf16) {
                const u16* wr = reinterpret_cast<const u16*>(w2.p) + u64(o) * n;
                for (u32 i = lane << 3; i < n; i += 256u) {
                    uint4 h = ld_nc_u4(wr + i);
                    uint4 xq = *reinterpret_cast<const uint4*>(x_h + (i >> 1));
                    acc = fmaf(__uint_as_float(h.x << 16), trellis::f16_to_f32(xq.x & 0xFFFFu), acc);
                    acc = fmaf(__uint_as_float(h.x & 0xFFFF0000u), trellis::f16_to_f32(xq.x >> 16), acc);
                    acc = fmaf(__uint_as_float(h.y << 16), trellis::f16_to_f32(xq.y & 0xFFFFu), acc);
                    acc = fmaf(__uint_as_float(h.y & 0xFFFF0000u), trellis::f16_to_f32(xq.y >> 16), acc);
                    acc = fmaf(__uint_as_float(h.z << 16), trellis::f16_to_f32(xq.z & 0xFFFFu), acc);
                    acc = fmaf(__uint_as_float(h.z & 0xFFFF0000u), trellis::f16_to_f32(xq.z >> 16), acc);
                    acc = fmaf(__uint_as_float(h.w << 16), trellis::f16_to_f32(xq.w & 0xFFFFu), acc);
                    acc = fmaf(__uint_as_float(h.w & 0xFFFF0000u), trellis::f16_to_f32(xq.w >> 16), acc);
                }
            } else {
                const float* wr = reinterpret_cast<const float*>(w2.p) + u64(o) * n;
                for (u32 i = lane << 2; i < n; i += 128u) {
                    float4 wv = *reinterpret_cast<const float4*>(wr + i);
                    uint2 xq = *reinterpret_cast<const uint2*>(x_h + (i >> 1));
                    acc = fmaf(wv.x, trellis::f16_to_f32(xq.x & 0xFFFFu), acc);
                    acc = fmaf(wv.y, trellis::f16_to_f32(xq.x >> 16), acc);
                    acc = fmaf(wv.z, trellis::f16_to_f32(xq.y & 0xFFFFu), acc);
                    acc = fmaf(wv.w, trellis::f16_to_f32(xq.y >> 16), acc);
                }
            }
            float sum = warp_sum(acc);
            if (lane == 0u) y2[o] = sum;
        }
    }
    __syncthreads();
}

template <bool IS_MXFP8>
static __device__ __noinline__ void gemv_q32_n(const mat& w, const float* x, u32 xs, float* y,
                                               u32 ys, u32 rows, u32 out, u64 cursor) {
    constexpr u32 blk_bytes = IS_MXFP8 ? 33u : 18u;
    u32 wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    u32 wstep = (gridDim.x * blockDim.x) >> 5;
    u32 lane = threadIdx.x & 31u;
    u32 chunks = (rows + 7u) >> 3;
    u32 blocks = w.in >> 5;
    bool aligned = (w.in & 31u) == 0u;
    u32 nib_byte = 2u + (lane >> 1), nib_shift = (lane & 1u) << 2;
    for (u32 u = warp_start(cursor, wid, wstep); u < out * chunks; u += wstep) {
        u32 o = u % out, r0 = (u / out) * 8u, n = min(8u, rows - r0);
        float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        if (aligned) {
            u64 row_blk = u64(o) * blocks;
            for (u32 blk = 0; blk < blocks; blk++) {
                const u8* b = w.p + (row_blk + blk) * blk_bytes;
                float sc = 0.0f;
                if (lane == 0u) {
                    sc = IS_MXFP8 ? mxfp8_scale(b[0]) : trellis::f16_to_f32(b[0] | (u32(b[1]) << 8));
                }
                sc = __shfl_sync(0xFFFFFFFFu, sc, 0);
                float q = IS_MXFP8 ? trellis::e4m3_to_f32(b[1u + lane])
                                   : (float((b[nib_byte] >> nib_shift) & 15u) - 8.0f);
                float wv = sc * q;
                u32 col = (blk << 5) + lane;
                for (u32 r = 0; r < n; r++) acc[r] = fmaf(wv, x[u64(r0 + r) * xs + col], acc[r]);
            }
        } else {
            for (u32 i = lane; i < w.in; i += 32u) {
                float wv = elem(w, 0u, o, i);
                for (u32 r = 0; r < n; r++) acc[r] = fmaf(wv, x[u64(r0 + r) * xs + i], acc[r]);
            }
        }
        store_warp(acc, n, r0, o, y, ys, lane);
    }
}

static __device__ __noinline__ void gemv_any(const mat& w, const float* x, u32 xs, float* y, u32 ys,
                                             u32 rows, u32 out, u64 cursor) {
    u32 wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    u32 wstep = (gridDim.x * blockDim.x) >> 5;
    u32 lane = threadIdx.x & 31u;
    u32 chunks = (rows + 7u) >> 3;
    for (u32 u = warp_start(cursor, wid, wstep); u < out * chunks; u += wstep) {
        u32 o = u % out, r0 = (u / out) * 8u, n = min(8u, rows - r0);
        float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (u32 i = lane; i < w.in; i += 32u) {
            float wv = elem(w, 0u, o, i);
            for (u32 r = 0; r < n; r++) acc[r] = fmaf(wv, x[u64(r0 + r) * xs + i], acc[r]);
        }
        store_warp(acc, n, r0, o, y, ys, lane);
    }
}

static __device__ void times_plain(const ctx& c, const mat& w, const float* x, u32 xs, float* y,
                                   u32 ys, u32 rows, u32 out, u64 cursor) {
    if (rows == 1u) {
        switch (w.klass) {
        case K_F32:
            gemv_dense_1<false>(w, x, y, out, cursor);
            return;
        case K_BF16:
            gemv_dense_1<true>(w, x, y, out, cursor);
            return;
        case K_INT4:
            if (cursor == 0ull && (w.in & 1023u) == 0u && w.in <= 6144u && (out & 1u) == 0u
                && (reinterpret_cast<u64>(w.p) & 15ull) == 0ull
                && (reinterpret_cast<u64>(x) & 15ull) == 0ull) {
                gemv_int4_2row(c, w, x, y, out);
            } else {
                gemv_q32_1<false>(c, w, x, y, out, cursor);
            }
            return;
        case K_MXFP8:
            gemv_q32_1<true>(c, w, x, y, out, cursor);
            return;
        default:
            gemv_any(w, x, xs, y, ys, rows, out, cursor);
            return;
        }
    }
    switch (w.klass) {
    case K_F32:
        gemv_dense_n<false>(w, x, xs, y, ys, rows, out, cursor);
        return;
    case K_BF16:
        gemv_dense_n<true>(w, x, xs, y, ys, rows, out, cursor);
        return;
    case K_INT4:
        gemv_q32_n<false>(w, x, xs, y, ys, rows, out, cursor);
        return;
    case K_MXFP8:
        gemv_q32_n<true>(w, x, xs, y, ys, rows, out, cursor);
        return;
    default:
        gemv_any(w, x, xs, y, ys, rows, out, cursor);
        return;
    }
}

// Computes y = x w^T using trellis MMA or warp GEMV.
static __device__ bool linear(const ctx& c, const mat& w, const float* x, u32 xs, float* y, u32 ys,
                              u32 rows, u32 out) {
    bool fast_t = (w.in & 127u) == 0u && (w.out & 127u) == 0u && out == w.out;
    if (w.klass == K_TURNED) {
        PHASE(turn_rows(x, xs, w.suh, nullptr, c.p.t1, w.in, rows, w.in));
        if (fast_t) {
            PHASE(trellis_run(c, w, c.p.t1, w.in, y, ys, rows, true));
            return true;
        }
        PHASE(gemv_any(w, c.p.t1, w.in, c.p.t2, w.out, rows, w.out, 0ull));
        PHASE(turn_rows(c.p.t2, w.out, nullptr, w.svh, y, ys, rows, w.out));
        return true;
    }
    if (w.klass == K_TRELLIS && fast_t) {
        PHASE(trellis_run(c, w, x, xs, y, ys, rows, false));
        return true;
    }
    PHASE(times_plain(c, w, x, xs, y, ys, rows, out, 0ull));
    return true;
}

// Dual projections sharing x, fused when both are plain GEMV.
static __device__ bool linear_pair(const ctx& c, const mat& w0, const mat& w1, const float* x,
                                   u32 xs, float* y0, u32 ys0, u32 out0, float* y1, u32 ys1,
                                   u32 out1, u32 rows) {
    if (w0.klass <= K_MXFP8 && w1.klass <= K_MXFP8) {
        u64 n0 = u64(out0) * ((rows + 7u) >> 3);
        PHASE(times_plain(c, w0, x, xs, y0, ys0, rows, out0, 0ull);
              times_plain(c, w1, x, xs, y1, ys1, rows, out1, n0));
        return true;
    }
    return linear(c, w0, x, xs, y0, ys0, rows, out0)
           && linear(c, w1, x, xs, y1, ys1, rows, out1);
}

// Block-parallel RMS norm with optional mean centering.
static __device__ void rms(const ctx& c, const float* x, u32 xs, const mat& w, float* y, u32 ys,
                           u32 rows, u32 n, const u32* from, bool centered) {
    bool vec_n = (n & 3u) == 0u;
    for (u32 r = blockIdx.x; r < rows; r += gridDim.x) {
        const float* row = x + u64(from ? from[r] : r) * xs;
        float* dst = y + u64(r) * ys;
        bool r_vec = vec_n && (reinterpret_cast<u64>(row) & 15u) == 0u;
        bool d_vec = vec_n && (reinterpret_cast<u64>(dst) & 15u) == 0u;
        float mean = 0.0f;
        if (centered) {
            float sum = 0.0f;
            if (r_vec) {
                for (u32 i = threadIdx.x << 2; i < n; i += blockDim.x << 2) {
                    float4 v = *reinterpret_cast<const float4*>(row + i);
                    sum += v.x + v.y + v.z + v.w;
                }
            } else {
                for (u32 i = threadIdx.x; i < n; i += blockDim.x) sum += row[i];
            }
            mean = block_sum(c, sum) / float(n);
        }
        float sq = 0.0f;
        if (r_vec) {
            for (u32 i = threadIdx.x << 2; i < n; i += blockDim.x << 2) {
                float4 v = *reinterpret_cast<const float4*>(row + i);
                float a0 = v.x - mean, a1 = v.y - mean, a2 = v.z - mean, a3 = v.w - mean;
                sq = fmaf(a0, a0, fmaf(a1, a1, fmaf(a2, a2, fmaf(a3, a3, sq))));
            }
        } else {
            for (u32 i = threadIdx.x; i < n; i += blockDim.x) {
                float d = row[i] - mean;
                sq = fmaf(d, d, sq);
            }
        }
        float inv = 1.0f / sqrtf(block_sum(c, sq) / float(n) + c.s.eps);
        if (r_vec && d_vec) {
            bool w_bf16 = w.klass == K_BF16 && (reinterpret_cast<u64>(w.p) & 7ull) == 0ull;
            for (u32 i = threadIdx.x << 2; i < n; i += blockDim.x << 2) {
                float4 v = *reinterpret_cast<const float4*>(row + i);
                float4 wv = norm_w4(w, w_bf16, i);
                *reinterpret_cast<float4*>(dst + i) =
                    make_float4((v.x - mean) * (inv * wv.x), (v.y - mean) * (inv * wv.y),
                                (v.z - mean) * (inv * wv.z), (v.w - mean) * (inv * wv.w));
            }
        } else {
            for (u32 i = threadIdx.x; i < n; i += blockDim.x) {
                dst[i] = (row[i] - mean) * (inv * vec_elem(w, 0u, i));
            }
        }
    }
}

static __device__ void gated(const float* a, const float* b, float* y, u32 n) {
    line l = threads();
    bool vec = (reinterpret_cast<u64>(a) & 15u) == 0u && (reinterpret_cast<u64>(b) & 15u) == 0u
               && (reinterpret_cast<u64>(y) & 15u) == 0u;
    u32 vec_end = vec ? (n & ~3u) : 0u;
    for (u32 i = l.at << 2; i < vec_end; i += l.step << 2) {
        float4 va = *reinterpret_cast<const float4*>(a + i);
        float4 vb = *reinterpret_cast<const float4*>(b + i);
        *reinterpret_cast<float4*>(y + i) = make_float4(
            va.x / (1.0f + __expf(-va.x)) * vb.x, va.y / (1.0f + __expf(-va.y)) * vb.y,
            va.z / (1.0f + __expf(-va.z)) * vb.z, va.w / (1.0f + __expf(-va.w)) * vb.w);
    }
    for (u32 i = vec_end + l.at; i < n; i += l.step) y[i] = a[i] / (1.0f + __expf(-a[i])) * b[i];
}

static __device__ __forceinline__ float4 load_part4(const ctx& c, u32 moe_cnt, u32 idx) {
    if (moe_cnt == 0xFFFFFFFFu) return *reinterpret_cast<const float4*>(c.p.part + idx);
    float4 acc = *reinterpret_cast<const float4*>(c.p.t2 + idx);
    for (u32 s = 0; s < moe_cnt; s++) {
        float4 v = *reinterpret_cast<const float4*>(c.p.d + u64(s) * c.s.hidden + idx);
        acc.x += v.x;
        acc.y += v.y;
        acc.z += v.z;
        acc.w += v.w;
    }
    return acc;
}

static __device__ __forceinline__ float load_part1(const ctx& c, u32 moe_cnt, u32 idx) {
    if (moe_cnt == 0xFFFFFFFFu) return c.p.part[idx];
    float acc = c.p.t2[idx];
    for (u32 s = 0; s < moe_cnt; s++) acc += c.p.d[u64(s) * c.s.hidden + idx];
    return acc;
}

static __device__ bool add_finish(const ctx& c, ticket& r, u32 done, u32 len, bool wrote) {
    line l = threads();
    bool x_vec = (len & 3u) == 0u && (reinterpret_cast<u64>(c.p.x + done) & 15ull) == 0ull;
    bool will_read = x_vec ? ((blockIdx.x * blockDim.x) << 2) < len
                           : (blockIdx.x * blockDim.x) < len;
    if (!trigger(r, len, wrote) || !wait(r, will_read)) return false;
    if (!will_read) return true;
    parts got = received(r);
    if (x_vec) {
        for (u32 i = l.at << 2; i < len; i += l.step << 2) {
            float4 vx = *reinterpret_cast<const float4*>(c.p.x + done + i);
            float4 vs = got.sum4(i);
            *reinterpret_cast<float4*>(c.p.x + done + i) =
                make_float4(vx.x + vs.x, vx.y + vs.y, vx.z + vs.z, vx.w + vs.w);
        }
    } else {
        for (u32 i = l.at; i < len; i += l.step) c.p.x[done + i] += got.sum(i);
    }
    return true;
}

// Residual add, with cross-rank all-reduce when split.
static __device__ bool add(const ctx& c, bool ranks, u32 moe_cnt = 0xFFFFFFFFu,
                           ticket* pre_r = nullptr) {
    line l = threads();
    u32 n = c.s.rows * c.s.hidden;
    if (!ranks) {
        bool vec = (reinterpret_cast<u64>(c.p.x) & 15u) == 0u
                   && (moe_cnt != 0xFFFFFFFFu
                           ? ((reinterpret_cast<u64>(c.p.t2) & 15u) == 0u
                              && (reinterpret_cast<u64>(c.p.d) & 15u) == 0u
                              && (c.s.hidden & 3u) == 0u)
                           : (reinterpret_cast<u64>(c.p.part) & 15u) == 0u);
        u32 vec_end = vec ? (n & ~3u) : 0u;
        for (u32 i = l.at << 2; i < vec_end; i += l.step << 2) {
            float4 vx = *reinterpret_cast<const float4*>(c.p.x + i);
            float4 vp = load_part4(c, moe_cnt, i);
            *reinterpret_cast<float4*>(c.p.x + i) =
                make_float4(vx.x + vp.x, vx.y + vp.y, vx.z + vp.z, vx.w + vp.w);
        }
        for (u32 i = vec_end + l.at; i < n; i += l.step) c.p.x[i] += load_part1(c, moe_cnt, i);
        return sync(c.t);
    }
    const lane* ln = get<const lane>(c.t, c.t->lanes);
    bool bf16 = ln->elem == BF16;
    u32 room = detail::pools(c.t)->pools[ln->pool].send_bytes >> (bf16 ? 1u : 2u);
    for (u32 done = 0; done < n; done += room) {
        u32 len = min(room, n - done);
        ticket r = (done == 0u && pre_r != nullptr) ? *pre_r : prepare(c.se, 0u);
        bool p_vec = (len & 3u) == 0u
                     && (moe_cnt != 0xFFFFFFFFu
                             ? ((reinterpret_cast<u64>(c.p.t2) & 15u) == 0u
                                && (reinterpret_cast<u64>(c.p.d) & 15u) == 0u
                                && (c.s.hidden & 3u) == 0u)
                             : (reinterpret_cast<u64>(c.p.part + done) & 15ull) == 0ull);
        bool wrote = false;
        if (bf16) {
            u16* sent = send_buffer<u16>(r);
            if (sent == nullptr) return false;
            if (p_vec) {
                for (u32 i = l.at << 2; i < len; i += l.step << 2) {
                    float4 vp = load_part4(c, moe_cnt, done + i);
                    u32 p0 = u32(trellis::f32_to_bf16(vp.x))
                             | (u32(trellis::f32_to_bf16(vp.y)) << 16);
                    u32 p1 = u32(trellis::f32_to_bf16(vp.z))
                             | (u32(trellis::f32_to_bf16(vp.w)) << 16);
                    *reinterpret_cast<uint2*>(sent + i) = make_uint2(p0, p1);
                    wrote = true;
                }
            } else {
                for (u32 i = l.at; i < len; i += l.step) {
                    sent[i] = trellis::f32_to_bf16(load_part1(c, moe_cnt, done + i));
                    wrote = true;
                }
            }
        } else {
            float* sent = send_buffer<float>(r);
            if (sent == nullptr) return false;
            if (p_vec) {
                for (u32 i = l.at << 2; i < len; i += l.step << 2) {
                    *reinterpret_cast<float4*>(sent + i) = load_part4(c, moe_cnt, done + i);
                    wrote = true;
                }
            } else {
                for (u32 i = l.at; i < len; i += l.step) {
                    sent[i] = load_part1(c, moe_cnt, done + i);
                    wrote = true;
                }
            }
        }
        if (!add_finish(c, r, done, len, wrote)) return false;
    }
    return sync(c.t);
}
