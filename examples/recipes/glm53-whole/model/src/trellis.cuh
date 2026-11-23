// Fused trellis GEMM using shared trellis primitives.

static __device__ __forceinline__ void zero_acc(float acc[8][4]) {
#pragma unroll
    for (u32 mt = 0; mt < 8; mt++) {
        acc[mt][0] = 0.0f;
        acc[mt][1] = 0.0f;
        acc[mt][2] = 0.0f;
        acc[mt][3] = 0.0f;
    }
}

// Stores tiles, prefetches next if present, and runs MMA.
template <u32 K, u32 BOOK>
static __device__ __forceinline__ void trellis_step(u32 reg[8], const u8* next, u32 lane,
                                                    bool wide, u32* wtiles, const float* act,
                                                    u32 n, u32 ks, const trellis::lane_map& pos,
                                                    float acc[8][4]) {
    __syncwarp();
    trellis::tile_store<K>(reg, lane, wide, wtiles);
    __syncwarp();
    if (next != nullptr) trellis::tile_load<K>(next, lane, wide, reg);
    u32 fb[2];
    trellis::act_frag(act, n, ks, lane, fb);
#pragma unroll
    for (u32 mt = 0; mt < 8; mt++) {
        u32 fa[4];
        trellis::decode_frag<K, BOOK>(wtiles + mt * 8u * K, pos, fa);
        trellis::mma_m16n8k16(acc[mt], fa, fb);
    }
}

template <u32 K>
static __device__ __forceinline__ const u8* next_tile(const mat& w, u32 m, u32 ob, u32 ib0,
                                                      u32 warp, u32 st, u32 steps,
                                                      bool input_side) {
    if (st + 1u >= steps) return nullptr;
    u32 nti = ((ib0 + warp + ((st + 1u) >> 3) * WARPS) << 3) + ((st + 1u) & 7u);
    return trellis::tile_ptr<K>(w.p, w.in, w.out, m, nti, ob, input_side);
}

static __device__ __forceinline__ void sum_warp_buf(const float buf[WARPS][ROW_TILE][128], u32 row,
                                                    u32 lane, float v[4]) {
    v[0] = v[1] = v[2] = v[3] = 0.0f;
    for (u32 widx = 0; widx < WARPS; widx++) {
        float4 p = *reinterpret_cast<const float4*>(&buf[widx][row][lane << 2]);
        v[0] += p.x;
        v[1] += p.y;
        v[2] += p.z;
        v[3] += p.w;
    }
}

template <u32 K, u32 BOOK>
static __device__ __forceinline__ void trellis_loop(const mat& w, u32 m, u32 ob, u32 ib0, u32 steps,
                                                    const float* x, u32 xs, u32 r0, u32 n,
                                                    bool x_vec, u32 lane, u32 warp, bool wide,
                                                    bool input_side, u32* wtiles, float* wbuf,
                                                    const trellis::lane_map& pos,
                                                    float acc[8][4]) {
    u32 reg[8];
    trellis::tile_load<K>(
        trellis::tile_ptr<K>(w.p, w.in, w.out, m, (ib0 + warp) << 3, ob, input_side), lane, wide,
        reg);
    for (u32 st = 0; st < steps; st++) {
        u32 ib = ib0 + warp + (st >> 3) * WARPS;
        u32 ks = st & 7u;
        if (ks == 0u) {
            __syncwarp();
            for (u32 r = 0; r < n; r++) {
                float v[4];
                load_vec4(x + u64(r0 + r) * xs + (ib << 7) + (lane << 2), x_vec, v);
                *reinterpret_cast<float4*>(wbuf + (r << 7) + (lane << 2)) =
                    make_float4(v[0], v[1], v[2], v[3]);
            }
            __syncwarp();
        }
        const u8* next = next_tile<K>(w, m, ob, ib0, warp, st, steps, input_side);
        trellis_step<K, BOOK>(reg, next, lane, wide, wtiles, wbuf, n, ks, pos, acc);
    }
}

template <u32 K, u32 BOOK>
static __device__ __noinline__ void trellis_gemm(const ctx& c, const mat& w, const float* x,
                                                 u32 xs, float* y, u32 ys, u32 rows,
                                                 bool fuse_out) {
    u32 warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    trellis::lane_map pos = trellis::map_lane<K, BOOK>(lane);
    u32 row_groups = (rows + ROW_TILE - 1u) / ROW_TILE;
    u32 out_blocks = w.out >> 7;
    u32 in_blocks = w.in >> 7;
    u32 total = row_groups * out_blocks;
    bool wide = (reinterpret_cast<u64>(w.p + trellis::HEAD) & 15u) == 0u;
    bool input_side = word(w.p, 8) == trellis::INPUT_SIDE;
    bool x_vec = (reinterpret_cast<u64>(x) & 15u) == 0u && (xs & 3u) == 0u;
    bool y_vec = (reinterpret_cast<u64>(y) & 15u) == 0u && (ys & 3u) == 0u;
    bool sv_vec = fuse_out && w.svh != nullptr && (reinterpret_cast<u64>(w.svh) & 15u) == 0u;
    u32 my_ibs = in_blocks > warp ? ((in_blocks - 1u - warp) / WARPS + 1u) : 0u;
    u32 steps = my_ibs << 3;
    float* wbuf = &c.sm.lin.buf[warp][0][0];
    u32* wtiles = c.sm.lin.tiles[warp];

    for (u32 job = blockIdx.x; job < total; job += gridDim.x) {
        u32 rg = job / out_blocks, ob = job % out_blocks;
        u32 r0 = rg * ROW_TILE, n = min(ROW_TILE, rows - r0);
        float acc[8][4];
        zero_acc(acc);
        if (steps > 0u) {
            trellis_loop<K, BOOK>(w, 0u, ob, 0u, steps, x, xs, r0, n, x_vec, lane, warp, wide,
                                  input_side, wtiles, wbuf, pos, acc);
        }
        __syncwarp();
        trellis::store_acc(acc, n, lane, wbuf);
        __syncthreads();
        if (warp < n) {
            u32 base = (ob << 7) + (lane << 2);
            float v[4];
            sum_warp_buf(c.sm.lin.buf, warp, lane, v);
            if (fuse_out) {
                trellis::wht128(v, trellis::INV_SQRT128);
                if (w.svh != nullptr) {
                    float sc[4];
                    load_vec4(w.svh + base, sv_vec, sc);
                    for (u32 k = 0; k < 4; k++) v[k] *= sc[k];
                }
            }
            store_vec4(y + u64(r0 + warp) * ys + base, y_vec, v);
        }
        __syncthreads();
    }
}

template <u32 BOOK>
static __device__ void trellis_book(const ctx& c, const mat& w, const float* x, u32 xs, float* y,
                                    u32 ys, u32 rows, bool fuse_out) {
    if (w.k == 2u) {
        trellis_gemm<2u, BOOK>(c, w, x, xs, y, ys, rows, fuse_out);
    } else if (w.k == 3u) {
        trellis_gemm<3u, BOOK>(c, w, x, xs, y, ys, rows, fuse_out);
    } else {
        trellis_gemm<4u, BOOK>(c, w, x, xs, y, ys, rows, fuse_out);
    }
}

static __device__ void trellis_run(const ctx& c, const mat& w, const float* x, u32 xs, float* y,
                                   u32 ys, u32 rows, bool fuse_out) {
    if (w.book == trellis::BOOK_PMCGX) {
        trellis_book<trellis::BOOK_PMCGX>(c, w, x, xs, y, ys, rows, fuse_out);
    } else if (w.book == trellis::BOOK_MUL1) {
        trellis_book<trellis::BOOK_MUL1>(c, w, x, xs, y, ys, rows, fuse_out);
    } else {
        trellis_book<trellis::BOOK_MCG>(c, w, x, xs, y, ys, rows, fuse_out);
    }
}

// Decode MoE trellis MMA with optional WHT and output scaling.
template <u32 K, u32 BOOK, bool POST>
static __device__ __noinline__ void moe_vec_mma(const ctx& c, const mat& w, const float* x_base,
                                                const mat& dsv, const float* out_signs, u32 m,
                                                u32 e, float gate_w, u32 ob, u32 ib0, u32 ib_cnt,
                                                bool is_sh, float* dst) {
    u32 warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    trellis::lane_map pos = trellis::map_lane<K, BOOK>(lane);
    bool wide = (reinterpret_cast<u64>(w.p + trellis::HEAD) & 15u) == 0u;
    bool input_side = word(w.p, 8) == trellis::INPUT_SIDE;
    u32* wtiles = c.sm.moe.tiles[warp];
    float* wbuf = &c.sm.moe.buf[warp][0][0];
    float acc[8][4];
    zero_acc(acc);
    if constexpr (POST) {
        if (ib_cnt < WARPS) {
            if (warp < ib_cnt) {
                *reinterpret_cast<float4*>(wbuf + (lane << 2)) =
                    *reinterpret_cast<const float4*>(x_base + u64(ib0 + warp) * 128u + (lane << 2));
            }
            __syncthreads();
            u32 tot_ti = ib_cnt << 3;
            u32 my_tis = tot_ti > warp ? ((tot_ti - 1u - warp) / WARPS + 1u) : 0u;
            if (my_tis > 0u) {
                u32 reg[8];
                trellis::tile_load<K>(
                    trellis::tile_ptr<K>(w.p, w.in, w.out, m, (ib0 << 3) + warp, ob, input_side),
                    lane, wide, reg);
                for (u32 st = 0; st < my_tis; st++) {
                    u32 ti = warp + st * WARPS;
                    const u8* next = nullptr;
                    if (st + 1u < my_tis) {
                        next = trellis::tile_ptr<K>(w.p, w.in, w.out, m,
                                                    (ib0 << 3) + warp + (st + 1u) * WARPS, ob,
                                                    input_side);
                    }
                    trellis_step<K, BOOK>(reg, next, lane, wide, wtiles,
                                          &c.sm.moe.buf[ti >> 3][0][0], 1u, ti & 7u, pos, acc);
                }
            }
            __syncthreads();
        } else {
            u32 my_ibs = (ib_cnt - 1u - warp) / WARPS + 1u;
            trellis_loop<K, BOOK>(w, m, ob, ib0, my_ibs << 3, x_base, 0u, 0u, 1u, true, lane, warp,
                                  wide, input_side, wtiles, wbuf, pos, acc);
        }
    } else {
        u32 my_ibs = ib_cnt > warp ? ((ib_cnt - 1u - warp) / WARPS + 1u) : 0u;
        if (my_ibs > 0u) {
            trellis_loop<K, BOOK>(w, m, ob, ib0, my_ibs << 3, x_base, 0u, 0u, 1u, true, lane, warp,
                                  wide, input_side, wtiles, wbuf, pos, acc);
        }
    }
    __syncwarp();
    trellis::store_acc(acc, 1u, lane, wbuf);
    __syncthreads();
    if (warp == 0u) {
        float v[4];
        sum_warp_buf(c.sm.moe.buf, 0u, lane, v);
        if constexpr (POST) {
            trellis::wht128(v, trellis::INV_SQRT128);
            u32 base = (ob << 7) + (lane << 2);
            if (is_sh) {
                float sc[4];
                load_vec4(w.svh + base, (reinterpret_cast<u64>(w.svh) & 15u) == 0u, sc);
                for (u32 q = 0; q < 4u; q++) v[q] *= sc[q];
            } else {
                float sg[4], sv4[4];
                load_vec4(out_signs + base, (reinterpret_cast<u64>(out_signs) & 15u) == 0u, sg);
                vec_elem4(dsv, e, base, sv4);
                for (u32 q = 0; q < 4u; q++) v[q] = gate_w * (v[q] * sv4[q] * sg[q]);
            }
            *reinterpret_cast<float4*>(dst + base) = make_float4(v[0], v[1], v[2], v[3]);
        } else {
            *reinterpret_cast<float4*>(dst + (lane << 2)) = make_float4(v[0], v[1], v[2], v[3]);
        }
    }
    __syncthreads();
}

template <u32 BOOK, bool POST>
static __device__ void moe_vec_book(const ctx& c, const mat& w, const float* x_base,
                                    const mat& dsv, const float* out_signs, u32 m, u32 e,
                                    float gate_w, u32 ob, u32 ib0, u32 ib_cnt, bool is_sh,
                                    float* dst) {
    if (w.k == 2u) {
        moe_vec_mma<2u, BOOK, POST>(c, w, x_base, dsv, out_signs, m, e, gate_w, ob, ib0, ib_cnt,
                                    is_sh, dst);
    } else if (w.k == 3u) {
        moe_vec_mma<3u, BOOK, POST>(c, w, x_base, dsv, out_signs, m, e, gate_w, ob, ib0, ib_cnt,
                                    is_sh, dst);
    } else {
        moe_vec_mma<4u, BOOK, POST>(c, w, x_base, dsv, out_signs, m, e, gate_w, ob, ib0, ib_cnt,
                                    is_sh, dst);
    }
}

template <bool POST>
static __device__ void moe_vec_run(const ctx& c, const mat& w, const float* x_base, const mat& dsv,
                                   const float* out_signs, u32 m, u32 e, float gate_w, u32 ob,
                                   u32 ib0, u32 ib_cnt, bool is_sh, float* dst) {
    if (w.book == trellis::BOOK_PMCGX) {
        moe_vec_book<trellis::BOOK_PMCGX, POST>(c, w, x_base, dsv, out_signs, m, e, gate_w, ob, ib0,
                                                ib_cnt, is_sh, dst);
    } else if (w.book == trellis::BOOK_MUL1) {
        moe_vec_book<trellis::BOOK_MUL1, POST>(c, w, x_base, dsv, out_signs, m, e, gate_w, ob, ib0,
                                               ib_cnt, is_sh, dst);
    } else {
        moe_vec_book<trellis::BOOK_MCG, POST>(c, w, x_base, dsv, out_signs, m, e, gate_w, ob, ib0,
                                              ib_cnt, is_sh, dst);
    }
}

static __device__ void call_s1(const ctx& c, const mat& w, const float* x_rot, u32 m, u32 ch,
                               u32 ib0, u32 ib_cnt, float* dst) {
    mat none = {nullptr, K_F32, 0u, 0u, 0u, 0u, nullptr, nullptr};
    moe_vec_run<false>(c, w, x_rot, none, nullptr, m, 0u, 1.0f, ch, ib0, ib_cnt, false, dst);
}

static __device__ void call_s2(const ctx& c, const mat& w, const mat& dsv, const float* out_signs,
                               u32 m, u32 e, float gate_w, u32 ob, u32 ch_off, u32 n_ch,
                               u32 cj_base, bool is_sh, float* dst) {
    moe_vec_run<true>(c, w, c.p.c + u64(cj_base) * 128u, dsv, out_signs, m, e, gate_w, ob, ch_off,
                      n_ch, is_sh, dst);
}
