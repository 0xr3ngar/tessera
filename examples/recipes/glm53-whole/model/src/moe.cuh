// MoE router, fused trellis expert pipeline, SwiGLU, and decoder layer.

static __device__ __forceinline__ bool local_pick(const ctx& c, u32 rk, u32* m) {
    u32 e = c.p.ids[rk];
    *m = e - c.s.held_first;
    return e >= c.s.held_first && e < c.s.held_first + c.s.held;
}

// Warp-parallel SwiGLU MLP into `part`.
static __device__ bool swiglu(const ctx& c, const place& pl, u32 gate, u32 inner) {
    const sizes& s = c.s;
    u32 r = s.rows;
    if (!linear_pair(c, field_mat(c, pl, gate), field_mat(c, pl, gate + 1u), c.p.h, s.hidden, c.p.a,
                     inner, inner, c.p.b, inner, inner, r)) {
        return false;
    }
    PHASE(gated(c.p.a, c.p.b, c.p.c, r * inner));
    return linear(c, field_mat(c, pl, gate + 2u), c.p.c, inner, c.p.part, s.hidden, r, s.hidden);
}

static __device__ void zero_buf(float* dst, u64 len) {
    line l = threads();
    bool vec = (reinterpret_cast<u64>(dst) & 15u) == 0u;
    u64 vec_end = vec ? (len & ~3ull) : 0ull;
    for (u64 i = u64(l.at) << 2; i < vec_end; i += u64(l.step) << 2) {
        *reinterpret_cast<float4*>(dst + i) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    for (u64 i = vec_end + l.at; i < len; i += l.step) dst[i] = 0.0f;
}

static __device__ void fill_moe_meta(const ctx& c, const place& pl, u32 mid) {
    moe_meta& mt = c.sm.meta;
    if (mid < 3u) mt.codes[mid] = field_mat(c, pl, E_GATE + 3u * mid);
    else if (mid < 6u) mt.su[mid - 3u] = field_mat(c, pl, E_GATE + 3u * (mid - 3u) + 1u);
    else if (mid < 9u) mt.sv[mid - 6u] = field_mat(c, pl, E_GATE + 3u * (mid - 6u) + 2u);
    else if (mid < 12u) mt.shared[mid - 9u] = field_mat(c, pl, S_GATE + (mid - 9u));
    else if (mid == 12u) mt.in_signs = reinterpret_cast<const float*>(field_mat(c, pl, SIGNS_IN).p);
    else mt.out_signs = reinterpret_cast<const float*>(field_mat(c, pl, SIGNS_OUT).p);
}

static __device__ void set_moe_flags(const ctx& c) {
    const sizes& s = c.s;
    moe_meta& mt = c.sm.meta;
    bool fast = (s.hidden & 127u) == 0u && (s.effn & 127u) == 0u && s.experts <= 2048u;
    bool fuse_sh = fast && (s.ffn & 127u) == 0u && s.ffn != 0u;
    for (u32 i = 0; i < 3u; i++) {
        if (mt.codes[i].klass != K_TRELLIS) fast = false;
        const mat& sh = mt.shared[i];
        u32 expect_in = i == 2u ? s.ffn : s.hidden;
        u32 expect_out = i == 2u ? s.hidden : s.ffn;
        if (sh.klass != K_TURNED || sh.suh == nullptr || sh.svh == nullptr || sh.in != expect_in
            || sh.out != expect_out) {
            fuse_sh = false;
        }
    }
    mt.flags = (fast ? 1u : 0u) | (fuse_sh && fast ? 2u : 0u);
}

static __device__ void load_moe_meta(const ctx& c, const place& pl) {
    if (threadIdx.x < 14u) fill_moe_meta(c, pl, threadIdx.x);
    __syncthreads();
    if (threadIdx.x == 0u) set_moe_flags(c);
    __syncthreads();
}

// Fused router sigmoid, bias top-k, gate norm, and buffer zeroing.
static __device__ bool route(const ctx& c, const place& pl, bool zero_t2, bool fuse_mlp_norm,
                             bool load_meta, bool* fast_routed) {
    const sizes& s = c.s;
    *fast_routed = false;
    u32 r_cnt = s.rows, e = s.experts;
    mat rw = field_mat(c, pl, ROUTER);
    u32 wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    u32 wstep = (gridDim.x * blockDim.x) >> 5;
    u32 lane = threadIdx.x & 31u;
    bool is_bf16 = rw.klass == K_BF16;
    bool is_f32 = rw.klass == K_F32;
    u64 align_mask = is_bf16 ? 7ull : 15ull;
    bool fast_r = r_cnt == 1u && (is_bf16 || is_f32) && (s.hidden & 31u) == 0u
                  && (reinterpret_cast<u64>(rw.p) & align_mask) == 0ull
                  && (reinterpret_cast<u64>(c.p.h) & 15ull) == 0ull && e > 0u
                  && e * 4u <= s.width;
    u32 r_splits = fast_r ? min(4u, max(1u, wstep / e)) : 1u;
    if (fast_r && r_splits > 1u) {
        const float* x_src = c.p.h;
        if (fuse_mlp_norm) {
            u32 n = s.hidden;
            mat nw = field_mat(c, pl, MLP_NORM);
            float inv = rms_inv(c, c.p.x, n);
            bool nw_bf16 = nw.klass == K_BF16 && (reinterpret_cast<u64>(nw.p) & 7ull) == 0ull;
            bool write_h = blockIdx.x == 0u;
            for (u32 i = threadIdx.x << 2; i < n; i += blockDim.x << 2) {
                float4 v = *reinterpret_cast<const float4*>(c.p.x + i);
                float4 wv = norm_w4(nw, nw_bf16, i);
                float4 out = make_float4(v.x * (inv * wv.x), v.y * (inv * wv.y), v.z * (inv * wv.z),
                                         v.w * (inv * wv.w));
                if (write_h) *reinterpret_cast<float4*>(c.p.h + i) = out;
                *reinterpret_cast<float4*>(c.sm.rf.x + i) = out;
            }
            __syncthreads();
            x_src = c.sm.rf.x;
        }
        u32 chunk = ((s.hidden / r_splits) + 7u) & ~7u;
        bool bf16_v4 = is_bf16 && (reinterpret_cast<u64>(rw.p) & 15ull) == 0ull
                       && (s.hidden & 7u) == 0u;
        for (u32 u = wid; u < e * r_splits; u += wstep) {
            u32 o = u / r_splits, sp = u % r_splits;
            u32 c0 = sp * chunk, c1 = min(s.hidden, c0 + chunk);
            float acc = 0.0f;
            if (bf16_v4) {
                const u16* wr = reinterpret_cast<const u16*>(rw.p) + u64(o) * s.hidden;
                for (u32 i = c0 + (lane << 3); i < c1; i += 256u) {
                    uint4 h = ld_nc_u4(wr + i);
                    float4 x0 = *reinterpret_cast<const float4*>(x_src + i);
                    float4 x1 = *reinterpret_cast<const float4*>(x_src + i + 4u);
                    acc = fmaf(__uint_as_float(h.x << 16), x0.x,
                               fmaf(__uint_as_float(h.x & 0xFFFF0000u), x0.y,
                                    fmaf(__uint_as_float(h.y << 16), x0.z,
                                         fmaf(__uint_as_float(h.y & 0xFFFF0000u), x0.w, acc))));
                    acc = fmaf(__uint_as_float(h.z << 16), x1.x,
                               fmaf(__uint_as_float(h.z & 0xFFFF0000u), x1.y,
                                    fmaf(__uint_as_float(h.w << 16), x1.z,
                                         fmaf(__uint_as_float(h.w & 0xFFFF0000u), x1.w, acc))));
                }
            } else if (is_bf16) {
                const u16* wr = reinterpret_cast<const u16*>(rw.p) + u64(o) * s.hidden;
                for (u32 i = c0 + (lane << 2); i < c1; i += 128u) {
                    uint2 h = *reinterpret_cast<const uint2*>(wr + i);
                    float4 xv = *reinterpret_cast<const float4*>(x_src + i);
                    acc = fmaf(__uint_as_float(h.x << 16), xv.x,
                               fmaf(__uint_as_float(h.x & 0xFFFF0000u), xv.y,
                                    fmaf(__uint_as_float(h.y << 16), xv.z,
                                         fmaf(__uint_as_float(h.y & 0xFFFF0000u), xv.w, acc))));
                }
            } else {
                const float* wr = reinterpret_cast<const float*>(rw.p) + u64(o) * s.hidden;
                for (u32 i = c0 + (lane << 2); i < c1; i += 128u) {
                    float4 wv = *reinterpret_cast<const float4*>(wr + i);
                    float4 xv = *reinterpret_cast<const float4*>(x_src + i);
                    acc = fmaf(wv.x, xv.x, fmaf(wv.y, xv.y, fmaf(wv.z, xv.z, fmaf(wv.w, xv.w, acc))));
                }
            }
            float sum = warp_sum(acc);
            if (lane == 0u) c.p.b[u] = sum;
        }
        if (!sync(c.t)) return false;
    } else {
        if (fuse_mlp_norm) {
            PHASE(rms(c, c.p.x, s.hidden, field_mat(c, pl, MLP_NORM), c.p.h, s.hidden, s.rows,
                      s.hidden, nullptr, false));
        }
        if (!linear(c, rw, c.p.h, s.hidden, c.p.a, e, r_cnt, e)) return false;
    }
    if (r_cnt == 1u && fast_r && r_splits > 1u && e <= 256u) {
        bool try_all = (load_meta || zero_t2) && s.topk <= 8u;
        if (load_meta && threadIdx.x >= 32u && threadIdx.x < 46u) {
            fill_moe_meta(c, pl, threadIdx.x - 32u);
        } else if (try_all ? (threadIdx.x < 32u) : (wid == 0u)) {
            mat bias = field_mat(c, pl, R_BIAS);
            float sc_reg[8], b_reg[8];
#pragma unroll
            for (u32 i = 0; i < 8u; i++) {
                u32 idx = lane + (i << 5);
                if (idx < e) {
                    float logit = 0.0f;
                    for (u32 sp = 0; sp < r_splits; sp++) logit += c.p.b[idx * r_splits + sp];
                    float sv = 1.0f / (1.0f + __expf(-logit));
                    sc_reg[i] = sv;
                    b_reg[i] = sv + vec_elem(bias, 0u, idx);
                } else {
                    sc_reg[i] = 0.0f;
                    b_reg[i] = __int_as_float(0xFF800000u);
                }
            }
            u32 taken = 0u;
            float sum = 0.0f;
            u32 sel_id[8];
            float sel_s[8];
            for (u32 k = 0; k < s.topk; k++) {
                u32 best = e;
                float most = __int_as_float(0xFF800000u);
                float best_sc = 0.0f;
#pragma unroll
                for (u32 i = 0; i < 8u; i++) {
                    u32 idx = lane + (i << 5);
                    if (idx >= e || ((taken >> i) & 1u) != 0u) continue;
                    float v = b_reg[i];
                    if (best == e || v > most) {
                        best = idx;
                        most = v;
                        best_sc = sc_reg[i];
                    }
                }
#pragma unroll
                for (u32 off = 16; off > 0; off >>= 1) {
                    float pv = __shfl_xor_sync(0xFFFFFFFFu, most, off);
                    u32 pe = __shfl_xor_sync(0xFFFFFFFFu, best, off);
                    float ps = __shfl_xor_sync(0xFFFFFFFFu, best_sc, off);
                    bool win = pe < e && (best == e || pv > most || (pv == most && pe < best));
                    if (win) {
                        best = pe;
                        most = pv;
                        best_sc = ps;
                    }
                }
                if ((best & 31u) == lane) taken |= 1u << (best >> 5);
                if (lane == 0u) {
                    sel_id[k] = best;
                    sel_s[k] = best_sc;
                    sum += best_sc;
                }
            }
            if (lane == 0u) {
                float inv = s.route / sum;
                u32 cnt = 0u;
                for (u32 k = 0; k < s.topk; k++) {
                    u32 id = sel_id[k];
                    float g = sel_s[k] * inv;
                    if (blockIdx.x == 0u) {
                        c.p.ids[k] = id;
                        c.p.gates[k] = g;
                    }
                    if (try_all && id >= s.held_first && id < s.held_first + s.held) {
                        c.sm.meta.loc_m[cnt] = id - s.held_first;
                        c.sm.meta.loc_e[cnt] = id;
                        c.sm.meta.loc_g[cnt] = g;
                        cnt++;
                    }
                }
                if (try_all) c.sm.meta.cnt = cnt;
            }
        }
        if (load_meta) {
            __syncthreads();
            if (threadIdx.x == 0u) set_moe_flags(c);
            __syncthreads();
            zero_t2 = (c.sm.meta.flags & 2u) != 0u;
        }
        if (zero_t2 && s.topk <= 8u) {
            if (!load_meta) __syncthreads();
            *fast_routed = true;
            return true;
        }
        zero_buf(c.p.d, u64(s.topk) * s.hidden);
        if (zero_t2) zero_buf(c.p.t2, s.hidden);
        return sync(c.t);
    }
    mat bias = field_mat(c, pl, R_BIAS);
    for (u32 row = wid; row < r_cnt; row += wstep) {
        float* scores = c.p.a + u64(row) * e;
        if (fast_r && r_splits > 1u) {
            for (u32 idx = lane; idx < e; idx += 32u) {
                float logit = 0.0f;
                for (u32 sp = 0; sp < r_splits; sp++) logit += c.p.b[idx * r_splits + sp];
                scores[idx] = 1.0f / (1.0f + __expf(-logit));
            }
        } else {
            for (u32 idx = lane; idx < e; idx += 32u) scores[idx] = 1.0f / (1.0f + __expf(-scores[idx]));
        }
        __syncwarp();
        u64 taken = 0ull;
        float sum = 0.0f;
        u32* out_ids = c.p.ids + u64(row) * s.topk;
        float* out_gates = c.p.gates + u64(row) * s.topk;
        for (u32 k = 0; k < s.topk; k++) {
            u32 best = e;
            float most = __int_as_float(0xFF800000u);
            for (u32 i = 0, idx = lane; idx < e; i++, idx += 32u) {
                bool used = i < 64u ? (((taken >> i) & 1ull) != 0ull) : false;
                if (e > 2048u && !used) {
                    for (u32 p = 0; p < k; p++) {
                        if (out_ids[p] == idx) used = true;
                    }
                }
                if (used) continue;
                float v = scores[idx] + vec_elem(bias, 0u, idx);
                if (best == e || v > most) {
                    best = idx;
                    most = v;
                }
            }
#pragma unroll
            for (u32 off = 16; off > 0; off >>= 1) {
                float pv = __shfl_xor_sync(0xFFFFFFFFu, most, off);
                u32 pe = __shfl_xor_sync(0xFFFFFFFFu, best, off);
                bool win = pe < e && (best == e || pv > most || (pv == most && pe < best));
                if (win) {
                    best = pe;
                    most = pv;
                }
            }
            if ((best & 31u) == lane && (best >> 5) < 64u) taken |= 1ull << (best >> 5);
            if (lane == 0u) {
                out_ids[k] = best;
                sum += scores[best];
            }
            __syncwarp();
        }
        sum = __shfl_sync(0xFFFFFFFFu, sum, 0);
        for (u32 k = lane; k < s.topk; k += 32u) {
            out_gates[k] = scores[out_ids[k]] / sum * s.route;
        }
    }
    bool skip_zero = r_cnt == 1u && zero_t2 && s.topk <= 8u;
    if (!skip_zero) {
        zero_buf(c.p.d, u64(r_cnt) * s.topk * s.hidden);
        if (zero_t2) zero_buf(c.p.t2, u64(r_cnt) * s.hidden);
    }
    return sync(c.t);
}

// One warp's gate or up trellis MMA pass over hidden.
template <u32 K, u32 BOOK>
static __device__ __noinline__ void routed_gu(const ctx& c, const mat& w, const mat& su,
                                              const float* in_signs, u32 m, u32 e, u32 ch, u32 n) {
    u32 warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    u32 pair_id = warp >> 1, h = c.s.hidden;
    u32 in_blocks = h >> 7;
    u32 my_ibs = in_blocks > pair_id ? ((in_blocks - 1u - pair_id) / (WARPS >> 1) + 1u) : 0u;
    u32 steps = my_ibs << 3;
    trellis::lane_map pos = trellis::map_lane<K, BOOK>(lane);
    bool wide = (reinterpret_cast<u64>(w.p + trellis::HEAD) & 15u) == 0u;
    bool input_side = word(w.p, 8) == trellis::INPUT_SIDE;
    bool h_vec = (reinterpret_cast<u64>(c.p.h) & 15u) == 0u;
    bool s_vec = in_signs != nullptr && (reinterpret_cast<u64>(in_signs) & 15u) == 0u;
    bool su_vec = w.suh != nullptr && (reinterpret_cast<u64>(w.suh) & 15u) == 0u;
    float* wbuf = &c.sm.moe.buf[warp][0][0];
    u32* wtiles = c.sm.moe.tiles[warp];
    float acc[8][4];
    zero_acc(acc);
    if (steps > 0u) {
        u32 reg[8];
        trellis::tile_load<K>(
            trellis::tile_ptr<K>(w.p, w.in, w.out, m, pair_id << 3, ch, input_side), lane, wide,
            reg);
        for (u32 st = 0; st < steps; st++) {
            u32 ib = pair_id + (st >> 3) * (WARPS >> 1);
            u32 ks = st & 7u;
            if (ks == 0u) {
                __syncwarp();
                u32 base = (ib << 7) + (lane << 2);
                float sc[4];
                if (in_signs != nullptr) {
                    float sg[4];
                    load_vec4(in_signs + base, s_vec, sg);
                    for (u32 q = 0; q < 4; q++) sc[q] = sg[q] * vec_elem(su, e, base + q);
                } else {
                    load_vec4(w.suh + base, su_vec, sc);
                }
                for (u32 r = 0; r < n; r++) {
                    float v[4];
                    load_vec4(c.p.h + u64(c.sm.moe.rows[r]) * h + base, h_vec, v);
                    for (u32 q = 0; q < 4; q++) v[q] *= sc[q];
                    trellis::wht128(v, trellis::INV_SQRT128);
                    *reinterpret_cast<float4*>(wbuf + (r << 7) + (lane << 2)) =
                        make_float4(v[0], v[1], v[2], v[3]);
                }
                __syncwarp();
            }
            const u8* next = nullptr;
            if (st + 1u < steps) {
                u32 nti = ((pair_id + ((st + 1u) >> 3) * (WARPS >> 1)) << 3) + ((st + 1u) & 7u);
                next = trellis::tile_ptr<K>(w.p, w.in, w.out, m, nti, ch, input_side);
            }
            trellis_step<K, BOOK>(reg, next, lane, wide, wtiles, wbuf, n, ks, pos, acc);
        }
    }
    __syncwarp();
    trellis::store_acc(acc, n, lane, wbuf);
}

// One warp's down trellis MMA pass into `dst_buf`.
template <u32 K, u32 BOOK>
static __device__ __noinline__ void routed_down(const ctx& c, const mat& w, u32 m, u32 ch,
                                                u32 chunks, u32 n, float* dst_buf) {
    u32 warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    u32 h = c.s.hidden, out_blocks = h >> 7;
    u32 my_obs = out_blocks > warp ? ((out_blocks - 1u - warp) / WARPS + 1u) : 0u;
    u32 steps = my_obs << 3;
    if (steps == 0u) return;
    trellis::lane_map pos = trellis::map_lane<K, BOOK>(lane);
    bool wide = (reinterpret_cast<u64>(w.p + trellis::HEAD) & 15u) == 0u;
    bool input_side = word(w.p, 8) == trellis::INPUT_SIDE;
    u32* wtiles = c.sm.moe.tiles[warp];
    u32 reg[8];
    trellis::tile_load<K>(trellis::tile_ptr<K>(w.p, w.in, w.out, m, ch << 3, warp, input_side),
                          lane, wide, reg);
    float out[8][4];
    u32 r = lane & 3u, g = lane >> 2;
    for (u32 st = 0; st < steps; st++) {
        u32 ob = warp + (st >> 3) * WARPS;
        u32 ks = st & 7u;
        if (ks == 0u) zero_acc(out);
        const u8* next = nullptr;
        if (st + 1u < steps) {
            u32 nob = warp + ((st + 1u) >> 3) * WARPS;
            next = trellis::tile_ptr<K>(w.p, w.in, w.out, m, (ch << 3) + ((st + 1u) & 7u), nob,
                                        input_side);
        }
        trellis_step<K, BOOK>(reg, next, lane, wide, wtiles, &c.sm.moe.vec[0][0], n, ks, pos, out);
        if (ks == 7u && r < n) {
            float* dst = dst_buf + u64(c.sm.moe.picks[r]) * h + (ob << 7) + g;
#pragma unroll
            for (u32 mt = 0; mt < 8; mt++) {
                float v0 = out[mt][0] + out[mt][1];
                float v1 = out[mt][2] + out[mt][3];
                if (chunks == 1u) {
                    dst[mt * 16u] = v0;
                    dst[mt * 16u + 8u] = v1;
                } else {
                    atomicAdd(&dst[mt * 16u], v0);
                    atomicAdd(&dst[mt * 16u + 8u], v1);
                }
            }
        }
    }
}

template <u32 BOOK>
static __device__ void gu_book(const ctx& c, const mat& w, const mat& su, const float* in_signs,
                               u32 m, u32 e, u32 ch, u32 n) {
    if (w.k == 2u) routed_gu<2u, BOOK>(c, w, su, in_signs, m, e, ch, n);
    else if (w.k == 3u) routed_gu<3u, BOOK>(c, w, su, in_signs, m, e, ch, n);
    else routed_gu<4u, BOOK>(c, w, su, in_signs, m, e, ch, n);
}

template <u32 BOOK>
static __device__ void down_book(const ctx& c, const mat& w, u32 m, u32 ch, u32 chunks, u32 n,
                                 float* dst_buf) {
    if (w.k == 2u) routed_down<2u, BOOK>(c, w, m, ch, chunks, n, dst_buf);
    else if (w.k == 3u) routed_down<3u, BOOK>(c, w, m, ch, chunks, n, dst_buf);
    else routed_down<4u, BOOK>(c, w, m, ch, chunks, n, dst_buf);
}

static __device__ void call_gu(const ctx& c, const mat& w, const mat& su, const float* in_signs,
                               u32 m, u32 e, u32 ch, u32 n) {
    if (w.book == trellis::BOOK_PMCGX) gu_book<trellis::BOOK_PMCGX>(c, w, su, in_signs, m, e, ch, n);
    else if (w.book == trellis::BOOK_MUL1) gu_book<trellis::BOOK_MUL1>(c, w, su, in_signs, m, e, ch, n);
    else gu_book<trellis::BOOK_MCG>(c, w, su, in_signs, m, e, ch, n);
}

static __device__ void call_down(const ctx& c, const mat& w, u32 m, u32 ch, u32 chunks, u32 n,
                                 float* dst_buf) {
    if (w.book == trellis::BOOK_PMCGX) down_book<trellis::BOOK_PMCGX>(c, w, m, ch, chunks, n, dst_buf);
    else if (w.book == trellis::BOOK_MUL1) down_book<trellis::BOOK_MUL1>(c, w, m, ch, chunks, n, dst_buf);
    else down_book<trellis::BOOK_MCG>(c, w, m, ch, chunks, n, dst_buf);
}

// MoE phase 1: fused shared and routed expert chunks.
static __device__ __noinline__ void routed_step(const ctx& c, const mat codes[3], const mat su[3],
                                                const mat sv[3], const float* in_signs,
                                                const mat shared[3], bool fuse_sh) {
    const sizes& s = c.s;
    u32 warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    u32 groups = (s.rows + ROW_TILE - 1u) / ROW_TILE;
    u32 chunks = s.effn >> 7;
    u32 sh_chunks = fuse_sh ? (s.ffn >> 7) : 0u;
    mat none = {nullptr, K_F32, 0u, 0u, 0u, 0u, nullptr, nullptr};
    u32 cursor = 0u;
    for (u32 gr = 0; gr < groups; gr++) {
        u32 base = gr * ROW_TILE, nr = min(ROW_TILE, s.rows - base);
        for (u32 i = threadIdx.x; i < 64u; i += blockDim.x) c.sm.moe.mask[i] = 0u;
        __syncthreads();
        for (u32 i = threadIdx.x; i < nr * s.topk; i += blockDim.x) {
            u32 e = c.p.ids[(base + i / s.topk) * s.topk + (i % s.topk)];
            if (e >= s.held_first && e < s.held_first + s.held) {
                atomicOr(&c.sm.moe.mask[e >> 5], 1u << (e & 31u));
            }
        }
        __syncthreads();
        if (threadIdx.x == 0u) {
            u32 tot = 0u;
            for (u32 i = 0; i < 64u; i++) tot += __popc(c.sm.moe.mask[i]);
            c.sm.moe.count = tot;
        }
        __syncthreads();
        u32 jobs = sh_chunks + c.sm.moe.count * chunks;
        u32 rem = cursor % gridDim.x;
        u32 first = blockIdx.x >= rem ? (blockIdx.x - rem) : (blockIdx.x + gridDim.x - rem);
        for (u32 job = first; job < jobs; job += gridDim.x) {
            bool is_sh = job < sh_chunks;
            u32 ch, e = 0u, m = 0u, n = nr;
            if (is_sh) {
                ch = job;
                if (threadIdx.x < nr) {
                    c.sm.moe.rows[threadIdx.x] = base + threadIdx.x;
                    c.sm.moe.picks[threadIdx.x] = base + threadIdx.x;
                }
                __syncthreads();
            } else {
                u32 rjob = job - sh_chunks;
                u32 slot = rjob / chunks;
                ch = rjob % chunks;
                if (threadIdx.x == 0u) {
                    u32 left = slot, exp_id = s.experts;
                    for (u32 i = 0; i < 64u && exp_id == s.experts; i++) {
                        u32 bits = c.sm.moe.mask[i], have = __popc(bits);
                        if (left < have) {
                            for (u32 t = 0; t < left; t++) bits &= bits - 1u;
                            exp_id = (i << 5) + u32(__ffs(bits) - 1);
                        } else {
                            left -= have;
                        }
                    }
                    u32 cnt = 0u;
                    for (u32 r = 0; r < nr; r++) {
                        for (u32 k = 0; k < s.topk; k++) {
                            u32 pick = (base + r) * s.topk + k;
                            if (c.p.ids[pick] == exp_id && cnt < ROW_TILE) {
                                c.sm.moe.rows[cnt] = base + r;
                                c.sm.moe.picks[cnt] = pick;
                                cnt++;
                            }
                        }
                    }
                    c.sm.moe.chosen = exp_id | (cnt << 16);
                }
                __syncthreads();
                e = c.sm.moe.chosen & 0xFFFFu;
                n = c.sm.moe.chosen >> 16;
                m = e - s.held_first;
            }
            u32 side = warp & 1u;
            if (is_sh) {
                call_gu(c, shared[side], none, nullptr, 0u, 0u, ch, n);
            } else {
                call_gu(c, codes[side], su[side], in_signs, m, e, ch, n);
            }
            __syncthreads();
            if (warp < n) {
                float gv[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                float uv[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                for (u32 widx = 0; widx < WARPS; widx += 2u) {
                    float4 pg =
                        *reinterpret_cast<const float4*>(&c.sm.moe.buf[widx][warp][lane << 2]);
                    float4 pu = *reinterpret_cast<const float4*>(
                        &c.sm.moe.buf[widx + 1u][warp][lane << 2]);
                    gv[0] += pg.x;
                    gv[1] += pg.y;
                    gv[2] += pg.z;
                    gv[3] += pg.w;
                    uv[0] += pu.x;
                    uv[1] += pu.y;
                    uv[2] += pu.z;
                    uv[3] += pu.w;
                }
                trellis::wht128(gv, trellis::INV_SQRT128);
                trellis::wht128(uv, trellis::INV_SQRT128);
                u32 o_base = (ch << 7) + (lane << 2);
                float act[4];
                for (u32 q = 0; q < 4; q++) {
                    u32 o = o_base + q;
                    float gval = gv[q] * (is_sh ? shared[0].svh[o] : vec_elem(sv[0], e, o));
                    float uval = uv[q] * (is_sh ? shared[1].svh[o] : vec_elem(sv[1], e, o));
                    float dsu = is_sh ? shared[2].suh[o] : vec_elem(su[2], e, o);
                    act[q] = (gval / (1.0f + expf(-gval))) * uval * dsu;
                }
                trellis::wht128(act, trellis::INV_SQRT128);
                *reinterpret_cast<float4*>(&c.sm.moe.vec[warp][lane << 2]) =
                    make_float4(act[0], act[1], act[2], act[3]);
            }
            __syncthreads();
            if (is_sh) {
                call_down(c, shared[2], 0u, ch, sh_chunks, n, c.p.t2);
            } else {
                call_down(c, codes[2], m, ch, chunks, n, c.p.d);
            }
            __syncthreads();
        }
        cursor += jobs;
        __syncthreads();
    }
}

// MoE phase 2: WHT, scales, shared merge, and `part` write.
static __device__ void routed_out(const ctx& c, const mat& down_sv, const float* out_signs,
                                  const float* shared_svh) {
    const sizes& s = c.s;
    u32 h = s.hidden, ob_cnt = h >> 7;
    u32 wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    u32 wstep = (gridDim.x * blockDim.x) >> 5;
    u32 lane = threadIdx.x & 31u;
    bool d_vec = (reinterpret_cast<u64>(c.p.d) & 15u) == 0u;
    bool t2_vec = (reinterpret_cast<u64>(c.p.t2) & 15u) == 0u;
    bool p_vec = (reinterpret_cast<u64>(c.p.part) & 15u) == 0u;
    bool s_vec = (reinterpret_cast<u64>(out_signs) & 15u) == 0u;
    bool sh_vec = shared_svh != nullptr && (reinterpret_cast<u64>(shared_svh) & 15u) == 0u;
    for (u32 u = wid; u < s.rows * ob_cnt; u += wstep) {
        u32 r = u / ob_cnt, ob = u % ob_cnt;
        u32 base = (ob << 7) + (lane << 2);
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        if (shared_svh != nullptr) {
            load_vec4(c.p.t2 + u64(r) * h + base, t2_vec, acc);
            trellis::wht128(acc, trellis::INV_SQRT128);
            float shs[4];
            load_vec4(shared_svh + base, sh_vec, shs);
            for (u32 q = 0; q < 4; q++) acc[q] *= shs[q];
        }
        float sg[4];
        load_vec4(out_signs + base, s_vec, sg);
        for (u32 k = 0; k < s.topk; k++) {
            u32 pick = r * s.topk + k, m;
            if (!local_pick(c, pick, &m)) continue;
            u32 e = m + s.held_first;
            float v[4];
            load_vec4(c.p.d + u64(pick) * h + base, d_vec, v);
            trellis::wht128(v, trellis::INV_SQRT128);
            float g = c.p.gates[pick];
            for (u32 q = 0; q < 4; q++) {
                acc[q] = fmaf(g, v[q] * vec_elem(down_sv, e, base + q) * sg[q], acc[q]);
            }
        }
        float* dst = c.p.part + u64(r) * h + base;
        if (shared_svh != nullptr) {
            store_vec4(dst, p_vec, acc);
        } else {
            float cur[4];
            load_vec4(dst, p_vec, cur);
            for (u32 q = 0; q < 4; q++) cur[q] += acc[q];
            store_vec4(dst, p_vec, cur);
        }
    }
}

// Scalar fallback for non-trellis or unaligned routed experts.
static __device__ void turn_picks(const ctx& c, const float* x, const float* sign, const mat& by,
                                  const mat& after, const float* also, float* y, u32 n) {
    line l = threads();
    for (u32 u = l.at; u < c.s.rows * c.s.topk * n; u += l.step) {
        u32 rk = u / n, i = u % n, m;
        if (!local_pick(c, rk, &m)) continue;
        u32 e = m + c.s.held_first, base = i & ~127u, o = i & 127u;
        const float* row = x + u64(sign ? rk / c.s.topk : rk) * n;
        float sum = 0.0f;
        for (u32 j = 0; j < 128u; j++) {
            float v = row[base + j] * (by.p ? vec_elem(by, e, base + j) : 1.0f);
            if (sign) v *= sign[base + j];
            sum += (__popc(o & j) & 1) ? -v : v;
        }
        float v = sum * trellis::INV_SQRT128;
        if (after.p) v *= vec_elem(after, e, i);
        y[u] = also ? v * also[i] : v;
    }
}

static __device__ void picked_mat(const ctx& c, const mat& w, const float* x, float* y, u32 out,
                                  u32 in) {
    u32 wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    u32 wstep = (gridDim.x * blockDim.x) >> 5;
    u32 lane = threadIdx.x & 31u;
    for (u32 u = wid; u < c.s.rows * c.s.topk * out; u += wstep) {
        u32 rk = u / out, o = u % out, m;
        if (!local_pick(c, rk, &m)) continue;
        float acc = 0.0f;
        for (u32 i = lane; i < in; i += 32u) acc = fmaf(elem(w, m, o, i), x[u64(rk) * in + i], acc);
        float sum = warp_sum(acc);
        if (lane == 0u) y[u] = sum;
    }
}

static __device__ __noinline__ bool routed_slow(const ctx& c, const mat codes[3], const mat su[3],
                                                const mat sv[3], const float* in_signs,
                                                const float* out_signs) {
    const sizes& s = c.s;
    line l = threads();
    u32 h = s.hidden, f = s.effn, rk = s.rows * s.topk;
    mat none = {nullptr, K_F32, 0u, 0u, 0u, 0u, nullptr, nullptr};
    PHASE(turn_picks(c, c.p.h, in_signs, su[0], none, nullptr, c.p.b, h);
          turn_picks(c, c.p.h, in_signs, su[1], none, nullptr, c.p.c, h));
    PHASE(picked_mat(c, codes[0], c.p.b, c.p.d, f, h);
          picked_mat(c, codes[1], c.p.c, c.p.t1, f, h));
    PHASE(turn_picks(c, c.p.d, nullptr, none, sv[0], nullptr, c.p.b, f);
          turn_picks(c, c.p.t1, nullptr, none, sv[1], nullptr, c.p.c, f));
    for (u32 u = l.at; u < rk * f; u += l.step) {
        u32 m;
        if (!local_pick(c, u / f, &m)) continue;
        float g = c.p.b[u];
        c.p.t2[u] = g / (1.0f + expf(-g)) * c.p.c[u];
    }
    if (!sync(c.t)) return false;
    PHASE(turn_picks(c, c.p.t2, nullptr, su[2], none, nullptr, c.p.b, f));
    PHASE(picked_mat(c, codes[2], c.p.b, c.p.d, h, f));
    PHASE(turn_picks(c, c.p.d, nullptr, none, sv[2], out_signs, c.p.c, h));
    for (u32 u = l.at; u < s.rows * h; u += l.step) {
        u32 r = u / h, o = u % h, m;
        float sum = 0.0f;
        for (u32 k = 0; k < s.topk; k++) {
            u32 pick = r * s.topk + k;
            if (local_pick(c, pick, &m)) sum = fmaf(c.p.gates[pick], c.p.c[u64(pick) * h + o], sum);
        }
        c.p.part[u] += sum;
    }
    return sync(c.t);
}

static __device__ bool routed(const ctx& c, const place& pl, bool fuse_sh, bool split,
                              bool fast_routed, u32* moe_cnt, ticket* pre_r, bool* has_pre) {
    const sizes& s = c.s;
    *moe_cnt = 0xFFFFFFFFu;
    *has_pre = false;
    moe_meta& mt = c.sm.meta;
    if (!fast_routed) {
        if (threadIdx.x == 0u) {
            u32 cnt = 0u;
            if ((mt.flags & 1u) != 0u && s.rows == 1u && fuse_sh && s.topk <= 8u) {
                for (u32 k = 0; k < s.topk; k++) {
                    u32 m;
                    if (local_pick(c, k, &m)) {
                        mt.loc_m[cnt] = m;
                        mt.loc_e[cnt] = m + s.held_first;
                        mt.loc_g[cnt] = c.p.gates[k];
                        cnt++;
                    }
                }
            }
            mt.cnt = cnt;
        }
        __syncthreads();
    }
    if ((mt.flags & 1u) == 0u) {
        return routed_slow(c, mt.codes, mt.su, mt.sv, mt.in_signs, mt.out_signs);
    }
    if (s.rows == 1u && fuse_sh && s.topk <= 8u) {
        u32 cnt = mt.cnt;
        u32 sh_chunks = s.ffn >> 7, r_chunks = s.effn >> 7;
        u32 tot_chunks = sh_chunks + cnt * r_chunks;
        u32 in_blocks = s.hidden >> 7;
        u32 k_splits = 1u;
        u32 best_cost = ((tot_chunks * 2u + gridDim.x - 1u) / gridDim.x) * in_blocks;
        u32 cands[3] = {2u, 3u, 6u};
        for (u32 ci = 0u; ci < 3u; ci++) {
            u32 ks = cands[ci];
            if ((in_blocks % ks) == 0u && ((in_blocks / ks) % WARPS) == 0u
                && tot_chunks * 128u * ks <= s.width) {
                u32 jobs = tot_chunks * 2u * ks;
                u32 cost = ((jobs + gridDim.x - 1u) / gridDim.x) * (in_blocks / ks);
                if (cost < best_cost) {
                    best_cost = cost;
                    k_splits = ks;
                }
            }
        }
        if (tot_chunks * 128u * k_splits <= s.width && 2u * s.hidden <= s.width
            && cnt * s.hidden <= s.width) {
            u32 wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
            u32 wstep = (gridDim.x * blockDim.x) >> 5;
            u32 lane = threadIdx.x & 31u;
            u32 warp = threadIdx.x >> 5;
            bool h_vec = (reinterpret_cast<u64>(c.p.h) & 15u) == 0u;
            bool s_vec = (reinterpret_cast<u64>(mt.in_signs) & 15u) == 0u;
            u32 n_vecs = (1u + cnt) * 2u;
            for (u32 u = wid; u < n_vecs * in_blocks; u += wstep) {
                u32 v_idx = u / in_blocks, ib = u % in_blocks;
                u32 base = (ib << 7) + (lane << 2);
                u32 side = v_idx & 1u;
                bool is_sh = v_idx < 2u;
                float v[4];
                load_vec4(c.p.h + base, h_vec, v);
                float* dst;
                if (is_sh) {
                    float sc[4];
                    load_vec4(mt.shared[side].suh + base,
                              (reinterpret_cast<u64>(mt.shared[side].suh) & 15u) == 0u, sc);
                    for (u32 q = 0; q < 4u; q++) v[q] *= sc[q];
                    dst = c.p.c + u64(side) * s.hidden;
                } else {
                    u32 rslot = (v_idx - 2u) >> 1;
                    u32 e = mt.loc_e[rslot];
                    float sg[4], su4[4];
                    load_vec4(mt.in_signs + base, s_vec, sg);
                    vec_elem4(mt.su[side], e, base, su4);
                    for (u32 q = 0; q < 4u; q++) v[q] *= sg[q] * su4[q];
                    dst = (side == 0u ? c.p.t1 : c.p.d) + u64(rslot) * s.hidden;
                }
                trellis::wht128(v, trellis::INV_SQRT128);
                *reinterpret_cast<float4*>(dst + base) = make_float4(v[0], v[1], v[2], v[3]);
            }
            if (!sync(c.t)) return false;
            u32 ib_cnt = in_blocks / k_splits;
            u32 jobs1 = tot_chunks * 2u * k_splits;
            for (u32 job = blockIdx.x; job < jobs1; job += gridDim.x) {
                u32 sp = job % k_splits;
                u32 rem = job / k_splits;
                u32 side = rem & 1u, cj = rem >> 1;
                bool is_sh = cj < sh_chunks;
                u32 rslot = is_sh ? 0u : ((cj - sh_chunks) / r_chunks);
                u32 ch = is_sh ? cj : ((cj - sh_chunks) % r_chunks);
                const float* x_rot = is_sh ? (c.p.c + u64(side) * s.hidden)
                                           : ((side == 0u ? c.p.t1 : c.p.d) + u64(rslot) * s.hidden);
                float* dst = (side == 0u ? c.p.a : c.p.b) + u64(sp * tot_chunks + cj) * 128u;
                call_s1(c, is_sh ? mt.shared[side] : mt.codes[side], x_rot,
                        is_sh ? 0u : mt.loc_m[rslot], ch, sp * ib_cnt, ib_cnt, dst);
            }
            if (!sync(c.t)) return false;
            if (warp == 0u) {
                for (u32 cj = blockIdx.x; cj < tot_chunks; cj += gridDim.x) {
                    bool is_sh = cj < sh_chunks;
                    u32 rslot = is_sh ? 0u : ((cj - sh_chunks) / r_chunks);
                    u32 ch = is_sh ? cj : ((cj - sh_chunks) % r_chunks);
                    u32 e = is_sh ? 0u : mt.loc_e[rslot];
                    float gv[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                    float uv[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                    for (u32 sp = 0u; sp < k_splits; sp++) {
                        float4 pg = *reinterpret_cast<const float4*>(
                            c.p.a + u64(sp * tot_chunks + cj) * 128u + (lane << 2));
                        float4 pu = *reinterpret_cast<const float4*>(
                            c.p.b + u64(sp * tot_chunks + cj) * 128u + (lane << 2));
                        gv[0] += pg.x;
                        gv[1] += pg.y;
                        gv[2] += pg.z;
                        gv[3] += pg.w;
                        uv[0] += pu.x;
                        uv[1] += pu.y;
                        uv[2] += pu.z;
                        uv[3] += pu.w;
                    }
                    trellis::wht128(gv, trellis::INV_SQRT128);
                    trellis::wht128(uv, trellis::INV_SQRT128);
                    u32 o_base = (ch << 7) + (lane << 2);
                    float sv0[4], sv1[4], dsu4[4];
                    if (is_sh) {
                        load_vec4(mt.shared[0].svh + o_base,
                                  (reinterpret_cast<u64>(mt.shared[0].svh) & 15u) == 0u, sv0);
                        load_vec4(mt.shared[1].svh + o_base,
                                  (reinterpret_cast<u64>(mt.shared[1].svh) & 15u) == 0u, sv1);
                        load_vec4(mt.shared[2].suh + o_base,
                                  (reinterpret_cast<u64>(mt.shared[2].suh) & 15u) == 0u, dsu4);
                    } else {
                        vec_elem4(mt.sv[0], e, o_base, sv0);
                        vec_elem4(mt.sv[1], e, o_base, sv1);
                        vec_elem4(mt.su[2], e, o_base, dsu4);
                    }
                    float act[4];
                    for (u32 q = 0u; q < 4u; q++) {
                        float gval = gv[q] * sv0[q];
                        float uval = uv[q] * sv1[q];
                        act[q] = (gval / (1.0f + __expf(-gval))) * uval * dsu4[q];
                    }
                    trellis::wht128(act, trellis::INV_SQRT128);
                    *reinterpret_cast<float4*>(c.p.c + u64(cj) * 128u + (lane << 2)) =
                        make_float4(act[0], act[1], act[2], act[3]);
                }
            }
            if (!sync(c.t)) return false;
            if (split) {
                const tessera::lane* ln = get<const tessera::lane>(c.t, c.t->lanes);
                u32 room = detail::pools(c.t)->pools[ln->pool].send_bytes
                           >> (ln->elem == BF16 ? 1u : 2u);
                if (s.hidden <= room) {
                    *pre_r = prepare(c.se, 0u);
                    *has_pre = true;
                }
            }
            u32 ob_cnt = s.hidden >> 7;
            u32 r_splits = (cnt > 0u && cnt * 2u * s.hidden <= s.width && (r_chunks & 1u) == 0u
                            && (r_chunks >> 1) >= WARPS)
                               ? 2u
                               : 1u;
            u32 r_nch = r_chunks / r_splits;
            u32 r_jobs = cnt * r_splits * ob_cnt;
            u32 jobs2 = r_jobs + ob_cnt;
            for (u32 job = blockIdx.x; job < jobs2; job += gridDim.x) {
                if (job < r_jobs) {
                    u32 slot = job / ob_cnt, ob = job % ob_cnt;
                    u32 rslot = slot / r_splits, half = slot % r_splits;
                    float* dst = c.p.d + u64(slot) * s.hidden;
                    call_s2(c, mt.codes[2], mt.sv[2], mt.out_signs, mt.loc_m[rslot], mt.loc_e[rslot],
                            mt.loc_g[rslot], ob, half * r_nch, r_nch, sh_chunks + rslot * r_chunks,
                            false, dst);
                } else {
                    u32 ob = job - r_jobs;
                    call_s2(c, mt.shared[2], mt.sv[2], mt.out_signs, 0u, 0u, 1.0f, ob, 0u,
                            sh_chunks, 0u, true, c.p.t2);
                }
            }
            if (!sync(c.t)) return false;
            *moe_cnt = cnt * r_splits;
            return true;
        }
    }
    PHASE(routed_step(c, mt.codes, mt.su, mt.sv, mt.in_signs, mt.shared, fuse_sh));
    PHASE(routed_out(c, mt.sv[2], mt.out_signs, fuse_sh ? mt.shared[2].svh : nullptr));
    return true;
}

static __device__ sizes for_layer(const sizes& s, u32 l) {
    sizes ls = s;
    if (!bit(s.attn_split, l)) ls.heads = s.heads_all;
    if (!bit(s.mlp_split, l)) {
        ls.dense_ffn = s.dense_all;
        ls.ffn = s.ffn_all;
        ls.held_first = 0u;
        ls.held = s.experts;
    }
    return ls;
}

static __device__ bool layer(const ctx& c, const place& pl) {
    const sizes& s = c.s;
    bool split = bit(s.mlp_split, pl.layer);
    bool mark = pl.layer == 0u || pl.layer == s.dense;
    if (mark) {
        TS_MARK();
    }
    if (!attention(c, pl)) return false;
    if (mark) {
        TS_MARK();
    }
    if (pl.layer < s.dense) {
        PHASE(rms(c, c.p.x, s.hidden, field_mat(c, pl, MLP_NORM), c.p.h, s.hidden, s.rows,
                  s.hidden, nullptr, false));
        bool ok = swiglu(c, pl, D_GATE, s.dense_ffn) && add(c, split);
        if (mark) {
            TS_MARK();
        }
        return ok;
    }
    bool fuse_mlp_norm = s.rows == 1u && s.hidden <= 6144u && (s.hidden & 31u) == 0u
                         && (reinterpret_cast<u64>(c.p.x) & 15ull) == 0ull
                         && (reinterpret_cast<u64>(c.p.h) & 15ull) == 0ull;
    if (!fuse_mlp_norm) {
        PHASE(rms(c, c.p.x, s.hidden, field_mat(c, pl, MLP_NORM), c.p.h, s.hidden, s.rows,
                  s.hidden, nullptr, false));
        load_moe_meta(c, pl);
    }
    bool fuse_sh = !fuse_mlp_norm && (c.sm.meta.flags & 2u) != 0u;
    bool fast_routed = false;
    if (!route(c, pl, fuse_sh, fuse_mlp_norm, fuse_mlp_norm, &fast_routed)) return false;
    if (fuse_mlp_norm && !fast_routed) load_moe_meta(c, pl);
    fuse_sh = (c.sm.meta.flags & 2u) != 0u;
    if (mark) {
        TS_MARK();
    }
    if (!fuse_sh) {
        if (!swiglu(c, pl, S_GATE, s.ffn)) return false;
        if (mark) {
            TS_MARK();
        }
    }
    u32 moe_cnt = 0xFFFFFFFFu;
    ticket pre_r;
    bool has_pre = false;
    if (!routed(c, pl, fuse_sh, split, fast_routed, &moe_cnt, &pre_r, &has_pre)) return false;
    if (mark) {
        TS_MARK();
    }
    return add(c, split, moe_cnt, has_pre ? &pre_r : nullptr);
}
