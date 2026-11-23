// MLA indexer, RoPE, KV absorb, block-parallel attend, and value projection.

static __device__ __forceinline__ u32 float_key(float v) {
    if (v == 0.0f) v = 0.0f;
    u32 b = __float_as_uint(v);
    return (b & 0x80000000u) ? ~b : (b ^ 0x80000000u);
}

static __device__ __forceinline__ u32 block_count(const ctx& c, u32 local) {
    u32 lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
#pragma unroll
    for (u32 off = 16; off > 0; off >>= 1) local += __shfl_xor_sync(0xFFFFFFFFu, local, off);
    if (lane == 0u) c.sm.sel.cnt[warp] = local;
    __syncthreads();
    u32 sum = 0u, nw = blockDim.x >> 5;
    for (u32 w = 0; w < nw; w++) sum += c.sm.sel.cnt[w];
    __syncthreads();
    return sum;
}

static __device__ __forceinline__ void block_prefix(const ctx& c, bool flag, u32* pos_out,
                                                    u32* tot_out) {
    u32 lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    u32 mask = __ballot_sync(0xFFFFFFFFu, flag);
    u32 lane_off = __popc(mask & ((1u << lane) - 1u));
    if (lane == 0u) c.sm.sel.cnt[warp] = __popc(mask);
    __syncthreads();
    u32 warp_off = 0u, total = 0u, nw = blockDim.x >> 5;
    for (u32 w = 0; w < nw; w++) {
        u32 wc = c.sm.sel.cnt[w];
        if (w < warp) warp_off += wc;
        total += wc;
    }
    __syncthreads();
    *pos_out = warp_off + lane_off;
    *tot_out = total;
}

// Centered RMS norm plus bias, stored into kept index table.
static __device__ void index_keys(const ctx& c, const place& pl) {
    const sizes& s = c.s;
    const u32* pos = at<const u32>(c.t, 1);
    const u32* slot = at<const u32>(c.t, 2);
    const kept* kt = kept_table(c.t, pl.kept + 2u);
    mat knorm = field_mat(c, pl, K_NORM);
    mat kbias = field_mat(c, pl, K_BIAS);
    for (u32 r = blockIdx.x; r < s.rows; r += gridDim.x) {
        const float* row = c.p.b + u64(r) * s.idim;
        float sum = 0.0f;
        for (u32 d = threadIdx.x; d < s.idim; d += blockDim.x) sum += row[d];
        float mean = block_sum(c, sum) / float(s.idim);
        float sq = 0.0f;
        for (u32 d = threadIdx.x; d < s.idim; d += blockDim.x) {
            float v = row[d] - mean;
            sq = fmaf(v, v, sq);
        }
        float inv = 1.0f / sqrtf(block_sum(c, sq) / float(s.idim) + s.eps);
        u8* dst = kept_row(c.t, kt, slot[r], pos[r]);
        for (u32 d = threadIdx.x; d < s.idim; d += blockDim.x) {
            float out = (row[d] - mean) * (inv * vec_elem(knorm, 0u, d)) + vec_elem(kbias, 0u, d);
            kv_store(dst, kt->elem, d, out);
        }
    }
}

// Warp-parallel causal indexer scores for rows exceeding keep.
static __device__ void index_scores(const ctx& c, const place& pl) {
    const sizes& s = c.s;
    const u32* pos = at<const u32>(c.t, 1);
    const u32* slot = at<const u32>(c.t, 2);
    const kept* kt = kept_table(c.t, pl.kept + 2u);
    u32 wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    u32 wstep = (gridDim.x * blockDim.x) >> 5;
    u32 lane = threadIdx.x & 31u;
    u32 cursor = 0u;
    for (u32 r = 0; r < s.rows; r++) {
        u32 lim = min(pos[r] + 1u, s.context);
        if (lim <= s.keep) continue;
        u32 rem = cursor % wstep;
        u32 first = wid >= rem ? (wid - rem) : (wid + wstep - rem);
        const float* qr = c.p.a + u64(r) * s.ih * s.idim;
        const float* wr = c.p.c + u64(r) * s.ih;
        float* dr = c.p.d + u64(r) * s.context;
        u32 sl = slot[r];
        for (u32 j = first; j < lim; j += wstep) {
            const u8* krow = kept_row(c.t, kt, sl, j);
            float score = 0.0f;
            if (s.idim <= 128u) {
                float kr[4];
#pragma unroll
                for (u32 t = 0; t < 4; t++) {
                    u32 d = lane + (t << 5);
                    kr[t] = d < s.idim ? kv_load(krow, kt->elem, d) : 0.0f;
                }
                for (u32 h = 0; h < s.ih; h++) {
                    const float* qh = qr + u64(h) * s.idim;
                    float hit = 0.0f;
#pragma unroll
                    for (u32 t = 0; t < 4; t++) {
                        u32 d = lane + (t << 5);
                        if (d < s.idim) hit = fmaf(qh[d], kr[t], hit);
                    }
                    hit = warp_sum(hit);
                    if (lane == 0u) score = fmaf(fmaxf(hit, 0.0f), wr[h], score);
                }
            } else {
                for (u32 h = 0; h < s.ih; h++) {
                    const float* qh = qr + u64(h) * s.idim;
                    float hit = 0.0f;
                    for (u32 d = lane; d < s.idim; d += 32u) {
                        hit = fmaf(qh[d], kv_load(krow, kt->elem, d), hit);
                    }
                    hit = warp_sum(hit);
                    if (lane == 0u) score = fmaf(fmaxf(hit, 0.0f), wr[h], score);
                }
            }
            if (lane == 0u) dr[j] = score;
        }
        cursor += lim;
    }
}

// Block-parallel exact top-k selection with lower-index tie break.
static __device__ void select_row(const ctx& c, const float* row, u32 lim, u32 keep, u32 ctx_len,
                                  u32* out) {
    if (lim <= keep) {
        for (u32 j = threadIdx.x; j < keep; j += blockDim.x) out[j] = j < lim ? j : ctx_len;
        return;
    }
    float neg_inf = __int_as_float(0xFF800000u);
    u32 my_valid = 0u;
    for (u32 j = threadIdx.x; j < lim; j += blockDim.x) {
        if (row[j] > neg_inf) my_valid++;
    }
    u32 valid_total = block_count(c, my_valid);
    if (valid_total <= keep) {
        u32 off = 0u;
        for (u32 base = 0; base < lim; base += blockDim.x) {
            u32 j = base + threadIdx.x;
            bool ok = j < lim && row[j] > neg_inf;
            u32 p, tot;
            block_prefix(c, ok, &p, &tot);
            if (ok) out[off + p] = j;
            off += tot;
        }
        for (u32 j = valid_total + threadIdx.x; j < keep; j += blockDim.x) out[j] = ctx_len;
        return;
    }
    u32 prefix = 0u;
    for (int b = 31; b >= 0; b--) {
        u32 cand = prefix | (1u << b);
        u32 cnt = 0u;
        for (u32 j = threadIdx.x; j < lim; j += blockDim.x) {
            float v = row[j];
            if (v > neg_inf && float_key(v) >= cand) cnt++;
        }
        if (block_count(c, cnt) >= keep) prefix = cand;
    }
    u32 my_gt = 0u;
    for (u32 j = threadIdx.x; j < lim; j += blockDim.x) {
        float v = row[j];
        if (v > neg_inf && float_key(v) > prefix) my_gt++;
    }
    u32 gt_total = block_count(c, my_gt);
    u32 eq_need = keep - gt_total;
    u32 gt_off = 0u, eq_off = 0u;
    for (u32 base = 0; base < lim; base += blockDim.x) {
        u32 j = base + threadIdx.x;
        float v = j < lim ? row[j] : neg_inf;
        bool ok = j < lim && v > neg_inf;
        u32 key = ok ? float_key(v) : 0u;
        bool is_gt = ok && key > prefix;
        bool is_eq = ok && key == prefix;
        u32 pg, tg, pe, te;
        block_prefix(c, is_gt, &pg, &tg);
        block_prefix(c, is_eq, &pe, &te);
        if (is_gt) out[gt_off + pg] = j;
        if (is_eq && eq_off + pe < eq_need) out[gt_total + eq_off + pe] = j;
        gt_off += tg;
        eq_off += te;
    }
}

static __device__ void select_top(const ctx& c) {
    const sizes& s = c.s;
    const u32* pos = at<const u32>(c.t, 1);
    for (u32 r = blockIdx.x; r < s.rows; r += gridDim.x) {
        u32 lim = min(pos[r] + 1u, s.context);
        select_row(c, c.p.d + u64(r) * s.context, lim, s.keep, s.context,
                   c.p.picked + u64(r) * s.keep);
    }
}

static __device__ void fill_direct(const ctx& c) {
    const sizes& s = c.s;
    const u32* pos = at<const u32>(c.t, 1);
    line l = threads();
    for (u32 u = l.at; u < s.rows * s.keep; u += l.step) {
        u32 r = u / s.keep, j = u % s.keep;
        u32 lim = min(pos[r] + 1u, s.context);
        c.p.picked[u] = j < lim ? j : s.context;
    }
}

static __device__ bool pick_keys(const ctx& c, const place& pl) {
    const sizes& s = c.s;
    u32 r = s.rows;
    const u32* pos = at<const u32>(c.t, 1);
    bool need_sel = false;
    for (u32 row = 0; row < r; row++) {
        if (min(pos[row] + 1u, s.context) > s.keep) need_sel = true;
    }
    if (!need_sel) {
        if (!linear(c, field_mat(c, pl, WK), c.p.h, s.hidden, c.p.b, s.idim, r, s.idim)) {
            return false;
        }
        PHASE(index_keys(c, pl));
        return true;
    }
    if (!linear(c, field_mat(c, pl, WQ), c.p.qr, s.q_latent, c.p.a, s.ih * s.idim, r,
                s.ih * s.idim)) {
        return false;
    }
    if (!linear_pair(c, field_mat(c, pl, WK), field_mat(c, pl, I_WEIGHTS), c.p.h, s.hidden, c.p.b,
                     s.idim, s.idim, c.p.c, s.ih, s.ih, r)) {
        return false;
    }
    PHASE(index_keys(c, pl));
    PHASE(index_scores(c, pl));
    PHASE(select_top(c));
    return true;
}

// Parallel RoPE on queries and key, plus kept stores.
static __device__ void turn_keys(const ctx& c, const place& pl) {
    const sizes& s = c.s;
    line l = threads();
    const u32* pos = at<const u32>(c.t, 1);
    const u32* slot = at<const u32>(c.t, 2);
    const kept* kt_l = kept_table(c.t, pl.kept);
    const kept* kt_r = kept_table(c.t, pl.kept + 1u);
    u32 pairs = s.rope >> 1, q_stride = s.nope + s.rope, kv_stride = s.latent + s.rope;
    for (u32 u = l.at; u < s.rows * s.heads * pairs; u += l.step) {
        u32 j = u % pairs, rh = u / pairs, r = rh / s.heads;
        float angle = float(pos[r]) * powf(s.theta, -float(j << 1) / float(s.rope));
        float sn, cs;
        sincosf(angle, &sn, &cs);
        float* v = c.p.a + u64(rh) * q_stride + s.nope + (j << 1);
        float a = v[0], b = v[1];
        v[0] = a * cs - b * sn;
        v[1] = a * sn + b * cs;
    }
    for (u32 u = l.at; u < s.rows * pairs; u += l.step) {
        u32 r = u / pairs, j = u % pairs;
        float angle = float(pos[r]) * powf(s.theta, -float(j << 1) / float(s.rope));
        float sn, cs;
        sincosf(angle, &sn, &cs);
        float* kv = c.p.kv + u64(r) * kv_stride + s.latent + (j << 1);
        float a = kv[0], b = kv[1];
        float r0 = a * cs - b * sn, r1 = a * sn + b * cs;
        kv[0] = r0;
        kv[1] = r1;
        u8* kr = kept_row(c.t, kt_r, slot[r], pos[r]);
        kv_store(kr, kt_r->elem, j << 1, r0);
        kv_store(kr, kt_r->elem, (j << 1) + 1u, r1);
    }
    for (u32 u = l.at; u < s.rows * s.latent; u += l.step) {
        u32 r = u / s.latent, d = u % s.latent;
        u8* kl = kept_row(c.t, kt_l, slot[r], pos[r]);
        kv_store(kl, kt_l->elem, d, c.p.kv[u64(r) * kv_stride + d]);
    }
}

// Block-parallel query absorption into latent space, coalesced along latent.
static __device__ void absorb(const ctx& c, const mat& kv_b, bool fuse_q_rope = false) {
    const sizes& s = c.s;
    u32 q_stride = s.nope + s.rope, width = s.latent + s.rope;
    if (kv_b.klass == K_BF16 && (s.latent & 127u) == 0u && s.nope <= ATTN_STAGE
        && (reinterpret_cast<u64>(kv_b.p) & 7ull) == 0ull && (kv_b.in & 3u) == 0u) {
        u32 ltiles = s.latent >> 7;
        u32 grp = threadIdx.x >> 5, tid = threadIdx.x & 31u;
        u32 quad_in = kv_b.in >> 2;
        const uint2* base_w = reinterpret_cast<const uint2*>(kv_b.p);
        u32 pos0 = fuse_q_rope ? at<const u32>(c.t, 1)[0] : 0u;
        for (u32 job = blockIdx.x; job < s.rows * s.heads * ltiles; job += gridDim.x) {
            u32 rh = job / ltiles, tile = job % ltiles, h = rh % s.heads;
            const float* qa = c.p.a + u64(rh) * q_stride;
            float* qb = c.p.b + u64(rh) * width;
            for (u32 n = threadIdx.x; n < s.nope; n += blockDim.x) c.sm.attn.q[n] = qa[n];
            if (tile == 0u) {
                if (fuse_q_rope) {
                    u32 pairs = s.rope >> 1;
                    if (threadIdx.x < pairs) {
                        u32 j = threadIdx.x;
                        float angle = float(pos0) * powf(s.theta, -float(j << 1) / float(s.rope));
                        float sn, cs;
                        sincosf(angle, &sn, &cs);
                        float a = qa[s.nope + (j << 1)], b = qa[s.nope + (j << 1) + 1u];
                        qb[s.latent + (j << 1)] = a * cs - b * sn;
                        qb[s.latent + (j << 1) + 1u] = a * sn + b * cs;
                    }
                } else {
                    for (u32 d = threadIdx.x; d < s.rope; d += blockDim.x) {
                        qb[s.latent + d] = qa[s.nope + d];
                    }
                }
            }
            __syncthreads();
            u32 base_row = h * (s.nope + s.hd);
            u32 col4 = (tile << 5) + tid;
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            for (u32 n = grp; n < s.nope; n += 8u) {
                uint2 p = base_w[u64(base_row + n) * quad_in + col4];
                float qv = c.sm.attn.q[n];
                a0 = fmaf(qv, __uint_as_float(p.x << 16), a0);
                a1 = fmaf(qv, __uint_as_float(p.x & 0xFFFF0000u), a1);
                a2 = fmaf(qv, __uint_as_float(p.y << 16), a2);
                a3 = fmaf(qv, __uint_as_float(p.y & 0xFFFF0000u), a3);
            }
            *reinterpret_cast<float4*>(&c.sm.attn.part[grp][tid << 2]) = make_float4(a0, a1, a2, a3);
            __syncthreads();
            if (threadIdx.x < 128u) {
                u32 ci = threadIdx.x;
                float sum = (c.sm.attn.part[0][ci] + c.sm.attn.part[1][ci])
                            + (c.sm.attn.part[2][ci] + c.sm.attn.part[3][ci])
                            + (c.sm.attn.part[4][ci] + c.sm.attn.part[5][ci])
                            + (c.sm.attn.part[6][ci] + c.sm.attn.part[7][ci]);
                qb[(tile << 7) + ci] = sum;
            }
            __syncthreads();
        }
        return;
    }
    bool stage = s.nope <= ATTN_STAGE;
    for (u32 rh = blockIdx.x; rh < s.rows * s.heads; rh += gridDim.x) {
        u32 h = rh % s.heads;
        const float* qa = c.p.a + u64(rh) * q_stride;
        float* qb = c.p.b + u64(rh) * width;
        if (stage) {
            for (u32 n = threadIdx.x; n < s.nope; n += blockDim.x) c.sm.attn.q[n] = qa[n];
            __syncthreads();
        }
        const float* q_src = stage ? c.sm.attn.q : qa;
        for (u32 d = threadIdx.x; d < s.rope; d += blockDim.x) qb[s.latent + d] = qa[s.nope + d];
        u32 base_row = h * (s.nope + s.hd);
        for (u32 i = threadIdx.x; i < s.latent; i += blockDim.x) {
            float acc = 0.0f;
            for (u32 n = 0; n < s.nope; n++) {
                acc = fmaf(q_src[n], vec_elem(kv_b, base_row + n, i), acc);
            }
            qb[i] = acc;
        }
        if (stage) __syncthreads();
    }
}

// Block-parallel online softmax attention, single or split-K.
static __device__ void attend_pass(const ctx& c, const place& pl, u32 splits) {
    const sizes& s = c.s;
    const u32* pos = at<const u32>(c.t, 1);
    const u32* slot = at<const u32>(c.t, 2);
    const kept* kt_l = kept_table(c.t, pl.kept);
    const kept* kt_r = kept_table(c.t, pl.kept + 1u);
    u32 width = s.latent + s.rope, rh_cnt = s.rows * s.heads, stride = s.latent + 2u;
    bool stage = width <= ATTN_STAGE;
    bool in_reg = s.latent <= 1024u && (s.latent & 3u) == 0u;
    u32 d4 = threadIdx.x << 2, rope_vec = s.rope & ~3u;
    for (u32 job = blockIdx.x; job < rh_cnt * splits; job += gridDim.x) {
        u32 rh = splits > 1u ? (job / splits) : job;
        u32 sp = splits > 1u ? (job % splits) : 0u;
        u32 r = rh / s.heads, h = rh % s.heads;
        u32 pr = pos[r], sl = slot[r];
        const float* qg = c.p.b + u64(rh) * width;
        float* og = c.p.c + u64(rh) * s.latent;
        if (stage) {
            for (u32 d = threadIdx.x; d < width; d += blockDim.x) c.sm.attn.q[d] = qg[d];
            __syncthreads();
        }
        const float* q = stage ? c.sm.attn.q : qg;
        bool q_vec = (reinterpret_cast<u64>(q) & 15u) == 0u;
        bool rq_vec = q_vec && (s.latent & 3u) == 0u;
        float q_reg[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float o_reg[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        if (in_reg) {
            if (d4 < s.latent) load_vec4(q + d4, q_vec, q_reg);
        } else {
            for (u32 d = threadIdx.x; d < s.latent; d += blockDim.x) og[d] = 0.0f;
        }
        float most = __int_as_float(0xFF800000u), total = 0.0f;
        u32 lim = min(pr + 1u, s.context);
        bool direct = lim <= s.keep;
        u32 n_end = direct ? lim : s.keep;
        for (u32 n = sp; n < n_end; n += splits) {
            u32 j = direct ? n : c.p.picked[u64(r) * s.keep + n];
            if (!direct && (j >= s.context || j > pr)) continue;
            const u8* kl = kept_row(c.t, kt_l, sl, j);
            const u8* kr = kept_row(c.t, kt_r, sl, j);
            float dot = 0.0f, kv_reg[4];
            if (in_reg) {
                if (d4 < s.latent) {
                    kv_load4(kl, kt_l->elem, d4, kv_reg);
#pragma unroll
                    for (u32 k = 0; k < 4; k++) dot = fmaf(q_reg[k], kv_reg[k], dot);
                }
            } else {
                for (u32 d = threadIdx.x; d < s.latent; d += blockDim.x) {
                    dot = fmaf(q[d], kv_load(kl, kt_l->elem, d), dot);
                }
            }
            for (u32 d = d4; d < rope_vec; d += blockDim.x << 2) {
                float rv[4], qr[4];
                kv_load4(kr, kt_r->elem, d, rv);
                load_vec4(q + s.latent + d, rq_vec, qr);
#pragma unroll
                for (u32 k = 0; k < 4; k++) dot = fmaf(qr[k], rv[k], dot);
            }
            for (u32 d = rope_vec + threadIdx.x; d < s.rope; d += blockDim.x) {
                dot = fmaf(q[s.latent + d], kv_load(kr, kt_r->elem, d), dot);
            }
            float score = block_sum(c, dot) * s.scale;
            float topmost = fmaxf(most, score);
            float shrink = expf(most - topmost), e = expf(score - topmost);
            total = total * shrink + e;
            most = topmost;
            if (in_reg) {
                if (d4 < s.latent) {
#pragma unroll
                    for (u32 k = 0; k < 4; k++) o_reg[k] = fmaf(o_reg[k], shrink, e * kv_reg[k]);
                }
            } else {
                for (u32 d = threadIdx.x; d < s.latent; d += blockDim.x) {
                    og[d] = fmaf(og[d], shrink, e * kv_load(kl, kt_l->elem, d));
                }
            }
        }
        if (splits > 1u) {
            float* dst = c.p.t1 + u64(r) * s.width + u64(h) * splits * stride + u64(sp) * stride;
            if (threadIdx.x == 0u) {
                dst[0] = most;
                dst[1] = total;
            }
            if (d4 < s.latent) {
#pragma unroll
                for (u32 k = 0; k < 4; k++) dst[2u + d4 + k] = o_reg[k];
            }
        } else if (in_reg) {
            if (d4 < s.latent) {
                float inv = 1.0f / total;
                float out[4] = {o_reg[0] * inv, o_reg[1] * inv, o_reg[2] * inv, o_reg[3] * inv};
                store_vec4(og + d4, (reinterpret_cast<u64>(og) & 15u) == 0u, out);
            }
        } else {
            for (u32 d = threadIdx.x; d < s.latent; d += blockDim.x) og[d] /= total;
        }
        if (stage) __syncthreads();
    }
}

// Merges split-K attention partials with exact online softmax.
static __device__ void attend_merge(const ctx& c, u32 splits) {
    const sizes& s = c.s;
    u32 rh_cnt = s.rows * s.heads, stride = s.latent + 2u;
    float neg_inf = __int_as_float(0xFF800000u);
    for (u32 rh = blockIdx.x; rh < rh_cnt; rh += gridDim.x) {
        u32 r = rh / s.heads, h = rh % s.heads;
        const float* base = c.p.t1 + u64(r) * s.width + u64(h) * splits * stride;
        float* og = c.p.c + u64(rh) * s.latent;
        float m_all = neg_inf;
        for (u32 sp = 0; sp < splits; sp++) {
            float tot = base[u64(sp) * stride + 1u];
            if (tot > 0.0f) m_all = fmaxf(m_all, base[u64(sp) * stride]);
        }
        float sum_t = 0.0f, w[8];
        for (u32 sp = 0; sp < splits; sp++) {
            float tot = base[u64(sp) * stride + 1u];
            if (tot > 0.0f) {
                float ms = base[u64(sp) * stride];
                float sc = ms == m_all ? 1.0f : expf(ms - m_all);
                w[sp] = sc;
                sum_t = fmaf(tot, sc, sum_t);
            } else {
                w[sp] = 0.0f;
            }
        }
        float inv = sum_t > 0.0f ? (1.0f / sum_t) : 0.0f;
        for (u32 d = threadIdx.x; d < s.latent; d += blockDim.x) {
            float acc = 0.0f;
            for (u32 sp = 0; sp < splits; sp++) {
                if (w[sp] != 0.0f) acc = fmaf(base[u64(sp) * stride + 2u + d], w[sp], acc);
            }
            og[d] = acc * inv;
        }
    }
}

// Block-local 8-warp split-K attention for decode.
static __device__ __noinline__ void attend_warp8(const ctx& c, const place& pl, u32 lim) {
    const sizes& s = c.s;
    const u32* slot = at<const u32>(c.t, 2);
    const kept* kt_l = kept_table(c.t, pl.kept);
    const kept* kt_r = kept_table(c.t, pl.kept + 1u);
    u32 sl = slot[0];
    u32 warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    float neg_inf = __int_as_float(0xFF800000u);
    for (u32 h = blockIdx.x; h < s.heads; h += gridDim.x) {
        const float* qg = c.p.b + u64(h) * 576u;
        float* og = c.p.c + u64(h) * 512u;
        float q_reg[4][4], o_reg[4][4];
#pragma unroll
        for (u32 k = 0; k < 4u; k++) {
            float4 qv = *reinterpret_cast<const float4*>(qg + (k << 7) + (lane << 2));
            q_reg[k][0] = qv.x;
            q_reg[k][1] = qv.y;
            q_reg[k][2] = qv.z;
            q_reg[k][3] = qv.w;
            o_reg[k][0] = o_reg[k][1] = o_reg[k][2] = o_reg[k][3] = 0.0f;
        }
        float qr0 = qg[512u + lane], qr1 = qg[544u + lane];
        float most = neg_inf, total = 0.0f;
        for (u32 j = warp; j < lim; j += WARPS) {
            const u8* kl = kept_row(c.t, kt_l, sl, j);
            const u8* kr = kept_row(c.t, kt_r, sl, j);
            float dot = 0.0f, kv[4][4];
#pragma unroll
            for (u32 k = 0; k < 4u; k++) {
                kv_load4(kl, kt_l->elem, (k << 7) + (lane << 2), kv[k]);
                dot = fmaf(q_reg[k][0], kv[k][0], dot);
                dot = fmaf(q_reg[k][1], kv[k][1], dot);
                dot = fmaf(q_reg[k][2], kv[k][2], dot);
                dot = fmaf(q_reg[k][3], kv[k][3], dot);
            }
            dot = fmaf(qr0, kv_load(kr, kt_r->elem, lane), dot);
            dot = fmaf(qr1, kv_load(kr, kt_r->elem, lane + 32u), dot);
            float score = warp_sum(dot) * s.scale;
            float topmost = fmaxf(most, score);
            float shrink = __expf(most - topmost), e = __expf(score - topmost);
            total = total * shrink + e;
            most = topmost;
#pragma unroll
            for (u32 k = 0; k < 4u; k++) {
                o_reg[k][0] = fmaf(o_reg[k][0], shrink, e * kv[k][0]);
                o_reg[k][1] = fmaf(o_reg[k][1], shrink, e * kv[k][1]);
                o_reg[k][2] = fmaf(o_reg[k][2], shrink, e * kv[k][2]);
                o_reg[k][3] = fmaf(o_reg[k][3], shrink, e * kv[k][3]);
            }
        }
        if (lane == 0u) {
            c.sm.mla.m[warp] = most;
            c.sm.mla.tot[warp] = total;
        }
#pragma unroll
        for (u32 k = 0; k < 4u; k++) {
            *reinterpret_cast<float4*>(&c.sm.mla.o[warp][(k << 7) + (lane << 2)]) =
                make_float4(o_reg[k][0], o_reg[k][1], o_reg[k][2], o_reg[k][3]);
        }
        __syncthreads();
        float m_all = neg_inf;
        for (u32 sp = 0; sp < WARPS; sp++) {
            if (c.sm.mla.tot[sp] > 0.0f) m_all = fmaxf(m_all, c.sm.mla.m[sp]);
        }
        float sum_t = 0.0f, w[WARPS];
        for (u32 sp = 0; sp < WARPS; sp++) {
            float tot = c.sm.mla.tot[sp];
            if (tot > 0.0f) {
                float ms = c.sm.mla.m[sp];
                float sc = ms == m_all ? 1.0f : expf(ms - m_all);
                w[sp] = sc;
                sum_t = fmaf(tot, sc, sum_t);
            } else {
                w[sp] = 0.0f;
            }
        }
        float inv = sum_t > 0.0f ? (1.0f / sum_t) : 0.0f;
        u32 d0 = threadIdx.x, d1 = threadIdx.x + 256u;
        float a0 = 0.0f, a1 = 0.0f;
        for (u32 sp = 0; sp < WARPS; sp++) {
            if (w[sp] != 0.0f) {
                a0 = fmaf(c.sm.mla.o[sp][d0], w[sp], a0);
                a1 = fmaf(c.sm.mla.o[sp][d1], w[sp], a1);
            }
        }
        og[d0] = a0 * inv;
        og[d1] = a1 * inv;
        __syncthreads();
    }
}

static __device__ bool attend_run(const ctx& c, const place& pl) {
    const sizes& s = c.s;
    const u32* pos = at<const u32>(c.t, 1);
    if (s.rows == 1u && s.latent == 512u && s.rope == 64u) {
        u32 lim = min(pos[0] + 1u, s.context);
        if (lim <= s.keep && lim <= 256u && (reinterpret_cast<u64>(c.p.b) & 15ull) == 0ull) {
            PHASE(attend_warp8(c, pl, lim));
            return true;
        }
    }
    u32 rh_cnt = s.rows * s.heads, stride = s.latent + 2u;
    u32 max_by_scratch = s.width / max(1u, s.heads * stride);
    u32 work_n = min(s.rows == 1u ? min(pos[0] + 1u, s.context) : s.context, s.keep);
    bool can_split = work_n >= 24u && s.latent <= 1024u && (s.latent & 3u) == 0u;
    u32 splits = can_split ? min(8u, min(gridDim.x / max(1u, rh_cnt), max_by_scratch)) : 1u;
    if (splits > 1u) {
        PHASE(attend_pass(c, pl, splits));
        PHASE(attend_merge(c, splits));
        return true;
    }
    PHASE(attend_pass(c, pl, 1u));
    return true;
}

// Warp-parallel value projection, coalesced along latent.
static __device__ void head_values(const ctx& c, const mat& kv_b) {
    const sizes& s = c.s;
    u32 wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    u32 wstep = (gridDim.x * blockDim.x) >> 5;
    u32 lane = threadIdx.x & 31u;
    u32 vec_end = s.latent & ~3u;
    bool o_vec = (reinterpret_cast<u64>(c.p.c) & 15u) == 0u && (s.latent & 3u) == 0u;
    for (u32 u = wid; u < s.rows * s.heads * s.hd; u += wstep) {
        u32 rh = u / s.hd, d = u % s.hd, h = rh % s.heads;
        u32 row_idx = h * (s.nope + s.hd) + s.nope + d;
        const float* o = c.p.c + u64(rh) * s.latent;
        float acc = 0.0f;
        if (kv_b.klass == K_BF16 && o_vec) {
            const u16* wr = reinterpret_cast<const u16*>(kv_b.p) + u64(row_idx) * kv_b.in;
            if ((reinterpret_cast<u64>(wr) & 15u) == 0u && (s.latent & 7u) == 0u) {
                for (u32 i = lane << 3; i < s.latent; i += 256u) {
                    uint4 b = ld_nc_u4(wr + i);
                    float4 o0 = *reinterpret_cast<const float4*>(o + i);
                    float4 o1 = *reinterpret_cast<const float4*>(o + i + 4u);
                    acc = fmaf(__uint_as_float(b.x << 16), o0.x,
                               fmaf(__uint_as_float(b.x & 0xFFFF0000u), o0.y,
                                    fmaf(__uint_as_float(b.y << 16), o0.z,
                                         fmaf(__uint_as_float(b.y & 0xFFFF0000u), o0.w, acc))));
                    acc = fmaf(__uint_as_float(b.z << 16), o1.x,
                               fmaf(__uint_as_float(b.z & 0xFFFF0000u), o1.y,
                                    fmaf(__uint_as_float(b.w << 16), o1.z,
                                         fmaf(__uint_as_float(b.w & 0xFFFF0000u), o1.w, acc))));
                }
            } else if ((reinterpret_cast<u64>(wr) & 7u) == 0u && vec_end > 0u) {
                for (u32 i = lane << 2; i < vec_end; i += 128u) {
                    uint2 b = *reinterpret_cast<const uint2*>(wr + i);
                    float4 ov = *reinterpret_cast<const float4*>(o + i);
                    acc = fmaf(__uint_as_float(b.x << 16), ov.x, acc);
                    acc = fmaf(__uint_as_float(b.x & 0xFFFF0000u), ov.y, acc);
                    acc = fmaf(__uint_as_float(b.y << 16), ov.z, acc);
                    acc = fmaf(__uint_as_float(b.y & 0xFFFF0000u), ov.w, acc);
                }
                for (u32 i = vec_end + lane; i < s.latent; i += 32u) {
                    acc = fmaf(__uint_as_float(u32(wr[i]) << 16), o[i], acc);
                }
            } else {
                for (u32 i = lane; i < s.latent; i += 32u) {
                    acc = fmaf(__uint_as_float(u32(wr[i]) << 16), o[i], acc);
                }
            }
        } else if (kv_b.klass == K_F32 && o_vec) {
            const float* wr = reinterpret_cast<const float*>(kv_b.p) + u64(row_idx) * kv_b.in;
            if ((reinterpret_cast<u64>(wr) & 15u) == 0u && vec_end > 0u) {
                for (u32 i = lane << 2; i < vec_end; i += 128u) {
                    float4 wv = *reinterpret_cast<const float4*>(wr + i);
                    float4 ov = *reinterpret_cast<const float4*>(o + i);
                    acc = fmaf(wv.x, ov.x, acc);
                    acc = fmaf(wv.y, ov.y, acc);
                    acc = fmaf(wv.z, ov.z, acc);
                    acc = fmaf(wv.w, ov.w, acc);
                }
                for (u32 i = vec_end + lane; i < s.latent; i += 32u) acc = fmaf(wr[i], o[i], acc);
            } else {
                for (u32 i = lane; i < s.latent; i += 32u) acc = fmaf(wr[i], o[i], acc);
            }
        } else {
            for (u32 i = lane; i < s.latent; i += 32u) {
                acc = fmaf(vec_elem(kv_b, row_idx, i), o[i], acc);
            }
        }
        float sum = warp_sum(acc);
        if (lane == 0u) c.p.d[u] = sum;
    }
}

// Fused decode norms, Q_B pre-WHT, kept stores, and index_keys.
static __device__ void qkv_idx_norm(const ctx& c, const place& pl, const mat& m_qb,
                                    bool fuse_qb_turn, bool do_idx_keys, bool fuse_kv_store) {
    const sizes& s = c.s;
    if (blockIdx.x == 0u) {
        u32 n = s.q_latent;
        mat nw = field_mat(c, pl, Q_NORM);
        float inv = rms_inv(c, c.p.qr, n);
        bool nw_bf16 = nw.klass == K_BF16 && (reinterpret_cast<u64>(nw.p) & 7ull) == 0ull;
        for (u32 i = threadIdx.x << 2; i < n; i += blockDim.x << 2) {
            float4 v = *reinterpret_cast<const float4*>(c.p.qr + i);
            float4 wv = norm_w4(nw, nw_bf16, i);
            float4 out = make_float4(v.x * (inv * wv.x), v.y * (inv * wv.y), v.z * (inv * wv.z),
                                     v.w * (inv * wv.w));
            if (fuse_qb_turn) {
                if (m_qb.suh != nullptr) {
                    float4 su = *reinterpret_cast<const float4*>(m_qb.suh + i);
                    out = make_float4(out.x * su.x, out.y * su.y, out.z * su.z, out.w * su.w);
                }
                *reinterpret_cast<float4*>(c.sm.attn.q + i) = out;
            } else {
                *reinterpret_cast<float4*>(c.p.qr + i) = out;
            }
        }
        if (fuse_qb_turn) {
            __syncthreads();
            u32 warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
            u32 blocks = n >> 7;
            for (u32 blk = warp; blk < blocks; blk += WARPS) {
                u32 base = (blk << 7) + (lane << 2);
                float4 qv = *reinterpret_cast<const float4*>(c.sm.attn.q + base);
                float v[4] = {qv.x, qv.y, qv.z, qv.w};
                trellis::wht128(v, trellis::INV_SQRT128);
                *reinterpret_cast<float4*>(c.p.t1 + base) = make_float4(v[0], v[1], v[2], v[3]);
            }
            __syncthreads();
        }
    } else if (blockIdx.x == 1u) {
        u32 n = s.latent;
        mat nw = field_mat(c, pl, KV_NORM);
        float inv = rms_inv(c, c.p.kv, n);
        bool nw_bf16 = nw.klass == K_BF16 && (reinterpret_cast<u64>(nw.p) & 7ull) == 0ull;
        const kept* kt_l = fuse_kv_store ? kept_table(c.t, pl.kept) : nullptr;
        u8* kl = fuse_kv_store
                     ? kept_row(c.t, kt_l, at<const u32>(c.t, 2)[0], at<const u32>(c.t, 1)[0])
                     : nullptr;
        for (u32 i = threadIdx.x << 2; i < n; i += blockDim.x << 2) {
            float4 v = *reinterpret_cast<const float4*>(c.p.kv + i);
            float4 wv = norm_w4(nw, nw_bf16, i);
            float4 out = make_float4(v.x * (inv * wv.x), v.y * (inv * wv.y), v.z * (inv * wv.z),
                                     v.w * (inv * wv.w));
            *reinterpret_cast<float4*>(c.p.kv + i) = out;
            if (fuse_kv_store) {
                kv_store(kl, kt_l->elem, i, out.x);
                kv_store(kl, kt_l->elem, i + 1u, out.y);
                kv_store(kl, kt_l->elem, i + 2u, out.z);
                kv_store(kl, kt_l->elem, i + 3u, out.w);
            }
        }
    } else if (blockIdx.x == 2u && do_idx_keys) {
        const u32* pos = at<const u32>(c.t, 1);
        const u32* slot = at<const u32>(c.t, 2);
        const kept* kt = kept_table(c.t, pl.kept + 2u);
        mat knorm = field_mat(c, pl, K_NORM);
        mat kbias = field_mat(c, pl, K_BIAS);
        const float* row = c.p.b;
        float sum = 0.0f;
        for (u32 d = threadIdx.x; d < s.idim; d += blockDim.x) sum += row[d];
        float mean = block_sum(c, sum) / float(s.idim);
        float sq = 0.0f;
        for (u32 d = threadIdx.x; d < s.idim; d += blockDim.x) {
            float v = row[d] - mean;
            sq = fmaf(v, v, sq);
        }
        float inv = 1.0f / sqrtf(block_sum(c, sq) / float(s.idim) + s.eps);
        u8* dst = kept_row(c.t, kt, slot[0], pos[0]);
        for (u32 d = threadIdx.x; d < s.idim; d += blockDim.x) {
            float out = (row[d] - mean) * (inv * vec_elem(knorm, 0u, d)) + vec_elem(kbias, 0u, d);
            kv_store(dst, kt->elem, d, out);
        }
    } else if (blockIdx.x == 3u && fuse_kv_store) {
        u32 pairs = s.rope >> 1;
        if (threadIdx.x < pairs) {
            const u32* pos = at<const u32>(c.t, 1);
            const u32* slot = at<const u32>(c.t, 2);
            const kept* kt_r = kept_table(c.t, pl.kept + 1u);
            u32 j = threadIdx.x;
            float angle = float(pos[0]) * powf(s.theta, -float(j << 1) / float(s.rope));
            float sn, cs;
            sincosf(angle, &sn, &cs);
            float* kv = c.p.kv + s.latent + (j << 1);
            float a = kv[0], b = kv[1];
            float r0 = a * cs - b * sn, r1 = a * sn + b * cs;
            kv[0] = r0;
            kv[1] = r1;
            u8* kr = kept_row(c.t, kt_r, slot[0], pos[0]);
            kv_store(kr, kt_r->elem, j << 1, r0);
            kv_store(kr, kt_r->elem, (j << 1) + 1u, r1);
        }
    }
}

static __device__ bool attention(const ctx& c, const place& pl) {
    const sizes& s = c.s;
    u32 r = s.rows, kv = s.latent + s.rope, q = s.heads * (s.nope + s.rope);
    mat m_qa = field_mat(c, pl, Q_A);
    mat m_kva = field_mat(c, pl, KV_A);
    bool idx_layer = indexes(s, pl.layer);
    const u32* pos = at<const u32>(c.t, 1);
    bool need_sel = false;
    if (idx_layer) {
        for (u32 row = 0; row < r; row++) {
            if (min(pos[row] + 1u, s.context) > s.keep) need_sel = true;
        }
    }
    bool fused_qa = r == 1u && m_qa.klass == K_MXFP8 && m_kva.klass == K_MXFP8
                    && m_qa.in == s.hidden && m_kva.in == s.hidden && s.hidden <= 6144u
                    && (s.hidden & 1023u) == 0u && (reinterpret_cast<u64>(m_qa.p) & 15ull) == 0ull
                    && (reinterpret_cast<u64>(m_kva.p) & 15ull) == 0ull
                    && (reinterpret_cast<u64>(c.p.x) & 15ull) == 0ull;
    bool fused_wk = false;
    mat m_wk = {nullptr, K_F32, 0u, 0u, 0u, 0u, nullptr, nullptr};
    if (fused_qa && idx_layer && !need_sel) {
        m_wk = field_mat(c, pl, WK);
        if ((m_wk.klass == K_MXFP8 || m_wk.klass == K_BF16 || m_wk.klass == K_F32)
            && m_wk.in == s.hidden && (reinterpret_cast<u64>(m_wk.p) & 15ull) == 0ull) {
            fused_wk = true;
        }
    }
    if (fused_qa) {
        PHASE(gemv_mxfp8_norm(c, field_mat(c, pl, A_NORM), m_qa, m_kva, m_wk, c.p.qr, s.q_latent,
                              c.p.kv, kv, c.p.b, fused_wk ? s.idim : 0u, idx_layer && !fused_wk));
    } else {
        PHASE(rms(c, c.p.x, s.hidden, field_mat(c, pl, A_NORM), c.p.h, s.hidden, r, s.hidden,
                  nullptr, false));
        if (!linear_pair(c, m_qa, m_kva, c.p.h, s.hidden, c.p.qr, s.q_latent, s.q_latent, c.p.kv,
                         kv, kv, r)) {
            return false;
        }
    }
    mat m_qb = field_mat(c, pl, Q_B);
    mat kv_b = field_mat(c, pl, KV_B);
    bool fast_norm = r == 1u && gridDim.x >= 4u && s.q_latent <= ATTN_STAGE
                     && (s.q_latent & 127u) == 0u && (s.latent & 3u) == 0u
                     && (reinterpret_cast<u64>(c.p.qr) & 15ull) == 0ull
                     && (reinterpret_cast<u64>(c.p.kv) & 15ull) == 0ull;
    bool fuse_qb_turn = fast_norm && (!idx_layer || fused_wk) && m_qb.klass == K_TURNED
                        && (m_qb.in & 127u) == 0u && (m_qb.out & 127u) == 0u && m_qb.out == q
                        && m_qb.in == s.q_latent
                        && (m_qb.suh == nullptr || (reinterpret_cast<u64>(m_qb.suh) & 15ull) == 0ull);
    bool fuse_turn_keys = fast_norm && kv_b.klass == K_BF16 && (s.latent & 127u) == 0u
                          && s.nope <= ATTN_STAGE && (s.rope >> 1) <= blockDim.x
                          && (reinterpret_cast<u64>(kv_b.p) & 7ull) == 0ull
                          && (kv_b.in & 3u) == 0u;
    if (fast_norm) {
        PHASE(qkv_idx_norm(c, pl, m_qb, fuse_qb_turn, fused_wk, fuse_turn_keys));
    } else {
        PHASE(rms(c, c.p.qr, s.q_latent, field_mat(c, pl, Q_NORM), c.p.qr, s.q_latent, r,
                  s.q_latent, nullptr, false);
              rms(c, c.p.kv, kv, field_mat(c, pl, KV_NORM), c.p.kv, kv, r, s.latent, nullptr,
                  false));
        if (fused_wk) PHASE(index_keys(c, pl));
    }
    if (idx_layer && !fused_wk && !pick_keys(c, pl)) return false;
    if (fuse_qb_turn) {
        PHASE(trellis_run(c, m_qb, c.p.t1, m_qb.in, c.p.a, q, 1u, true));
    } else {
        if (!linear(c, m_qb, c.p.qr, s.q_latent, c.p.a, q, r, q)) return false;
    }
    if (!fuse_turn_keys) {
        PHASE(turn_keys(c, pl));
    }
    PHASE(absorb(c, kv_b, fuse_turn_keys));
    if (!attend_run(c, pl)) return false;
    PHASE(head_values(c, kv_b));
    bool attn_sp = bit(s.attn_split, pl.layer);
    if (attn_sp) {
        const lane* ln = get<const lane>(c.t, c.t->lanes);
        bool bf16 = ln->elem == BF16;
        u32 room = detail::pools(c.t)->pools[ln->pool].send_bytes >> (bf16 ? 1u : 2u);
        u32 n_elem = r * s.hidden;
        if (n_elem <= room) {
            ticket tr = prepare(c.se, 0u);
            mat wo = field_mat(c, pl, O);
            if (!bf16 && wo.klass <= K_MXFP8) {
                float* sent = send_buffer<float>(tr);
                if (sent == nullptr) return false;
                times_plain(c, wo, c.p.d, s.heads * s.hd, sent, s.hidden, r, s.hidden, 0ull);
                if (!add_finish(c, tr, 0u, n_elem, (threadIdx.x & 31u) == 0u)) return false;
                return sync(c.t);
            }
            if (bf16 && r == 1u && wo.klass == K_INT4 && (wo.in & 1023u) == 0u && wo.in <= 6144u
                && (s.hidden & 1u) == 0u && (reinterpret_cast<u64>(wo.p) & 15ull) == 0ull
                && (reinterpret_cast<u64>(c.p.d) & 15ull) == 0ull) {
                u16* sent = send_buffer<u16>(tr);
                if (sent == nullptr) return false;
                gemv_int4_2row(c, wo, c.p.d, nullptr, s.hidden, sent);
                if (!add_finish(c, tr, 0u, n_elem, (threadIdx.x & 31u) == 0u)) return false;
                return sync(c.t);
            }
            if (!linear(c, wo, c.p.d, s.heads * s.hd, c.p.part, s.hidden, r, s.hidden)) {
                return false;
            }
            return add(c, true, 0xFFFFFFFFu, &tr);
        }
    }
    if (!linear(c, field_mat(c, pl, O), c.p.d, s.heads * s.hd, c.p.part, s.hidden, r, s.hidden)) {
        return false;
    }
    return add(c, attn_sp);
}
