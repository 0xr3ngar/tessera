// GLM-5.3's whole step in one persistent grid, sums inside.
#include <tessera/1/all_reduce>
#include <tessera/1/grid>
#include <tessera/1/kept>
#include <tessera/1/trellis>
#include "math.cuh"
#include "trellis.cuh"
#include "gemv.cuh"
#include "attn.cuh"
#include "moe.cuh"

// Vectorized embedding lookup for F32, BF16, and quantized tables.
static __device__ void embed_rows(const mat& embed, const u32* tokens, float* x, u32 rows,
                                  u32 hidden) {
    line l = threads();
    bool x_vec = (reinterpret_cast<u64>(x) & 15u) == 0u && (hidden & 3u) == 0u;
    if (embed.klass == K_BF16 && x_vec && (reinterpret_cast<u64>(embed.p) & 7u) == 0u) {
        const u16* tbl = reinterpret_cast<const u16*>(embed.p);
        u32 quads = hidden >> 2;
        for (u32 u = l.at; u < rows * quads; u += l.step) {
            u32 r = u / quads, q = u % quads;
            uint2 h = *reinterpret_cast<const uint2*>(tbl + u64(tokens[r]) * hidden + (q << 2));
            *reinterpret_cast<float4*>(x + u64(r) * hidden + (q << 2)) = make_float4(
                __uint_as_float(h.x << 16), __uint_as_float(h.x & 0xFFFF0000u),
                __uint_as_float(h.y << 16), __uint_as_float(h.y & 0xFFFF0000u));
        }
        return;
    }
    if (embed.klass == K_F32 && x_vec && (reinterpret_cast<u64>(embed.p) & 15u) == 0u) {
        const float* tbl = reinterpret_cast<const float*>(embed.p);
        u32 quads = hidden >> 2;
        for (u32 u = l.at; u < rows * quads; u += l.step) {
            u32 r = u / quads, q = u % quads;
            *reinterpret_cast<float4*>(x + u64(r) * hidden + (q << 2)) =
                *reinterpret_cast<const float4*>(tbl + u64(tokens[r]) * hidden + (q << 2));
        }
        return;
    }
    for (u32 i = l.at; i < rows * hidden; i += l.step) {
        x[i] = elem(embed, 0u, tokens[i / hidden], i % hidden);
    }
}

// Persistent grid entry: embed, decoder layers, final norm, logits.
extern "C" __global__ __launch_bounds__(256, 2) void glm53_whole(const table* t) {
    if (!check(t)) return;
    __shared__ Smem smem;
    if (threadIdx.x == 0u) {
        smem.s0 = read(t);
        smem.p = lay_out(t, smem.s0);
    }
    __syncthreads();
    const sizes& s = smem.s0;
    const spans_t& p = smem.p;
    session se = begin(t);
    ctx c = {t, s, p, se, smem};
    u32 norm = IDS + (is_turned(t, IDS) ? 3u : 1u);
    u32 head = norm + (is_turned(t, norm) ? 3u : 1u);
    u32 first = head + (is_turned(t, head) ? 3u : 1u);
    TS_MARK();
    embed_rows(weight(t, IDS), at<const u32>(t, 0), p.x, s.rows, s.hidden);
    if (!sync(t)) return;
    TS_MARK();
    u32 from = first, kept_rows = 0u;
    for (u32 n = 0; n < s.layers; n++) {
        if (threadIdx.x < 128u) {
            u32 it = from + threadIdx.x;
            smem.wk.turned[threadIdx.x] = (it < t->items_len && is_turned(t, it)) ? 1u : 0u;
        }
        __syncthreads();
        if (threadIdx.x == 0u) {
            smem.ls = for_layer(s, n);
            smem.pl = walk_fast(t, smem.ls, n, from, kept_rows, smem.wk.turned);
        }
        __syncthreads();
        ctx lc = {t, smem.ls, p, se, smem};
        if (!layer(lc, smem.pl)) return;
        from = smem.pl.next;
        kept_rows += indexes(s, n) ? 3u : 2u;
    }
    TS_MARK();
    rms(c, p.x, s.hidden, weight(t, norm), p.h, s.hidden, s.picks, s.hidden,
        at<const u32>(t, 3), false);
    if (!sync(t)) return;
    TS_MARK();
    if (!linear(c, weight(t, head), p.h, s.hidden, at<float>(t, from + kept_rows), s.vocab, s.picks,
                s.vocab)) {
        return;
    }
    TS_MARK();
    end(se);
}
