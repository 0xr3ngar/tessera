// One all-reduce round over the pinned slots verbs.rs lays out.

struct Ctl {
  volatile unsigned long long seq;
  volatile unsigned long long bytes[2];
  volatile unsigned long long err;
  unsigned blocks;
  unsigned pad;
  volatile unsigned long long go;
};

// out = the sum of every rank's in, in rank order.
extern "C" __global__ void all_reduce(unsigned long long send_at, unsigned long long recv_at,
                                      unsigned long long flags_at, unsigned long long ctl_at,
                                      const float* in, float* out, unsigned n, unsigned rank,
                                      unsigned ranks, unsigned lanes, unsigned long long seq,
                                      unsigned long long capacity) {
  const unsigned par = (unsigned)(seq & 1);
  char* send = (char*)send_at + par * capacity;
  float* mine = (float*)(send + 16);
  const volatile unsigned long long* flags = (const volatile unsigned long long*)flags_at;
  Ctl* ctl = (Ctl*)ctl_at;
  const unsigned stride = gridDim.x * blockDim.x;
  const unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
  for (unsigned i = t; i < n; i += stride) mine[i] = in[i];
  __threadfence_system();
  __syncthreads();
  // The last block to finish copying publishes the round.
  if (threadIdx.x == 0 && atomicAdd(&ctl->blocks, 1u) == gridDim.x - 1) {
    ctl->blocks = 0;
    *(volatile unsigned long long*)send = (unsigned long long)n * 4;
    ctl->bytes[par] = (unsigned long long)n * 4;
    __threadfence_system();
    ctl->seq = seq;
  }
  // Block 0 polls the flags. The others wait on `go`.
  if (blockIdx.x == 0) {
    if (threadIdx.x < ranks * lanes && threadIdx.x / lanes != rank)
      while (flags[threadIdx.x] < seq && !ctl->err) {
      }
    __syncthreads();
    if (threadIdx.x == 0) {
      __threadfence_system();
      ctl->go = seq;
    }
  } else {
    if (threadIdx.x == 0)
      while (ctl->go < seq && !ctl->err) {
      }
    __syncthreads();
  }
  if (ctl->err) return;
  __threadfence_system();
  const char* recv = (const char*)recv_at + (size_t)par * ranks * capacity + 16;
  for (unsigned i = t; i < n; i += stride) {
    float acc = 0.0f;
    for (unsigned src = 0; src < ranks; ++src) {
      const volatile float* slot =
          src == rank ? mine : (const volatile float*)(recv + (size_t)src * capacity);
      acc += slot[i];
    }
    out[i] = acc;
  }
}
