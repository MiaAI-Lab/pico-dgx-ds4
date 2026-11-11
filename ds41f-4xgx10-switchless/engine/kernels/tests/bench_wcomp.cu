// Lossless weight compression, Part 2 (docs/engine.md, "Lossless weight compression: a feasibility
// study"): an FP8 GEMV that decodes Huffman-coded weights inline, timed against today's kernels on
// the same tensors. EXPERIMENT: bench-only, nothing in serving calls it.
//
// The format. One canonical Huffman code per site, its codes at most 12 bits long, so one lookup in a
// 4096-entry table decodes a byte. A launch's weights are cut into runs, one per (CTA, warp) of the
// persistent GEMV's decomposition (k_gemv_fp8p: CTA b takes tiles [t0, t1), warp w the 64-byte steps
// [p0, p1) of each). Lane l decodes, step by step, the 16 bytes k_gemv_fp8p's lane l reads (bytes
// 16l..16l+15 of each 512-byte step). The 32 lanes decode in lockstep from one stream of 32-bit words:
// before every second byte, each lane holding 32 bits or fewer takes the next word (in lane order,
// by ballot), and the encoder wrote the words in that order. A run starts at a multiple of 16 bytes and
// streams through a ring of 512-byte cp.async stages in shared memory, as k_gemv_fp8p's raw bytes do.
// The runs are cut for a grid of 1, 2 or 4 CTAs per SM (more warps to hide the decoder's latency; a
// tile's arithmetic does not depend on the grid). Overhead: the run offsets ([ctas][wk] u32), up to 63
// bits per lane fetched past a run's end, the padding to 16 bytes, and the table (8 KB per site).
//
// The decoded bytes reach the MMAs exactly as k_gemv_fp8p's do, and the partial sums meet in the same
// order, so the outputs are k_gemv_fp8's with the same wk, bit for bit. `run` checks that against
// today's kernels (and the decoded bytes against the file's) before it times anything.
//
// History: the first decoder (d0610b1) checked the ring and refilled with branches around every ballot:
// 723 SASS instructions a warp-step, 1.4-3x slower than today's kernels. This one manages the ring once
// per step and refills without branches: 497. Result: exact on every layer and grid, and 10-58 % slower
// than today's kernels at one row (wo_b, wq_b, wq_a + wkv), so FP8 Huffman was dropped (docs/engine.md).
//
// usage: bench_wcomp prep RANKFILE OUTDIR [layers]   CPU only: read the sites' tensors, build the codes,
//                                                    encode, check the decode, write OUTDIR/<site>.zb
//        bench_wcomp run OUTDIR [site...] [rows...]  GPU: check, then time (rows: 1 and 6 by default)
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iterator>
#include <map>
#include <numeric>
#include <string>
#include <vector>

#include "common/check.h"
#include "kernels/common.cuh"
#include "kernels/dsk.h"

namespace {

constexpr int kSMs = 48;              // GB10: the persistent grid the runs are cut for (checked at run time)
constexpr int kCpm[] = {1, 2, 4};     // CTAs per SM the runs are cut for
constexpr int kLmax = 12;             // longest code
constexpr int kTab = 1 << kLmax;

// ------------------------------------------------------------------ the sites
struct Site {
  const char* name;
  std::vector<const char*> segs;  // tensors after "layers.L."
  bool norm;                      // the GEMV normalises its input (gemv_fp8_norm)
  int wk;                         // warps along K: the serving GemvCfg's (impl.h: c8 16, c8k 8)
};
const std::vector<Site> kSites = {
    {"wo_b", {"attn.wo_b"}, false, 16},
    {"wq_b", {"attn.wq_b"}, true, 8},
    {"wqa_wkv", {"attn.wq_a", "attn.wkv"}, true, 8},
};

// ------------------------------------------------------------------ the rank file (docs/layout.md)
struct Ent {
  uint32_t dtype = 0, layout = 0;
  uint64_t shape[4] = {}, offset = 0, bytes = 0;
};
struct RankFile {
  int fd = -1;
  std::map<std::string, Ent> toc;
  explicit RankFile(const char* path) {
    fd = open(path, O_RDONLY);
    if (fd < 0) die("open %s", path);
    uint8_t h[4096];
    rd(0, sizeof h, h);
    if (std::memcmp(h, "DSGB10W1", 8)) die("%s: not a rank file", path);
    uint32_t n;
    uint64_t toc_off;
    std::memcpy(&n, h + 24, 4);
    std::memcpy(&toc_off, h + 32, 8);
    std::vector<uint8_t> t((size_t)n * 512);
    rd(toc_off, t.size(), t.data());
    for (uint32_t i = 0; i < n; ++i) {
      const uint8_t* e = t.data() + (size_t)i * 512;
      Ent x;
      std::memcpy(&x.dtype, e + 352, 4);
      std::memcpy(x.shape, e + 360, 32);
      std::memcpy(&x.offset, e + 392, 8);
      std::memcpy(&x.bytes, e + 400, 8);
      std::memcpy(&x.layout, e + 428, 4);
      toc[std::string(reinterpret_cast<const char*>(e), strnlen(reinterpret_cast<const char*>(e), 128))] = x;
    }
  }
  // [off, off + n) into buf, then its pages dropped (rank 0 serves while this runs)
  void rd(uint64_t off, uint64_t n, uint8_t* buf) {
    for (uint64_t got = 0; got < n;) {
      ssize_t r = pread(fd, buf + got, n - got, (off_t)(off + got));
      if (r <= 0) die("short read");
      got += (uint64_t)r;
    }
    posix_fadvise(fd, (off_t)off, (off_t)n, POSIX_FADV_DONTNEED);
  }
  const Ent& at(const std::string& name) {
    auto it = toc.find(name);
    if (it == toc.end()) die("no tensor %s", name.c_str());
    return it->second;
  }
  std::vector<uint8_t> read(const std::string& name) {
    const Ent& e = at(name);
    std::vector<uint8_t> v(e.bytes);
    rd(e.offset, e.bytes, v.data());
    return v;
  }
};

// ------------------------------------------------------------------ the code
struct Code {
  uint32_t code[256];
  int len[256];
  uint16_t table[kTab];  // (len << 8) | byte, indexed by the next kLmax bits
};

// optimal code lengths of at most L bits (package-merge); every byte gets a code
std::vector<int> code_lengths(std::vector<uint64_t> cnt, int L) {
  const int n = 256;
  for (auto& c : cnt) c = std::max<uint64_t>(c, 1);
  std::vector<int> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return cnt[a] < cnt[b]; });
  struct Item {
    uint64_t w;
    std::vector<uint8_t> c;  // times each leaf (in `order`) occurs in the item
  };
  std::vector<Item> leaves(n);
  for (int k = 0; k < n; ++k) {
    leaves[k].w = cnt[order[k]];
    leaves[k].c.assign(n, 0);
    leaves[k].c[k] = 1;
  }
  std::vector<Item> cur = leaves;
  for (int lev = 1; lev < L; ++lev) {
    std::vector<Item> pk;
    for (size_t j = 0; j + 1 < cur.size(); j += 2) {
      Item it{cur[j].w + cur[j + 1].w, std::vector<uint8_t>(n)};
      for (int k = 0; k < n; ++k) it.c[k] = cur[j].c[k] + cur[j + 1].c[k];
      pk.push_back(std::move(it));
    }
    std::vector<Item> m;
    m.reserve(leaves.size() + pk.size());
    std::merge(leaves.begin(), leaves.end(), pk.begin(), pk.end(), std::back_inserter(m),
               [](const Item& a, const Item& b) { return a.w < b.w; });
    cur.swap(m);
  }
  std::vector<int> len(n, 0);
  for (int j = 0; j < 2 * n - 2; ++j)
    for (int k = 0; k < n; ++k) len[order[k]] += cur[j].c[k];
  return len;
}

Code canonical(const std::vector<int>& len) {
  Code C{};
  std::vector<int> s(256);
  std::iota(s.begin(), s.end(), 0);
  std::stable_sort(s.begin(), s.end(), [&](int a, int b) { return len[a] < len[b]; });
  uint32_t code = 0;
  double kraft = 0;
  for (int i = 0; i < 256; ++i) {
    const int sym = s[i], l = len[sym];
    if (l < 1 || l > kLmax) die("code length %d", l);
    if (i) code = (code + 1) << (l - len[s[i - 1]]);
    C.code[sym] = code, C.len[sym] = l;
    kraft += std::ldexp(1.0, -l);
    for (uint32_t e = code << (kLmax - l); e < (code + 1) << (kLmax - l); ++e) C.table[e] = (uint16_t)(l << 8 | sym);
  }
  if (kraft > 1.0 + 1e-12 || code + 1 != (1u << len[s[255]])) die("not a complete prefix code (Kraft %.9f)", kraft);
  return C;
}

// ------------------------------------------------------------------ encode / decode (host)
// W: a launch's segments, tiled ([rows / 8][K / 64][8][64]) and concatenated in row order
struct Enc {
  int ctas = 0;
  std::vector<uint32_t> words;
  std::vector<uint32_t> off;  // [ctas][wk]: each run's first word (a multiple of 4)
  uint64_t flushed = 0;       // bits in the words past the codes
};

struct Bits {  // one lane's codes, MSB first
  std::vector<uint32_t> v;
  uint64_t n = 0;
  void put(uint32_t code, int len) {
    const uint64_t pos = n;
    n += len;
    if (v.size() < (n + 31) / 32 + 1) v.resize((n + 31) / 32 + 1, 0);
    const int bo = (int)(pos % 32);
    const uint64_t x = (uint64_t)code << (64 - len - bo);
    v[pos / 32] |= (uint32_t)(x >> 32);
    if (bo + len > 32) v[pos / 32 + 1] |= (uint32_t)x;
  }
  uint32_t get32(uint64_t pos) const {
    const size_t wi = pos / 32;
    const int bo = (int)(pos % 32);
    const uint32_t hi = wi < v.size() ? v[wi] : 0, lo = wi + 1 < v.size() ? v[wi + 1] : 0;
    return bo ? (hi << bo) | (lo >> (32 - bo)) : hi;
  }
};

struct Run {
  int t0, t1, p0, p1;
};
Run run_of(int b, int w, int ntile, int P, int ctas, int wk) {
  return {(int)((long long)ntile * b / ctas), (int)((long long)ntile * (b + 1) / ctas), P * w / wk, P * (w + 1) / wk};
}

void encode(const uint8_t* W, int ntile, int K, int ctas, int wk, const Code& C, Enc& e) {
  const int P = K / 64;
  e.ctas = ctas;
  e.off.assign((size_t)ctas * wk, 0);
  for (int b = 0; b < ctas; ++b)
    for (int w = 0; w < wk; ++w) {
      const Run r = run_of(b, w, ntile, P, ctas, wk);
      while (e.words.size() % 4) e.words.push_back(0);
      e.off[(size_t)b * wk + w] = (uint32_t)e.words.size();
      Bits lane[32];
      std::vector<uint8_t> lens[32];
      for (int tl = r.t0; tl < r.t1; ++tl)
        for (int p = r.p0; p < r.p1; ++p) {
          const uint8_t* s = W + ((size_t)tl * P + p) * 512;
          for (int l = 0; l < 32; ++l)
            for (int j = 0; j < 16; ++j) {
              const uint8_t sym = s[16 * l + j];
              lane[l].put(C.code[sym], C.len[sym]);
              lens[l].push_back((uint8_t)C.len[sym]);
            }
        }
      // lockstep: before every pair of bytes, each lane with <= 32 bits takes the next word, in lane order
      int nb[32] = {};
      uint64_t fed[32] = {};
      for (size_t i = 0; i < lens[0].size(); i += 2) {
        for (int l = 0; l < 32; ++l)
          if (nb[l] <= 32) e.words.push_back(lane[l].get32(fed[l])), fed[l] += 32, nb[l] += 32;
        for (int l = 0; l < 32; ++l) nb[l] -= lens[l][i] + lens[l][i + 1];
      }
      for (int l = 0; l < 32; ++l) e.flushed += fed[l] - lane[l].n;
    }
  while (e.words.size() % 4) e.words.push_back(0);
}

// the GPU decoder's steps on the CPU: every byte must come back
bool check_decode(const uint8_t* W, int ntile, int K, int wk, const Code& C, const Enc& e) {
  const int P = K / 64;
  for (int b = 0; b < e.ctas; ++b)
    for (int w = 0; w < wk; ++w) {
      const Run r = run_of(b, w, ntile, P, e.ctas, wk);
      const uint32_t* src = e.words.data() + e.off[(size_t)b * wk + w];
      uint64_t buf[32] = {};
      int nb[32] = {};
      size_t wpos = 0;
      for (int tl = r.t0; tl < r.t1; ++tl)
        for (int p = r.p0; p < r.p1; ++p) {
          const uint8_t* s = W + ((size_t)tl * P + p) * 512;
          for (int j = 0; j < 16; j += 2) {
            for (int l = 0; l < 32; ++l)
              if (nb[l] <= 32) buf[l] |= (uint64_t)src[wpos++] << (32 - nb[l]), nb[l] += 32;
            for (int l = 0; l < 32; ++l)
              for (int jj = j; jj < j + 2; ++jj) {
                const uint16_t t = C.table[buf[l] >> (64 - kLmax)];
                const int len = t >> 8;
                buf[l] <<= len, nb[l] -= len;
                if ((t & 0xFF) != s[16 * l + jj]) return false;
              }
          }
        }
    }
  return true;
}

// ------------------------------------------------------------------ the blob (prep -> run)
struct Layer {
  std::vector<uint8_t> W;               // the segments' tiled weights, concatenated
  std::vector<std::vector<uint8_t>> S;  // each segment's block scales
  std::vector<Enc> z;                   // one per grid
};
struct Blob {
  int K = 0, wk = 0;
  std::vector<int> N, layers, ctas;  // ctas: the grids the runs are cut for
  Code C;
  std::vector<Layer> L;
};

template <class T>
void put(FILE* f, const T* p, size_t n) {
  if (fwrite(p, sizeof(T), n, f) != n) die("write");
}
template <class T>
void get(FILE* f, T* p, size_t n) {
  if (fread(p, sizeof(T), n, f) != n) die("read");
}
template <class T>
void put_vec(FILE* f, const std::vector<T>& v) {
  const uint64_t n = v.size();
  put(f, &n, 1);
  put(f, v.data(), n);
}
template <class T>
void get_vec(FILE* f, std::vector<T>& v) {
  uint64_t n;
  get(f, &n, 1);
  v.resize(n);
  get(f, v.data(), n);
}

void save(const std::string& path, const Blob& b) {
  FILE* f = fopen(path.c_str(), "wb");
  if (!f) die("open %s", path.c_str());
  const int hdr[5] = {b.K, b.wk, (int)b.ctas.size(), (int)b.N.size(), (int)b.layers.size()};
  put(f, "WCOMPZ2", 8);
  put(f, hdr, 5);
  put(f, b.ctas.data(), b.ctas.size());
  put(f, b.N.data(), b.N.size());
  put(f, b.layers.data(), b.layers.size());
  put(f, &b.C, 1);
  for (const Layer& l : b.L) {
    put_vec(f, l.W);
    for (const auto& s : l.S) put_vec(f, s);
    for (const Enc& z : l.z) {
      put_vec(f, z.words);
      put_vec(f, z.off);
      put(f, &z.flushed, 1);
    }
  }
  fclose(f);
}

Blob load(const std::string& path) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) die("open %s", path.c_str());
  char magic[8];
  int hdr[5];
  get(f, magic, 8);
  if (std::memcmp(magic, "WCOMPZ2", 8)) die("%s: not a blob", path.c_str());
  get(f, hdr, 5);
  Blob b;
  b.K = hdr[0], b.wk = hdr[1];
  b.ctas.resize(hdr[2]);
  b.N.resize(hdr[3]);
  b.layers.resize(hdr[4]);
  get(f, b.ctas.data(), b.ctas.size());
  get(f, b.N.data(), b.N.size());
  get(f, b.layers.data(), b.layers.size());
  get(f, &b.C, 1);
  b.L.resize(b.layers.size());
  for (Layer& l : b.L) {
    get_vec(f, l.W);
    l.S.resize(b.N.size());
    for (auto& s : l.S) get_vec(f, s);
    l.z.resize(b.ctas.size());
    for (size_t v = 0; v < b.ctas.size(); ++v) {
      Enc& z = l.z[v];
      z.ctas = b.ctas[v];
      get_vec(f, z.words);
      get_vec(f, z.off);
      get(f, &z.flushed, 1);
    }
  }
  fclose(f);
  return b;
}

int prep(const char* file, const std::string& dir, const std::vector<int>& layers) {
  RankFile rf(file);
  for (const Site& site : kSites) {
    const auto t0 = std::chrono::steady_clock::now();
    Blob b;
    b.wk = site.wk;
    b.layers = layers;
    std::vector<uint64_t> hist(256, 0);
    for (int L : layers) {
      Layer l;
      int K = 0;
      for (const char* sg : site.segs) {
        const std::string base = "layers." + std::to_string(L) + "." + sg;
        const Ent& e = rf.at(base + ".weight");
        if (e.dtype != 3 || e.layout != 1) die("%s: not tiled FP8", base.c_str());
        const int N = (int)e.shape[0];
        K = (int)e.shape[1];
        if (b.N.size() < site.segs.size()) b.N.push_back(N);
        std::vector<uint8_t> w = rf.read(base + ".weight");
        l.W.insert(l.W.end(), w.begin(), w.end());
        l.S.push_back(rf.read(base + ".scale"));
      }
      b.K = K;
      for (uint8_t v : l.W) ++hist[v];
      b.L.push_back(std::move(l));
    }
    // one code for the site: every layer's bytes
    std::vector<int> len = code_lengths(hist, kLmax);
    b.C = canonical(len);
    double H = 0, avg = 0, tot = 0;
    for (int v = 0; v < 256; ++v) tot += (double)hist[v];
    for (int v = 0; v < 256; ++v)
      if (hist[v]) H -= hist[v] / tot * std::log2(hist[v] / tot), avg += hist[v] / tot * len[v];
    const int rows = std::accumulate(b.N.begin(), b.N.end(), 0), ntile = rows / 8;
    for (int cpm : kCpm) {
      const int c = std::min(ntile, cpm * kSMs);
      if (std::find(b.ctas.begin(), b.ctas.end(), c) == b.ctas.end()) b.ctas.push_back(c);
    }
    std::printf("%-8s K %d rows %d wk %d layers %zu: entropy %.4f, code %.4f bits/byte (max len %d)\n", site.name, b.K,
                rows, b.wk, layers.size(), H, avg, *std::max_element(len.begin(), len.end()));
    for (size_t v = 0; v < b.ctas.size(); ++v) {
      uint64_t raw = 0, comp = 0, flushed = 0;
      for (Layer& l : b.L) {
        l.z.emplace_back();
        encode(l.W.data(), ntile, b.K, b.ctas[v], b.wk, b.C, l.z.back());
        if (!check_decode(l.W.data(), ntile, b.K, b.wk, b.C, l.z.back()))
          die("%s ctas %d: the CPU decode differs", site.name, b.ctas[v]);
        raw += l.W.size();
        comp += l.z.back().words.size() * 4 + l.z.back().off.size() * 4;
        flushed += l.z.back().flushed;
      }
      std::printf("  ctas %3d: stream %.4f bits/byte (flush %.3f %%, offsets %.3f %%), %.2f %% smaller with the table\n",
                  b.ctas[v], 8.0 * comp / raw, 100.0 * flushed / 8 / raw,
                  100.0 * b.ctas[v] * b.wk * 4 * layers.size() / raw, 100.0 * (1 - (double)(comp + 2 * kTab) / raw));
    }
    save(dir + "/" + site.name + ".zb", b);
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("  %.1f s\n", s);
    std::fflush(stdout);
  }
  return 0;
}

// ------------------------------------------------------------------ the kernel
struct ZSegs {  // gemv_fp8.cu's Segs
  dsk::Fp8Seg s[dsk::kMaxSegs];
  int row0[dsk::kMaxSegs + 1];
  int n;
};
struct ZW {
  const uint32_t* words;
  const uint32_t* off;    // [gridDim.x][wk]
  const uint16_t* table;  // [kTab]
};

size_t z_smem(int S, int wk, int M, int K, bool norm) {
  return (size_t)2 * kTab + (size_t)wk * S * 512 + (size_t)2 * wk * 64 * sizeof(float) +
         (norm ? (size_t)M * (K + 32) + (size_t)M * (K / 32) : 0);
}

// k_gemv_fp8p with the ring carrying the run's code words instead of raw bytes, and 16 bytes decoded per
// lane and step. A lane keeps up to 64 bits in two registers, MSB first from hi; a refill takes no branch
// (a ballot, a shared load and two funnel shifts). The ring is managed once per step: a step reads at most
// 8 words a lane (the bits it decodes, at most 16 x 12, plus up to 62 left over), 256 a warp, so chunks
// head .. head + 2 must have landed. DUMP: also write the decoded bytes where the file has them (dump: the
// segments' tiled weights, concatenated).
template <int S, bool NORM, bool DUMP>
__global__ void __launch_bounds__(512, 2) k_gemv_fp8z(const uint8_t* __restrict__ xq, const uint8_t* __restrict__ xs,
                                                       int M, ZSegs segs, int K, int ntile, dsk::NormIn nin, ZW z,
                                                       uint8_t* __restrict__ dump) {
  using namespace dsk;
  static_assert(S == 4 || S == 8 || S == 16, "a power of two, at least 4");
  // table [kTab] u16 | ring [wk][S][512] | red [2][wk][8][8] | NORM: q [M][K + 32], s [M][K/32]
  extern __shared__ __align__(16) uint8_t sm[];
  const int wk = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int P = K / 64, KB = K / 32, ldq = NORM ? K + 32 : K;
  const uint16_t* const tab = reinterpret_cast<const uint16_t*>(sm);
  uint32_t* const ring = reinterpret_cast<uint32_t*>(sm + 2 * kTab + (size_t)warp * S * 512);
  float* const red = reinterpret_cast<float*>(sm + 2 * kTab + (size_t)wk * S * 512);
  uint8_t* const sq = reinterpret_cast<uint8_t*>(red + 2 * wk * 64);
  uint8_t* const ss = sq + (size_t)M * ldq;
  const int p0 = (P * warp) / wk, p1 = (P * (warp + 1)) / wk;
  const int t0 = (int)((long long)ntile * blockIdx.x / gridDim.x);
  const int t1 = (int)((long long)ntile * (blockIdx.x + 1) / gridDim.x);

  // the run's words: chunk c (128 words, 4 a lane) goes to slot c % S; chunks [head, head + S) are in flight
  const uint32_t* const src = z.words + z.off[blockIdx.x * wk + warp] + 4 * lane;
#pragma unroll
  for (int i = 0; i < S; ++i) {
    cp_async16(ring + i * 128 + 4 * lane, src + i * 128);
    cp_async_commit();
  }
  for (int i = threadIdx.x; i < kTab / 8; i += blockDim.x)
    reinterpret_cast<uint4*>(sm)[i] = __ldg(reinterpret_cast<const uint4*>(z.table) + i);
  pdl_launch_dependents();
  pdl_wait();
  if constexpr (NORM) {
    const bool first = blockIdx.x == 0;
    rmsnorm_rows(nin.x, K, nin.w, M, K, nin.eps, red, first ? nin.y : nullptr, K, sq, ldq, ss, first ? nin.q : nullptr,
                 first ? nin.qs : nullptr);
  }
  __syncthreads();  // the table, and the normed rows
  const bool live = g < M, odd = lane & 1, lo_lane = t < 2;
  const int off = 16 * (t & 1) + 8 * (t >> 1);
  const uint8_t* arow = (NORM ? sq : xq) + (size_t)(live ? g : 0) * ldq + off;
  const uint8_t* srowa = (NORM ? ss : xs) + (size_t)(live ? g : 0) * KB;

  const uint32_t below = (1u << lane) - 1u;
  uint32_t hi = 0, lo = 0;  // the next bits, MSB first; bits past nb are zero
  int nb = 0;
  int wpos = 0, head = 0;  // the same in every lane
  auto refill = [&]() {
    const bool need = nb <= 32;
    const uint32_t mask = __ballot_sync(0xffffffffu, need);
    const uint32_t w = ring[(wpos + __popc(mask & below)) & (S * 128 - 1)];
    if (need) {
      hi |= __funnelshift_rc(w, 0u, (uint32_t)nb);  // w >> nb
      lo = __funnelshift_rc(0u, w, (uint32_t)nb);   // w << (32 - nb); lo held no bits
      nb += 32;
    }
    wpos += __popc(mask);
  };
  auto dec = [&]() -> uint32_t {
    const uint32_t e = tab[hi >> (32 - kLmax)];
    const uint32_t len = e >> 8;
    hi = __funnelshift_l(lo, hi, len);
    lo <<= len;
    nb -= (int)len;
    return e & 0xFFu;
  };

  int cs = 0;
  while (cs + 1 < segs.n && 8 * t0 >= segs.row0[cs + 1]) ++cs;
#pragma unroll 1
  for (int tl = t0; tl < t1; ++tl) {
    while (cs + 1 < segs.n && 8 * tl >= segs.row0[cs + 1]) ++cs;
    const int n = 8 * tl - segs.row0[cs];
    const uint8_t* const scl = segs.s[cs].S + (size_t)(n / 32) * KB;
    float c[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 1
    for (int p = p0; p < p1; ++p) {
      const uint32_t sw = __ldg(reinterpret_cast<const uint16_t*>(scl + 2 * p));  // blocks 2p, 2p + 1
      while ((wpos >> 7) > head) {  // chunk head is read: its slot takes chunk head + S
        __syncwarp();
        cp_async16(ring + (head & (S - 1)) * 128 + 4 * lane, src + (size_t)(head + S) * 128);
        cp_async_commit();
        ++head;
      }
      cp_async_wait<S - 3>();  // chunks head .. head + 2
      __syncwarp();
      uint32_t v[4];
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        refill();
        const uint32_t b0 = dec();
        const uint32_t b1 = dec();
        refill();
        const uint32_t b2 = dec();
        const uint32_t b3 = dec();
        v[q] = b0 | b1 << 8 | b2 << 16 | b3 << 24;
      }
      const uint4 w = make_uint4(v[0], v[1], v[2], v[3]);
      if constexpr (DUMP)
        *reinterpret_cast<uint4*>(dump + (size_t)(segs.row0[cs] + n) * K + (size_t)p * 512 + 16 * lane) = w;
      uint2 a[2];
      uint32_t sfa[2];
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int kb = 2 * p + h;
        if constexpr (NORM) {
          a[h] = live ? *reinterpret_cast<const uint2*>(arow + kb * 32) : make_uint2(0, 0);
          sfa[h] = (odd || !live) ? 127u : srowa[kb];
        } else {
          a[h] = live ? __ldg(reinterpret_cast<const uint2*>(arow + kb * 32)) : make_uint2(0, 0);
          sfa[h] = (odd || !live) ? 127u : __ldg(srowa + kb);
        }
      }
      const uint32_t s0 = lo_lane ? w.z : w.x, s1 = lo_lane ? w.w : w.y;
      const uint32_t r0 = __shfl_xor_sync(0xffffffffu, s0, 2), r1 = __shfl_xor_sync(0xffffffffu, s1, 2);
      mma_e4m3_e4m3(c, a[0].x, 0u, a[0].y, 0u, lo_lane ? w.x : r0, lo_lane ? w.y : r1, sfa[0], sw & 0xFFu);
      mma_e4m3_e4m3(c, a[1].x, 0u, a[1].y, 0u, lo_lane ? r0 : w.z, lo_lane ? r1 : w.w, sfa[1], (sw >> 8) & 0xFFu);
    }
    // the slices' partial sums, then k_gemv_fp8's fixed-order sum over them (k_gemv_fp8p's code)
    float* rd = red + (tl & 1) * wk * 64;
    rd[(warp * 8 + g) * 8 + 2 * t] = c[0];
    rd[(warp * 8 + g) * 8 + 2 * t + 1] = c[1];
    __syncthreads();
    if (threadIdx.x < M * 8) {
      const int m = threadIdx.x >> 3, r = threadIdx.x & 7;
      float v = 0.f;
      for (int j = 0; j < wk; ++j) v += rd[(j * 8 + m) * 8 + r];
      segs.s[cs].y[(size_t)m * segs.s[cs].N + n + r] = __float2bfloat16_rn(v);
    }
  }
  cp_async_wait<0>();  // the ring reads past the run's end: nothing in flight at exit
}

template <int S, bool NORM, bool DUMP>
auto z_kernel() {
  return k_gemv_fp8z<S, NORM, DUMP>;
}

template <int S, bool NORM, bool DUMP>
void z_attr() {
  static bool done = false;
  if (done) return;
  CUDA_CHECK(cudaFuncSetAttribute(z_kernel<S, NORM, DUMP>(), cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024));
  CUDA_CHECK(cudaFuncSetAttribute(z_kernel<S, NORM, DUMP>(), cudaFuncAttributePreferredSharedMemoryCarveout,
                                  cudaSharedmemCarveoutMaxShared));
  done = true;
}

// CTAs of this kernel that fit on an SM at once
template <bool NORM>
int z_fit(int S, int wk, int M, int K) {
  int n = 0;
  const size_t smem = z_smem(S, wk, M, K, NORM);
  switch (S) {
    case 4: z_attr<4, NORM, false>(); CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, z_kernel<4, NORM, false>(), 32 * wk, smem)); break;
    case 8: z_attr<8, NORM, false>(); CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, z_kernel<8, NORM, false>(), 32 * wk, smem)); break;
    case 16: z_attr<16, NORM, false>(); CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, z_kernel<16, NORM, false>(), 32 * wk, smem)); break;
    default: die("ring %d", S);
  }
  return n;
}

template <int S, bool NORM, bool DUMP>
void launch_z1(const uint8_t* xq, const uint8_t* xs, int M, const ZSegs& segs, int K, int wk, int ctas,
               const dsk::NormIn& nin, const ZW& z, uint8_t* dump, cudaStream_t st) {
  z_attr<S, NORM, DUMP>();
  dsk::launch_pdl(z_kernel<S, NORM, DUMP>(), dim3(ctas), dim3(32 * wk), z_smem(S, wk, M, K, NORM), st, true, xq, xs, M,
                  segs, K, segs.row0[segs.n] / 8, nin, z, dump);
}

template <bool NORM, bool DUMP>
void launch_z(int S, const uint8_t* xq, const uint8_t* xs, int M, const ZSegs& segs, int K, int wk, int ctas,
              const dsk::NormIn& nin, const ZW& z, uint8_t* dump, cudaStream_t st) {
  switch (S) {
    case 4: launch_z1<4, NORM, DUMP>(xq, xs, M, segs, K, wk, ctas, nin, z, dump, st); break;
    case 8: launch_z1<8, NORM, DUMP>(xq, xs, M, segs, K, wk, ctas, nin, z, dump, st); break;
    case 16: launch_z1<16, NORM, DUMP>(xq, xs, M, segs, K, wk, ctas, nin, z, dump, st); break;
    default: die("ring %d", S);
  }
}


// ------------------------------------------------------------------ the fixed-length code (stack5, 2026-11-11)
// What the study costed from SASS only ("FP8: a 3-bit exponent code, with a lane fallback"), built and run.
// An e4m3 byte is sign | 4-bit exponent | 3-bit mantissa. Per 64-byte step of 8 rows (512 bytes, a warp-step)
// the code keeps 448 bytes: lane l's 16 bytes become a u64 of (sign << 3 | mantissa) nibbles at 8l, and 48 bits
// of 3-bit exponent codes (a u32 at 256 + 4l, a u16 at 384 + 2l). The code of a byte is (exponent - B) mod 16
// for the layer's window base B; its low 3 bits are the fixed part. A lane chunk with a code >= 8 is flagged
// in the step's u32 (one bit a lane), and its 16 fourth bits are a u16 in a side list, found by a running
// popcount. Every byte value has a code, so the decode is exact for any tensor. The steps stay in the file's
// order, so one coding serves any grid; the side list is ordered slice by slice (the warps' k ranges), so a
// (CTA, warp) run reads it front to back from one offset.
struct XEnc {
  std::vector<uint8_t> fix;    // [ntile * P][448]
  std::vector<uint32_t> flg;   // [ntile * P]
  std::vector<uint16_t> side;  // for slice w, for tile, for p in the slice: the flagged lanes' fourth bits
  std::vector<uint32_t> soff;  // [wk][ntile]: where (slice, tile) starts in side
  int B = 0;
  uint64_t flagged = 0, chunks = 0;
  size_t bytes() const { return fix.size() + flg.size() * 4 + side.size() * 2 + soff.size() * 4; }
};

// One word (4 bytes) of a lane: n16 its four nibbles, t its four 3-bit codes, p3n their fourth bits, bm the
// base in every byte. The same arithmetic on the host (the encoder's check) and in the kernel.
__host__ __device__ inline uint32_t x_word(uint32_t n16, uint32_t t, uint32_t p3n, uint32_t bm) {
  const uint32_t x1 = (n16 | n16 << 8) & 0x00FF00FFu;
  const uint32_t zz = (x1 | x1 << 4) & 0x0F0F0F0Fu;
  const uint32_t sgm = (zz | zz << 4) & 0x87878787u;
  const uint32_t y = (t | t << 10) & 0x003F003Fu;
  const uint32_t cz = (y | y << 5) & 0x07070707u;
  const uint32_t c4 = cz | ((p3n * 0x01020408u) & 0x08080808u);
  return sgm | (((c4 + bm) & 0x0F0F0F0Fu) << 3);
}

XEnc x_encode(const uint8_t* W, int ntile, int K, int wk) {
  const int P = K / 64;
  const size_t nstep = (size_t)ntile * P;
  XEnc e;
  uint64_t h[16] = {};
  for (size_t i = 0; i < nstep * 512; ++i) ++h[(W[i] >> 3) & 15];
  uint64_t best = 0;
  for (int b = 0; b < 16; ++b) {
    uint64_t sum = 0;
    for (int j = 0; j < 8; ++j) sum += h[(b + j) & 15];
    if (sum > best) best = sum, e.B = b;
  }
  e.fix.assign(nstep * 448, 0);
  e.flg.assign(nstep, 0);
  e.soff.assign((size_t)wk * ntile, 0);
  std::vector<uint16_t> p3all(nstep * 32);
  for (size_t st = 0; st < nstep; ++st)
    for (int l = 0; l < 32; ++l) {
      const uint8_t* src = W + st * 512 + 16 * l;
      uint64_t nib = 0, c48 = 0;
      uint32_t p3 = 0;
      for (int j = 0; j < 16; ++j) {
        const uint32_t b = src[j], c4 = (((b >> 3) & 15) - e.B) & 15;
        nib |= (uint64_t)(((b >> 7) << 3) | (b & 7)) << (4 * j);
        c48 |= (uint64_t)(c4 & 7) << (3 * j);
        p3 |= (c4 >> 3) << j;
      }
      const uint32_t clo = (uint32_t)c48;
      const uint16_t chi = (uint16_t)(c48 >> 32);
      std::memcpy(&e.fix[st * 448 + 8 * l], &nib, 8);
      std::memcpy(&e.fix[st * 448 + 256 + 4 * l], &clo, 4);
      std::memcpy(&e.fix[st * 448 + 384 + 2 * l], &chi, 2);
      if (p3) e.flg[st] |= 1u << l, ++e.flagged;
      p3all[st * 32 + l] = (uint16_t)p3;
      ++e.chunks;
    }
  for (int w = 0; w < wk; ++w) {
    const int p0 = P * w / wk, p1 = P * (w + 1) / wk;
    for (int tl = 0; tl < ntile; ++tl) {
      e.soff[(size_t)w * ntile + tl] = (uint32_t)e.side.size();
      for (int p = p0; p < p1; ++p) {
        const size_t st = (size_t)tl * P + p;
        for (int l = 0; l < 32; ++l)
          if (e.flg[st] >> l & 1) e.side.push_back(p3all[st * 32 + l]);
      }
    }
  }
  e.side.resize(e.side.size() + 64, 0);  // a predicated load never reads past the list; padding all the same
  return e;
}

// the kernel's steps on the CPU: every byte must come back
bool x_check(const uint8_t* W, int ntile, int K, int wk, const XEnc& e) {
  const int P = K / 64;
  const uint32_t bm = (uint32_t)e.B * 0x01010101u;
  for (int w = 0; w < wk; ++w) {
    const int p0 = P * w / wk, p1 = P * (w + 1) / wk;
    size_t sidx = e.soff[(size_t)w * ntile];
    for (int tl = 0; tl < ntile; ++tl) {
      if (sidx != e.soff[(size_t)w * ntile + tl]) return false;
      for (int p = p0; p < p1; ++p) {
        const size_t st = (size_t)tl * P + p;
        const uint32_t f = e.flg[st];
        for (int l = 0; l < 32; ++l) {
          uint64_t nib;
          uint32_t clo;
          uint16_t chi;
          std::memcpy(&nib, &e.fix[st * 448 + 8 * l], 8);
          std::memcpy(&clo, &e.fix[st * 448 + 256 + 4 * l], 4);
          std::memcpy(&chi, &e.fix[st * 448 + 384 + 2 * l], 2);
          const uint32_t p3 = (f >> l & 1) ? e.side[sidx + __builtin_popcount(f & ((1u << l) - 1u))] : 0u;
          const uint64_t c48 = clo | (uint64_t)chi << 32;
          for (int q = 0; q < 4; ++q) {
            const uint32_t v = x_word((uint32_t)(nib >> (16 * q)) & 0xFFFFu, (uint32_t)(c48 >> (12 * q)) & 0xFFFu,
                                      (p3 >> (4 * q)) & 0xFu, bm);
            if (std::memcmp(&v, W + st * 512 + 16 * l + 4 * q, 4)) return false;
          }
        }
        sidx += __builtin_popcount(f);
      }
    }
  }
  return true;
}

struct XW {
  const uint8_t* fix;
  const uint32_t* flg;
  const uint16_t* side;
  const uint32_t* soff;
  uint32_t bm;
};

__device__ __forceinline__ uint32_t ldg_stream16(const void* p) {
  uint32_t r;
  asm volatile("ld.global.nc.L1::no_allocate.u16 %0, [%1];" : "=r"(r) : "l"(p));
  return r;
}
__device__ __forceinline__ uint32_t ldg_stream16_if(uint32_t c, const void* p) {
  uint32_t r;
  asm volatile("{\n .reg .pred q;\n setp.ne.b32 q, %1, 0;\n mov.u32 %0, 0;\n @q ld.global.nc.L1::no_allocate.u16 %0, [%2];\n}"
               : "=r"(r)
               : "r"(c), "l"(p));
  return r;
}

constexpr int kXF = 128;  // a (CTA, warp) run's flags, read into shared memory at the start: at most this many steps
size_t x_smem(int wk, int M, int K, bool norm) {
  return (size_t)wk * kXF * 4 + (size_t)2 * wk * 64 * sizeof(float) + (norm ? (size_t)M * (K + 32) + (size_t)M * (K / 32) : 0);
}

// k_gemv_fp8p's decomposition and arithmetic (CTA b: tiles [t0, t1); warp w: slice w of each; per tile the
// slices' partial sums added in slice order), the weights read straight from global memory in two register
// buffers of U steps used in turn, as k_gemv_fp8 does, and decoded in registers. The side list's addresses
// depend on the flags, so a warp reads its whole run's flags into shared memory first (v1 requested them one
// batch ahead, a dependent load chain: 1.2 % faster to 7.6 % slower than today's kernels).
template <bool NORM, bool DUMP, int CPM, int U>
__global__ void __launch_bounds__(512, CPM) k_gemv_fp8x(const uint8_t* __restrict__ xq, const uint8_t* __restrict__ xs,
                                                         int M, ZSegs segs, int K, int ntile, dsk::NormIn nin, XW z,
                                                         uint8_t* __restrict__ dump) {
  using namespace dsk;
  extern __shared__ __align__(16) uint8_t sm[];  // flags [wk][kXF] | red [2][wk][8][8] | NORM: q [M][K + 32], s [M][K/32]
  constexpr int kXU = U;
  const int wk = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int P = K / 64, KB = K / 32, ldq = NORM ? K + 32 : K;
  uint32_t* const sf = reinterpret_cast<uint32_t*>(sm) + warp * kXF;
  float* const red = reinterpret_cast<float*>(sm + (size_t)wk * kXF * 4);
  uint8_t* const sq = reinterpret_cast<uint8_t*>(red + 2 * wk * 64);
  uint8_t* const ss = sq + (size_t)M * ldq;
  const int p0 = (P * warp) / wk, p1 = (P * (warp + 1)) / wk;
  const int t0 = (int)((long long)ntile * blockIdx.x / gridDim.x);
  const int t1 = (int)((long long)ntile * (blockIdx.x + 1) / gridDim.x);
  const uint32_t below = (1u << lane) - 1u, bit = 1u << lane;
  uint32_t sidx = __ldg(z.soff + (size_t)warp * ntile + t0);
  int ltl = t0, lp = p0, li = 0;
  {
    const int PW = p1 - p0, nrun = (t1 - t0) * PW;
    for (int i = lane; i < nrun; i += 32) sf[i] = __ldg(z.flg + (size_t)(t0 + i / PW) * P + p0 + i % PW);
    __syncwarp();
  }
  struct XB {
    uint2 nib[kXU];
    uint32_t clo[kXU], chi[kXU], p3[kXU];
  };
  XB ba, bb;
  auto load_w = [&](XB& b) {
#pragma unroll
    for (int u = 0; u < kXU; ++u)
      if (ltl < t1) {
        const uint8_t* s = z.fix + ((size_t)ltl * P + lp) * 448;
        const uint32_t f = sf[li++];
        b.nib[u] = ldg_stream64(s + 8 * lane);
        b.clo[u] = ldg_stream32(s + 256 + 4 * lane);
        b.chi[u] = ldg_stream16(s + 384 + 2 * lane);
        b.p3[u] = ldg_stream16_if(f & bit, z.side + sidx + __popc(f & below));
        sidx += __popc(f);
        if (++lp == p1) lp = p0, ++ltl;
      }
  };
  load_w(ba);
  pdl_launch_dependents();
  pdl_wait();
  if constexpr (NORM) {
    const bool first = blockIdx.x == 0;
    rmsnorm_rows(nin.x, K, nin.w, M, K, nin.eps, red, first ? nin.y : nullptr, K, sq, ldq, ss, first ? nin.q : nullptr,
                 first ? nin.qs : nullptr);
  }
  __syncthreads();
  const bool live = g < M, odd = lane & 1, lo_lane = t < 2;
  const int off = 16 * (t & 1) + 8 * (t >> 1);
  const uint8_t* arow = (NORM ? sq : xq) + (size_t)(live ? g : 0) * ldq + off;
  const uint8_t* srowa = (NORM ? ss : xs) + (size_t)(live ? g : 0) * KB;
  int ctl = t0, cp = p0, cs = 0;
  while (cs + 1 < segs.n && 8 * t0 >= segs.row0[cs + 1]) ++cs;
  float c[4] = {0.f, 0.f, 0.f, 0.f};
  auto comp = [&](const XB& b) {
#pragma unroll
    for (int u = 0; u < kXU; ++u) {
      if (ctl < t1) {
        const int n = 8 * ctl - segs.row0[cs];
        const uint8_t* const scl = segs.s[cs].S + (size_t)(n / 32) * KB;
        const uint32_t sw = __ldg(reinterpret_cast<const uint16_t*>(scl + 2 * cp));
        const uint64_t c48 = (uint64_t)b.clo[u] | (uint64_t)b.chi[u] << 32;
        uint32_t v[4];
#pragma unroll
        for (int q = 0; q < 4; ++q)
          v[q] = x_word(((q < 2 ? b.nib[u].x : b.nib[u].y) >> (16 * (q & 1))) & 0xFFFFu,
                        (uint32_t)(c48 >> (12 * q)) & 0xFFFu, (b.p3[u] >> (4 * q)) & 0xFu, z.bm);
        const uint4 w = make_uint4(v[0], v[1], v[2], v[3]);
        if constexpr (DUMP)
          *reinterpret_cast<uint4*>(dump + (size_t)(segs.row0[cs] + n) * K + (size_t)cp * 512 + 16 * lane) = w;
        uint2 a[2];
        uint32_t sfa[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int kb = 2 * cp + h;
          if constexpr (NORM) {
            a[h] = live ? *reinterpret_cast<const uint2*>(arow + kb * 32) : make_uint2(0, 0);
            sfa[h] = (odd || !live) ? 127u : srowa[kb];
          } else {
            a[h] = live ? __ldg(reinterpret_cast<const uint2*>(arow + kb * 32)) : make_uint2(0, 0);
            sfa[h] = (odd || !live) ? 127u : __ldg(srowa + kb);
          }
        }
        const uint32_t s0 = lo_lane ? w.z : w.x, s1 = lo_lane ? w.w : w.y;
        const uint32_t r0 = __shfl_xor_sync(0xffffffffu, s0, 2), r1 = __shfl_xor_sync(0xffffffffu, s1, 2);
        mma_e4m3_e4m3(c, a[0].x, 0u, a[0].y, 0u, lo_lane ? w.x : r0, lo_lane ? w.y : r1, sfa[0], sw & 0xFFu);
        mma_e4m3_e4m3(c, a[1].x, 0u, a[1].y, 0u, lo_lane ? r0 : w.z, lo_lane ? r1 : w.w, sfa[1], (sw >> 8) & 0xFFu);
        if (++cp == p1) {  // the tile is done: its slices' partial sums, in slice order (k_gemv_fp8p's code)
          float* rd = red + (ctl & 1) * wk * 64;
          rd[(warp * 8 + g) * 8 + 2 * t] = c[0];
          rd[(warp * 8 + g) * 8 + 2 * t + 1] = c[1];
          __syncthreads();
          if (threadIdx.x < M * 8) {
            const int m = threadIdx.x >> 3, r = threadIdx.x & 7;
            float vv = 0.f;
            for (int j = 0; j < wk; ++j) vv += rd[(j * 8 + m) * 8 + r];
            segs.s[cs].y[(size_t)m * segs.s[cs].N + n + r] = __float2bfloat16_rn(vv);
          }
          c[0] = c[1] = c[2] = c[3] = 0.f;
          cp = p0;
          ++ctl;
          while (cs + 1 < segs.n && 8 * ctl >= segs.row0[cs + 1]) ++cs;
        }
      }
    }
  };
#pragma unroll 1
  while (ctl < t1) {
    load_w(bb);
    comp(ba);
    if (ctl >= t1) break;
    load_w(ba);
    comp(bb);
  }
}

template <bool NORM, bool DUMP, int CPM, int U>
void launch_x1(const uint8_t* xq, const uint8_t* xs, int M, const ZSegs& segs, int K, int wk, int ctas,
               const dsk::NormIn& nin, const XW& z, uint8_t* dump, cudaStream_t st) {
  static bool done = false;
  if (!done) {
    CUDA_CHECK(cudaFuncSetAttribute(k_gemv_fp8x<NORM, DUMP, CPM, U>, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024));
    done = true;
  }
  dsk::launch_pdl(k_gemv_fp8x<NORM, DUMP, CPM, U>, dim3(ctas), dim3(32 * wk), x_smem(wk, M, K, NORM), st, true, xq, xs, M,
                  segs, K, segs.row0[segs.n] / 8, nin, z, dump);
}
// cfg = 10 * (CTAs per SM) + (steps a register buffer)
template <bool NORM>
void launch_x(int cfg, bool dmp, const uint8_t* xq, const uint8_t* xs, int M, const ZSegs& segs, int K, int wk, int ctas,
              const dsk::NormIn& nin, const XW& z, uint8_t* dump, cudaStream_t st) {
  if (dmp) return launch_x1<NORM, true, 1, 2>(xq, xs, M, segs, K, wk, ctas, nin, z, dump, st);
  switch (cfg) {
    case 12: launch_x1<NORM, false, 1, 2>(xq, xs, M, segs, K, wk, ctas, nin, z, dump, st); break;
    case 14: launch_x1<NORM, false, 1, 4>(xq, xs, M, segs, K, wk, ctas, nin, z, dump, st); break;
    case 22: launch_x1<NORM, false, 2, 2>(xq, xs, M, segs, K, wk, ctas, nin, z, dump, st); break;
    case 24: launch_x1<NORM, false, 2, 4>(xq, xs, M, segs, K, wk, ctas, nin, z, dump, st); break;
    case 42: launch_x1<NORM, false, 4, 2>(xq, xs, M, segs, K, wk, ctas, nin, z, dump, st); break;
    case 44: launch_x1<NORM, false, 4, 4>(xq, xs, M, segs, K, wk, ctas, nin, z, dump, st); break;
    default: die("cfg %d", cfg);
  }
}

// ------------------------------------------------------------------ run
// Median time (us) per launch of fn(copy) over rotating copies, launches captured in a graph
// (bench_kernels.cu's time_us).
double time_us(int copies, const std::function<void(int, cudaStream_t)>& fn, int launches = 0) {
  cudaStream_t st;
  CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  if (!launches) launches = std::max(copies, 40);
  for (int i = 0; i < copies; ++i) fn(i, st);
  CUDA_CHECK(cudaStreamSynchronize(st));
  cudaGraph_t graph;
  cudaGraphExec_t exec;
  CUDA_CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
  for (int i = 0; i < launches; ++i) fn(i % copies, st);
  CUDA_CHECK(cudaStreamEndCapture(st, &graph));
  CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a));
  CUDA_CHECK(cudaEventCreate(&b));
  std::vector<double> ts;
  for (int r = 0; r < 9; ++r) {
    CUDA_CHECK(cudaEventRecord(a, st));
    CUDA_CHECK(cudaGraphLaunch(exec, st));
    CUDA_CHECK(cudaEventRecord(b, st));
    CUDA_CHECK(cudaEventSynchronize(b));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    ts.push_back(ms * 1000.0 / launches);
  }
  std::sort(ts.begin(), ts.end());
  cudaGraphExecDestroy(exec), cudaGraphDestroy(graph), cudaEventDestroy(a), cudaEventDestroy(b), cudaStreamDestroy(st);
  return ts[ts.size() / 2];
}

template <class T>
T* to_dev(const std::vector<T>& v, size_t pad = 0) {
  T* p;
  CUDA_CHECK(cudaMalloc(&p, v.size() * sizeof(T) + pad));
  CUDA_CHECK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
  if (pad) CUDA_CHECK(cudaMemset(reinterpret_cast<uint8_t*>(p) + v.size() * sizeof(T), 0, pad));
  return p;
}

uint32_t hash32(uint32_t i, uint32_t seed) {
  uint32_t h = i * 2654435761u ^ seed;
  h ^= h >> 15;
  h *= 2246822519u;
  h ^= h >> 13;
  return h;
}

int run(const std::string& dir, std::vector<std::string> sites, std::vector<int> rows_t) {
  int sms = 0, dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
  if (sms != kSMs) die("the runs were cut for %d SMs, this GPU has %d", kSMs, sms);
  if (sites.empty())
    for (const Site& s : kSites) sites.push_back(s.name);
  if (rows_t.empty()) rows_t = {1, 6};
  for (const std::string& name : sites) {
    const Site* site = nullptr;
    for (const Site& s : kSites)
      if (name == s.name) site = &s;
    if (!site) die("no site %s", name.c_str());
    Blob b = load(dir + "/" + name + ".zb");
    const int nseg = (int)b.N.size(), K = b.K, wk = b.wk, nv = (int)b.ctas.size();
    const int nl = (int)b.L.size();
    // copies: the layers, repeated until a rotation reads >= 256 MB (L2 24 MB + a 16 MB SLC)
    const size_t raw1 = b.L[0].W.size();
    const int rep = (int)std::max<size_t>(1, ((256u << 20) + raw1 * nl - 1) / (raw1 * nl));
    const int copies = nl * rep;
    struct Dev {
      uint8_t* W;
      std::vector<uint8_t*> S;
      std::vector<uint32_t*> words, off;  // per grid
    };
    std::vector<Dev> d(copies);
    uint16_t* table = to_dev(std::vector<uint16_t>(b.C.table, b.C.table + kTab));
    uint64_t raw_all = 0;
    std::vector<uint64_t> z_all(nv, 0);
    for (int i = 0; i < copies; ++i) {
      const Layer& l = b.L[i % nl];
      d[i].W = to_dev(l.W);
      for (int s = 0; s < nseg; ++s) d[i].S.push_back(to_dev(l.S[s]));
      for (int v = 0; v < nv; ++v) {
        d[i].words.push_back(to_dev(l.z[v].words, 17 * 512));  // the ring reads up to S chunks past a run's end
        d[i].off.push_back(to_dev(l.z[v].off));
        if (i < nl) z_all[v] += l.z[v].words.size() * 4 + l.z[v].off.size() * 4;
      }
      if (i < nl) raw_all += l.W.size();
    }
    // inputs for 8 rows: FP8 activations and scales (non-normed sites), bf16 rows and a norm weight (normed)
    const int MM = dsk::kLowRows;
    std::vector<uint8_t> hq((size_t)MM * K), hs((size_t)MM * (K / 32));
    for (size_t i = 0; i < hq.size(); ++i) hq[i] = (uint8_t)(hash32((uint32_t)i, 7) & 0x77);
    for (size_t i = 0; i < hs.size(); ++i) hs[i] = (uint8_t)(120 + hash32((uint32_t)i, 9) % 12);
    std::vector<__nv_bfloat16> hx((size_t)MM * K), hw(K);
    for (size_t i = 0; i < hx.size(); ++i)
      hx[i] = __float2bfloat16(((int)(hash32((uint32_t)i, 11) % 2001) - 1000) / 250.f);
    for (int i = 0; i < K; ++i) hw[i] = __float2bfloat16(0.5f + (hash32((uint32_t)i, 13) % 1000) / 1000.f);
    uint8_t* xq = to_dev(hq);
    uint8_t* xs = to_dev(hs);
    __nv_bfloat16* x = to_dev(hx);
    __nv_bfloat16* nw = to_dev(hw);
    std::vector<__nv_bfloat16*> yr(nseg), yz(nseg);
    for (int s = 0; s < nseg; ++s) {
      CUDA_CHECK(cudaMalloc(&yr[s], (size_t)MM * b.N[s] * 2));
      CUDA_CHECK(cudaMalloc(&yz[s], (size_t)MM * b.N[s] * 2));
    }
    uint8_t* dump;
    CUDA_CHECK(cudaMalloc(&dump, raw1));
    const float eps = 1e-6f;
    // today's kernel at the site (impl.h's GemvCfg), the persistent kernel at the same wk, and the fused one
    dsk::GemvCfg c_today{1, 1, wk};
    c_today.tiled = true;
    dsk::GemvCfg c_pers = c_today;
    c_pers.ring = 0;
    auto segs_of = [&](int i, std::vector<__nv_bfloat16*>& y, std::vector<dsk::Fp8Seg>& sg) {
      sg.clear();
      size_t o = 0;
      for (int s = 0; s < nseg; ++s) {
        sg.push_back(dsk::Fp8Seg{d[i].W + o, d[i].S[s], y[s], b.N[s]});
        o += (size_t)b.N[s] * K;
      }
    };
    auto today = [&](int i, int M, const dsk::GemvCfg& c, std::vector<__nv_bfloat16*>& y, cudaStream_t st) {
      std::vector<dsk::Fp8Seg> sg;
      segs_of(i, y, sg);
      if (site->norm) dsk::gemv_fp8_norm(x, nw, eps, M, sg.data(), nseg, K, c, st);
      else dsk::gemv_fp8_multi(xq, xs, M, sg.data(), nseg, K, c, st);
    };
    auto fused = [&](int i, int v, int M, int S, std::vector<__nv_bfloat16*>& y, bool dmp, cudaStream_t st) {
      std::vector<dsk::Fp8Seg> sg;
      segs_of(i, y, sg);
      ZSegs zs{};
      zs.n = nseg;
      zs.row0[0] = 0;
      for (int s = 0; s < nseg; ++s) zs.s[s] = sg[s], zs.row0[s + 1] = zs.row0[s] + b.N[s];
      const dsk::NormIn nin{x, nw, eps, nullptr, nullptr, nullptr};
      const ZW z{d[i].words[v], d[i].off[v], table};
      const int ctas = b.ctas[v];
      if (site->norm) {
        if (dmp) launch_z<true, true>(S, nullptr, nullptr, M, zs, K, wk, ctas, nin, z, dump, st);
        else launch_z<true, false>(S, nullptr, nullptr, M, zs, K, wk, ctas, nin, z, nullptr, st);
      } else {
        if (dmp) launch_z<false, true>(S, xq, xs, M, zs, K, wk, ctas, nin, z, dump, st);
        else launch_z<false, false>(S, xq, xs, M, zs, K, wk, ctas, nin, z, nullptr, st);
      }
    };
    // the grids and rings to try at M rows: every CTA of the grid resident at once
    auto configs = [&](int M) {
      std::vector<std::pair<int, int>> c;  // (grid variant, S)
      for (int v = 0; v < nv; ++v)
        for (int S : {4, 8, 16}) {
          if (z_smem(S, wk, M, K, site->norm) > 99 * 1024) continue;
          const int fit = site->norm ? z_fit<true>(S, wk, M, K) : z_fit<false>(S, wk, M, K);
          if (fit * kSMs >= b.ctas[v]) c.emplace_back(v, S);
        }
      return c;
    };
    cudaStream_t st;
    CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    CUDA_CHECK(cudaDeviceSynchronize());  // the uploads
    // 1. the decoder: every layer's bytes, as the file has them, on every grid
    std::vector<uint8_t> back(raw1);
    for (int v = 0; v < nv; ++v)
      for (int i = 0; i < nl; ++i) {
        CUDA_CHECK(cudaMemsetAsync(dump, 0xFF, raw1, st));
        fused(i, v, 1, 4, yz, true, st);
        CUDA_CHECK(cudaStreamSynchronize(st));
        CUDA_CHECK(cudaMemcpy(back.data(), dump, raw1, cudaMemcpyDeviceToHost));
        if (std::memcmp(back.data(), b.L[i].W.data(), raw1))
          die("%s layer %d ctas %d: the GPU decode differs", name.c_str(), b.layers[i], b.ctas[v]);
      }
    // 2. the outputs: today's kernel's, bit for bit, at 1..8 rows, every grid and ring depth that fits
    int checked = 0;
    for (int i = 0; i < nl; i += std::max(1, nl / 8))
      for (int M = 1; M <= MM; ++M)
        for (auto [v, S] : configs(M)) {
          for (int s = 0; s < nseg; ++s) {
            CUDA_CHECK(cudaMemsetAsync(yr[s], 0x11, (size_t)MM * b.N[s] * 2, st));
            CUDA_CHECK(cudaMemsetAsync(yz[s], 0x22, (size_t)MM * b.N[s] * 2, st));
          }
          today(i, M, c_today, yr, st);
          fused(i, v, M, S, yz, false, st);
          CUDA_CHECK(cudaStreamSynchronize(st));
          for (int s = 0; s < nseg; ++s) {
            std::vector<uint16_t> a((size_t)M * b.N[s]), c2((size_t)M * b.N[s]);
            CUDA_CHECK(cudaMemcpy(a.data(), yr[s], a.size() * 2, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(c2.data(), yz[s], c2.size() * 2, cudaMemcpyDeviceToHost));
            if (a != c2)
              die("%s layer %d M=%d ctas %d S=%d seg %d: outputs differ", name.c_str(), b.layers[i], M, b.ctas[v], S, s);
          }
          ++checked;
        }
    std::printf("%s: decode exact on %d layers x %d grids; outputs identical to today's kernel in %d cases "
                "(1-8 rows, grids, rings)\n", name.c_str(), nl, nv, checked);
    // 3. time: us per launch, rotating over the copies
    const double rawGB = (double)raw_all / nl / 1e9;
    std::printf("%s: %.3f MB raw; coded", name.c_str(), rawGB * 1e3);
    for (int v = 0; v < nv; ++v)
      std::printf(" %.3f MB at %d CTAs (%.2f %% smaller)", z_all[v] / nl / 1e6, b.ctas[v],
                  100 * (1 - (double)z_all[v] / raw_all));
    std::printf("; %d copies\n", copies);
    std::printf("%-8s %4s %9s %9s | %-48s | %9s %4s %3s %8s %8s %8s\n", "site", "rows", "today us", "persist",
                "fused us per (CTAs, ring)", "best", "CTAs", "S", "raw GB/s", "codedGB/s", "realised");
    for (int M : rows_t) {
      const double tt = time_us(copies, [&](int i, cudaStream_t s) { today(i, M, c_today, yr, s); });
      const double tp = time_us(copies, [&](int i, cudaStream_t s) { today(i, M, c_pers, yr, s); });
      double best = 1e30;
      int bv = 0, bS = 0;
      std::string all;
      for (auto [v, S] : configs(M)) {
        const double tz = time_us(copies, [&](int i, cudaStream_t s) { fused(i, v, M, S, yz, false, s); });
        char cell[48];
        std::snprintf(cell, sizeof cell, " %d,%d:%.1f", b.ctas[v], S, tz);
        all += cell;
        if (tz < best) best = tz, bv = v, bS = S;
      }
      const double zGB = (double)z_all[bv] / nl / 1e9;
      const double proj = tt * zGB / rawGB;  // the projection: today's time on the coded bytes
      std::printf("%-8s %4d %9.2f %9.2f |%-49s | %9.2f %4d %3d %8.1f %8.1f %7.0f %%\n", name.c_str(), M, tt, tp,
                  all.c_str(), best, b.ctas[bv], bS, rawGB * 1e6 / best, zGB * 1e6 / best,
                  100 * (tt - best) / (tt - proj));
      std::fflush(stdout);
    }
    // ---- the fixed-length code: encode each layer (host), check the bytes and the outputs, time
    {
      const int ntile = std::accumulate(b.N.begin(), b.N.end(), 0) / 8;
      struct XDev {
        uint8_t* fix;
        uint32_t* flg;
        uint16_t* side;
        uint32_t* soff;
        uint32_t bm;
      };
      std::vector<XDev> xd(copies);
      uint64_t x_all = 0, flagged = 0, chunks = 0;
      std::vector<XEnc> xe(nl);
      for (int i = 0; i < nl; ++i) {
        xe[i] = x_encode(b.L[i].W.data(), ntile, K, wk);
        if (!x_check(b.L[i].W.data(), ntile, K, wk, xe[i])) die("%s layer %d: the fixed code's CPU decode differs", name.c_str(), b.layers[i]);
        x_all += xe[i].bytes() - 128, flagged += xe[i].flagged, chunks += xe[i].chunks;
      }
      for (int i = 0; i < copies; ++i) {
        const XEnc& e = xe[i % nl];
        xd[i] = XDev{to_dev(e.fix, 4096), to_dev(e.flg, 4096), to_dev(e.side, 4096), to_dev(e.soff, 4096),
                     (uint32_t)e.B * 0x01010101u};
      }
      CUDA_CHECK(cudaDeviceSynchronize());
      auto fx = [&](int i, int cpm, int M, std::vector<__nv_bfloat16*>& y, bool dmp, cudaStream_t s2) {
        std::vector<dsk::Fp8Seg> sg;
        segs_of(i, y, sg);
        ZSegs zs{};
        zs.n = nseg;
        zs.row0[0] = 0;
        for (int s = 0; s < nseg; ++s) zs.s[s] = sg[s], zs.row0[s + 1] = zs.row0[s] + b.N[s];
        const dsk::NormIn nin{x, nw, eps, nullptr, nullptr, nullptr};
        const XW z{xd[i].fix, xd[i].flg, xd[i].side, xd[i].soff, xd[i].bm};
        const int ctas = std::min(ntile, (cpm / 10) * kSMs);
        if (site->norm) launch_x<true>(cpm, dmp, nullptr, nullptr, M, zs, K, wk, ctas, nin, z, dump, s2);
        else launch_x<false>(cpm, dmp, xq, xs, M, zs, K, wk, ctas, nin, z, dump, s2);
      };
      // configurations: 10 * (CTAs per SM) + (steps a buffer); a run's flags must fit in kXF
      std::vector<int> cpms;
      for (int c : {1, 2, 4}) {
        if (c == 4 && 32 * wk > 256) continue;
        const int ctas = std::min(ntile, c * kSMs), P = K / 64;
        if (((ntile + ctas - 1) / ctas + 1) * ((P + wk - 1) / wk) > kXF) continue;
        cpms.push_back(10 * c + 2), cpms.push_back(10 * c + 4);
      }
      if (cpms.empty()) die("%s: no grid whose runs fit %d steps of flags", name.c_str(), kXF);
      for (int i = 0; i < nl; ++i) {
        CUDA_CHECK(cudaMemsetAsync(dump, 0xFF, raw1, st));
        fx(i, cpms[0], 1, yz, true, st);
        CUDA_CHECK(cudaStreamSynchronize(st));
        CUDA_CHECK(cudaMemcpy(back.data(), dump, raw1, cudaMemcpyDeviceToHost));
        if (std::memcmp(back.data(), b.L[i].W.data(), raw1)) die("%s layer %d: the fixed code's GPU decode differs", name.c_str(), b.layers[i]);
      }
      int xchecked = 0;
      for (int i = 0; i < nl; ++i)
        for (int M = 1; M <= MM; ++M)
          for (int cpm : cpms) {
            if (x_smem(wk, M, K, site->norm) > 99 * 1024) continue;
            for (int s = 0; s < nseg; ++s) {
              CUDA_CHECK(cudaMemsetAsync(yr[s], 0x11, (size_t)MM * b.N[s] * 2, st));
              CUDA_CHECK(cudaMemsetAsync(yz[s], 0x22, (size_t)MM * b.N[s] * 2, st));
            }
            today(i, M, c_today, yr, st);
            fx(i, cpm, M, yz, false, st);
            CUDA_CHECK(cudaStreamSynchronize(st));
            for (int s = 0; s < nseg; ++s) {
              std::vector<uint16_t> a((size_t)M * b.N[s]), c2((size_t)M * b.N[s]);
              CUDA_CHECK(cudaMemcpy(a.data(), yr[s], a.size() * 2, cudaMemcpyDeviceToHost));
              CUDA_CHECK(cudaMemcpy(c2.data(), yz[s], c2.size() * 2, cudaMemcpyDeviceToHost));
              if (a != c2) die("%s layer %d M=%d cpm %d seg %d: the fixed code's outputs differ", name.c_str(), b.layers[i], M, cpm, s);
            }
            ++xchecked;
          }
      const double xGB = (double)x_all / nl / 1e9;
      std::printf("FIX %s: every byte decoded exactly on %d layers (CPU and GPU); outputs identical to today's kernel in %d "
                  "cases (1-8 rows x grids). stored %.3f MB of %.3f MB raw (%.2f %% smaller), %.1f %% of lane chunks flagged\n",
                  name.c_str(), nl, xchecked, xGB * 1e3, rawGB * 1e3, 100 * (1 - xGB / rawGB), 100.0 * flagged / chunks);
      std::printf("FIX %-8s %4s %9s %9s | %-56s | %8s %9s %9s %9s %8s\n", "site", "rows", "today us", "persist", "fixed us per (10 x CTAs/SM + steps a buffer)",
                  "best us", "rawGB/s", "storedGB/s", "ms/GB raw", "vs best");
      for (int M : rows_t) {
        // interleaved: today, persistent, fixed, twice; the smaller of each
        double tt = 1e30, tp = 1e30;
        std::vector<double> tx(cpms.size(), 1e30);
        for (int rep2 = 0; rep2 < 2; ++rep2) {
          tt = std::min(tt, time_us(copies, [&](int i, cudaStream_t s) { today(i, M, c_today, yr, s); }));
          tp = std::min(tp, time_us(copies, [&](int i, cudaStream_t s) { today(i, M, c_pers, yr, s); }));
          for (size_t k = 0; k < cpms.size(); ++k) {
            if (x_smem(wk, M, K, site->norm) > 99 * 1024) continue;
            tx[k] = std::min(tx[k], time_us(copies, [&](int i, cudaStream_t s) { fx(i, cpms[k], M, yz, false, s); }));
          }
        }
        std::string all;
        double best = 1e30;
        for (size_t k = 0; k < cpms.size(); ++k) {
          char cell[32];
          std::snprintf(cell, sizeof cell, " %d:%.2f", cpms[k], tx[k]);
          all += cell;
          best = std::min(best, tx[k]);
        }
        const double base = std::min(tt, tp);
        std::printf("FIX %-8s %4d %9.2f %9.2f |%-57s | %8.2f %9.1f %9.1f %9.4f %+7.2f %%\n", name.c_str(), M, tt, tp, all.c_str(),
                    best, rawGB * 1e6 / best, xGB * 1e6 / best, best / 1e3 / rawGB, 100 * (base - best) / base);
        std::fflush(stdout);
      }
      for (auto& e : xd) cudaFree(e.fix), cudaFree(e.flg), cudaFree(e.side), cudaFree(e.soff);
    }
    for (auto& e : d) {
      cudaFree(e.W);
      for (auto p : e.S) cudaFree(p);
      for (auto p : e.words) cudaFree(p);
      for (auto p : e.off) cudaFree(p);
    }
    for (int s = 0; s < nseg; ++s) cudaFree(yr[s]), cudaFree(yz[s]);
    cudaFree(table), cudaFree(xq), cudaFree(xs), cudaFree(x), cudaFree(nw), cudaFree(dump);
    cudaStreamDestroy(st);
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc >= 4 && !std::strcmp(argv[1], "prep")) {
    std::vector<int> layers;
    if (argc >= 5) {
      for (const char* p = argv[4]; *p;) {
        layers.push_back(std::atoi(p));
        while (*p && *p != ',') ++p;
        if (*p) ++p;
      }
    } else {
      for (int i = 0; i < 40; ++i) layers.push_back(i);
    }
    return prep(argv[2], argv[3], layers);
  }
  if (argc >= 3 && !std::strcmp(argv[1], "run")) {
    std::vector<std::string> sites;
    std::vector<int> rows;
    for (int i = 3; i < argc; ++i) {
      if (argv[i][0] >= '0' && argv[i][0] <= '9') rows.push_back(std::atoi(argv[i]));
      else sites.push_back(argv[i]);
    }
    return run(argv[2], sites, rows);
  }
  die("usage: bench_wcomp prep RANKFILE OUTDIR [l0,l1,..] | run OUTDIR [site...] [rows...]");
}
