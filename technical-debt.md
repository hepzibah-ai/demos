# Interactive Demos — Technical Debt and Roadmap

This is the main planning file for the demo notebook series.

**On session start**: Read this file first.

---

## Deployment

Linode 172.105.0.10:8081 via Docker Compose. Server config in `deploy/`.
To deploy changes: `ssh 172.105.0.10 "cd ~/demos && git pull && docker-compose -f deploy/docker-compose.yml up -d --build"`

Pending: HTTPS via Caddy + DNS for `tutorials.hepzibah.ai` (Chris).

---

## Notebook Status

| # | File | Title | Route | Status |
|---|------|-------|-------|--------|
| 1 | `tokenizer_demo.py` | What's a Token? | `/tokenizer` | Deployed |
| 2 | `embedding_demo.py` | What's an Embedding? | `/embedding` | Deployed |
| 3 | `dot_product_demo.py` | The Dot Product | `/dot-product` | Deployed (beta) |
| 4 | `high_dimensions_demo.py` | High Dimensions | `/high-dimensions` | Deployed (beta) |
| 5 | `precision_energy_demo.py` | Precision and Energy | `/precision-energy` | Deployed (beta) — shipped to Takis |
| 5a | — | Microscaling | — | Planned (deep-dive, may go to sim0) |
| 6 | — | PCA | — | Planned |
| 7 | — | Clustering | — | Planned |
| TBD | — | How Were These Vectors Trained? | — | Placeholder |
| TBD | — | Ideograms as Embeddings | — | Placeholder (essay?) |

Wiki companion pages: [How LLMs See Words](https://github.com/hepzibah-ai/general/wiki/How-LLMs-See-Words) links to notebooks 1–3.

---

## Notebook 4: "High Dimensions" — shipped (beta)

8 sections: one-hot baseline → curse of dimensionality → concentration
of measure (random orthogonality, shell concentration) → heavy tails
(log-log survival, Zipf vs transformer outliers) → linear representations
→ superposition/sparsity → nearest-neighbor collapse → packing bounds
with quantization noise lines (E1M6–E2M1, numerical Monte Carlo, zoom
slider). Matplotlib sizing fix: manual PNG render to avoid bbox_inches
cropping.

---

## Notebook 5: "Precision and Energy" — shipped (beta)

6 sections as built:
1. **Format anatomy + quantize the dot product**: sign/exponent/mantissa
   explanation, subnormals, ±0. Interactive word-pair cosine table across
   fp32/E5M2/E4M3/E2M5/E1M6/E2M3/E2M1.
2. **ExMy family + scaling**: RMS error vs scale factor (2σ–5.5σ) for 6
   formats. Bar chart at current scale. E2M5 beats E4M3 ~3.5× at optimal;
   E5M2 for training, E2M5 for inference.
3. **Distribution meets format**: linear picket fence + peak-normalized
   code density vs data distribution shape comparison. Log survival plot
   with code density and data density on twin axis. Dropdown to switch
   formats.
4. **MAC energy breakdown (E4M3)**: 10-component bar from h0-pe-8b
   (45 fJ/MAC @ 5nm/0.75V). Voltage scaling (0.4V: ×0.28) and process
   scaling (22nm: ×5). Horowitz 2014 comparison table with H0 5nm/0.4V
   numbers. Multiplier quadratic commentary.
5. **System energy budget**: stacked bar (core + ~10 fJ/OP overhead,
   conservative). E4M3: 42 fJ/OP → 24 TOPS/W. E2M1: 12 fJ/OP →
   81 TOPS/W. Boqueria MLPerf reference at ~20 TOPS/W.
6. **Why custom silicon**: Amdahl's Law framing — "almost uniform" because
   SiLU/softmax/RMSNorm decompose into MACs. Low precision, enormous
   volume.

Key numbers: SRAM read 2 fJ/bit (64 fJ/32b), DRAM 5 pJ/bit (sustained).
Starter code for scaling analysis in `scratch/quant_noise_test.py`.

---

## Notebook 5a: "Microscaling: How 4-bit Actually Works"

Deep-dive branching off notebook 5. Uses real frontier-model weight data
instead of GloVe. **IP decision TBD**: public (demos repo — models are
open-weight) or private (sim0 — closer to hardware analysis).

### Sections:
1. **The problem**: you've seen the survival plot (notebook 4) and the
   clipping boundaries (notebook 5). How do real quantization formats
   solve this? Introduce block-scaled microscaling — shared exponent per
   block.
2. **MXFP4 vs NVFP4**: two answers to the same problem. MXFP4 (OCP
   standard): E2M1 + E8M0 power-of-2 scale, 32-element blocks. NVFP4
   (NVIDIA proprietary): E2M1 + E4M3 scale + F32 global, 16-element
   blocks. Side-by-side: block size, bits/weight, scale granularity.
3. **Real weight distributions**: survival plot with actual weights from
   GPT-OSS-120B (MXFP4) and DeepSeek-R1 (NVFP4).
   - E8M0 scale distribution: 17–22 unique exponents out of 256 — format
     is over-provisioned.
   - E4M3 scale distribution: 71–89 unique values — finer granularity,
     better utilization, needs a multiplier instead of a bit-shift.
   - E2M1 code histograms: which of 16 codes actually appear, symmetry.
4. **What the hardware sees**: scale variation within a tensor — bank
   switching at 3.11 per 16-element group on real data vs 12 worst-case.
   Bridge to PE architecture.

### Source data (in sim0, not this repo):
- `h0-pe-4b/notebooks/gpt_oss_mxfp4_weights.ipynb`
- `h0-pe-4b/notebooks/deepseek_r1_nvfp4_weights.ipynb`
- `h0-pe-4b/docs/mxfp4-primer.md`, `nvfp4-primer.md`
- `h0-pe-4b/docs/Scaling_Statistical_Analysis_{MXFP4,NVFP4}.md`

---

## Notebook 6: "PCA"

### Sections:
1. **Warm-up**: PCA on simple 2D → 1D (correlated scatter, find the line
   of max variance). Maybe interactive rotation.
2. **Curated word set**: ~20 words spanning 3 intuitive axes
   (alive/inorganic, safe/dangerous, concrete/abstract). PCA discovers
   axes. Test whether GloVe cooperates before committing to word set.
3. **Explained variance**: bar chart of eigenvalues. First N components
   capture X%.
4. **One-liners connecting PCA/SVD/eigenvalues**:
   - PCA = directions of maximum variance
   - SVD of centered data gives PCA: right singular vectors = principal
     components
   - Singular values² / n = eigenvalues of covariance matrix = variance
     per PC
5. **Big Five personality traits**: "PCA on questionnaires discovered the
   Big Five. What might PCA on embeddings discover?"
6. **Connection to quantization**: if most variance in few dims, can
   truncate AND quantize. Compound savings.

---

## Notebook 7: "Clustering and Search"

Two sides of the same problem: finding structure (clustering) and finding
one thing fast (search). Both are responses to the curse of dimensionality
from notebook 4.

### Clustering:
- k-means on embeddings, color by cluster in 3D plotly
- Elbow / silhouette — interactive slider for k
- Hierarchical clustering — dendrogram
- Maybe t-SNE/UMAP for "map of all words" (precompute on subset)

### Vector search (approximate nearest neighbor):
- Callback to notebook 4 §8: exact nearest-neighbor breaks down in high-D
  (distance concentration). So how do vector databases actually work?
- **LSH** (locality-sensitive hashing): random projections bucket similar
  vectors together. The dot product shows up again — each hash is a
  sign(random_vector · query).
- **IVF** (inverted file index): cluster first (Voronoi cells), then only
  search nearby clusters. Clustering as a search accelerator.
- **HNSW** (hierarchical navigable small world): graph-based, greedy
  traversal. Skip-list intuition.
- Interactive: build a small index on GloVe, compare exact vs approximate
  search — show recall/speed tradeoff with a slider for number of probes.
- This is the bridge to RAG and production LLM systems: context
  engineering starts with retrieval, retrieval starts with vector search.

---

## Placeholder: "How Were These Vectors Trained?"

By notebook 4 the reader has been using GloVe for three notebooks without
knowing where the vectors came from. The curious ones will ask.

- **Word2Vec**: skip-gram and CBOW. Predict context from word (or vice
  versa). The embedding *is* the learned weight matrix. Mikolov et al.
  2013.
- **GloVe**: factorize the log co-occurrence matrix directly. Weighted
  least squares, no dropout, no L1/L2. Pointwise mutual information falls
  out of the objective. Pennington et al. 2014.
- **The connection**: both capture distributional semantics ("you shall
  know a word by the company it keeps" — Firth 1957). Different
  algorithms, same underlying signal.
- **What changed with transformers**: static → contextual embeddings. Same
  word, different vector depending on context. Training objective (predict
  next/masked token) is still distributional.

References: Mikolov et al. 2013, Pennington et al. 2014, Alammar
"Illustrated Word2Vec", Jurafsky & Martin ch.6 (free online).

---

## Placeholder: "Ideograms as Embeddings"

Chinese radicals as a human-designed embedding system: visual
sub-components encode semantic and phonetic features. 氵(water) +
每(every) = 海(sea). The radical is a basis direction; the character is a
point in "radical space." To what extent does this structure align with
learned embeddings?

Probably more essay than notebook. Needs a CJK-aware embedding model
(GloVe-50d is English words, won't help).

---

## Cross-cutting themes

- **Dot product as universal primitive**: similarity, projection,
  normalization, attention
- **Random vs trained**: high-D geometry sets the rules; training finds
  structure within them
- **Math → algorithm → silicon**: each notebook foreshadows the hardware
  payoff
- **Always use real data alongside theory**: numpy.rand() only where
  comparing to GloVe/embeddings
- All public (demos repo) — H0 implementation details stay in
  sim0/tutorials (private)

## References to weave in

- Bellman 1961, "Adaptive Control Processes" (curse of dimensionality, original source)
- Hamming, "The Art of Doing Science and Engineering" ch.9 (high-D, accessible)
- Jay Alammar, "Illustrated Word2Vec" (embeddings, already linked)
- Mikolov et al. 2013 (word2vec arithmetic, linear representations)
- Elhage et al. / Anthropic 2022, "Toy Models of Superposition"
- Dettmers et al. 2022, "LLM.int8()" (outlier features, heavy tails)
- Tim Dettmers blog "The case for 4-bit precision" (accessible version)
- 3Blue1Brown higher-dimensions video
- 3Blue1Brown "Dot products and duality" (already linked from notebook 3)
- Untether "Boqueria" / Speed AI MLPerf result (energy efficiency)
- Pennington et al. 2014 (GloVe training)
- Powers 1998 / Wikipedia (Zipf's law)
- Wikipedia: Gram-Schmidt (already linked from embedding demo)
