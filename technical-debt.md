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
| 4 | `high_dimensions_demo.py` | High Dimensions | `/high-dimensions` | Deployed (beta) ✓ |
| 5 | `precision_energy_demo.py` | Precision and Energy | `/precision-energy` | Deployed (alpha) |
| 5a | — | Microscaling | — | Planned (deep-dive, may go to sim0) |
| 6 | — | PCA | — | Planned |
| 7 | — | Clustering | — | Planned |
| TBD | — | How Were These Vectors Trained? | — | Placeholder |
| TBD | — | Ideograms as Embeddings | — | Placeholder (essay?) |

Wiki companion pages: [How LLMs See Words](https://github.com/hepzibah-ai/general/wiki/How-LLMs-See-Words) links to notebooks 1–3.

---

## Notebook 4: "High Dimensions"

Next to build. The story in three acts:
- **Act I** (§1–2): Where we came from — one-hot codes, the curse
- **Act II** (§3–5): The geometry — what high-D space is actually like
- **Act III** (§6–8): Structure — how training exploits the geometry,
  and why sparsity is a fundamental property (not a hardware trick)

No hardware discussion here — just build the understanding. Later
notebooks (precision/energy, microscaling) and sim0 analyses (seven
dwarfs) harvest these ideas.

### Sections:
1. **One-hot: where we came from**: before embeddings, every word was a
   one-hot vector — vocabulary of 50K words → 50K-dim vector with a
   single 1. Maximally sparse, zero structure. Every word is equidistant
   from every other (all pairwise distances = √2). This is the baseline
   the rest of the notebook improves on.
2. **The curse of dimensionality**: Bellman's original insight — volume
   grows exponentially with dimension, so uniform sampling becomes
   hopeless. The one-hot representation is the curse in its purest form:
   50K dimensions, all information in 1 of them. Show: fraction of
   hypercube volume within ε of the surface vs D (same calculation as
   sphere shell, but starting from the cube makes the "curse" framing
   concrete). Reference: Bellman 1961; Hamming ch.9 for the accessible
   version.
3. **Random vectors are nearly orthogonal**: 1000 random unit vectors in
   D dims, histogram of pairwise cosines. Slider for dimension — watch
   histogram sharpen around zero. Compare random vs GloVe side by side.
   Key insight: in high-D, "most directions are roughly perpendicular
   to most other directions." This is concentration of measure.
4. **Everything lives on the shell**: fraction of sphere volume within ε
   of surface vs D. By D=50 it's ~1.0. Coordinates of a random unit
   vector ≈ N(0, 1/√D) — show histogram. This is why norms concentrate.
5. **Heavy tails — what's NOT Gaussian**: pool all GloVe coordinates,
   show they're heavier-tailed than the matched Gaussian. Two distinct
   mechanisms:
   - **GloVe**: tails from language statistics (Zipf's law — a few words
     dominate co-occurrence). No regularization; the tails are structural.
     Reference: Pennington et al. 2014; Wikipedia "Zipf's law"; Powers
     1998.
   - **Transformer activations**: more extreme outliers emerging at scale
     (~6.7B+ params). Reference: Dettmers et al. 2022 "LLM.int8()";
     Anthropic "Toy Models of Superposition" 2022.
   - **Visualization**: log-log survival plot (complementary CDF). GloVe
     coordinates vs matched Gaussian on log-log axes. Gaussian drops like
     a cliff; GloVe sits above in the tails. The log abscissa has physical
     meaning (floating-point codes are log-spaced). Draw clipping
     boundaries for fp8/int8 — foreshadows notebook 5.
   - Frame honestly: "GloVe shows mild heavy tails from Zipf; transformer
     activations show extreme outliers from a different mechanism."
6. **The curse becomes a blessing — linear representations**: trained
   embeddings have structure where random vectors don't. The network
   learns to encode concepts as *directions*. King − man + woman = queen
   because gender is a direction. The high-D space that cursed one-hot
   codes is exactly what makes this possible: there's room for thousands
   of near-orthogonal meaningful directions in 50 dimensions.
   References: Mikolov 2013, Elhage et al. "Toy Models of Superposition".
7. **Superposition and sparsity**: more concepts than dimensions, packed
   via near-orthogonal directions. Callback to embedding demo's
   orthogonality section. Key insight: if representations are
   superposed, activations are naturally *sparse* — most features are
   off for any given input. This isn't an accident or a compression
   trick; it's a fundamental consequence of how high-D representations
   work. (Don't connect to hardware here — just establish sparsity as a
   property of the representations themselves.)
8. **Nearest-neighbor collapse**: for random data, farthest/nearest
   distance ratio → 1 as D grows (the curse). But NOT for trained
   embeddings — show both. This is why embeddings are useful despite
   high dimensionality: the learned structure defeats the curse.
7. **References**: Hamming ch.9, 3Blue1Brown higher-dimensions video,
   Elhage et al., Dettmers et al., Pennington et al. 2014, Powers 1998 /
   Wikipedia (Zipf's law).

---

## Notebook 5: "Precision and Energy"

Story arc: you've seen that low precision preserves geometry (notebook 4).
Now: *why* low precision, *how* number formats work, and *what it costs*
in energy. Develop the clearest story first; redact IP later if needed.

### Sections:
1. **Quantize the dot product**: fp32, fp16, E4M3, E2M5, INT8, E2M1
   side by side on same GloVe word pair. Interactive: pick two words,
   see cosine at each precision. The "barely moves" moment.
2. **The ExMy family and scaling**: all formats are one family — trade
   exponent bits for mantissa bits within a fixed budget. Present the
   full landscape: E1M6 (≈symmetric INT8), E2M5 (inference sweet spot),
   E4M3 (OCP standard), E5M2 (training — gradient dynamic range),
   E2M3 (6b), E2M1 (4b/MXFP4). Interactive sweep of scale factor
   (2σ–5σ + peak) vs RMS error for Gaussian data — shows that optimal
   scaling depends on both format and data distribution. E2M5 beats E4M3
   by ~3.5× at optimal scaling; E2M3 is worse than E4M3 despite same
   mantissa (subnormal degradation from narrow exponent). Starter code
   in `scratch/quant_noise_test.py`.
3. **Distribution meets number format**: bring back the log-log survival
   plot from notebook 4. Overlay representable values for each format
   as tick marks or code-density rug plot. fp codes are log-spaced →
   dense where the data is dense. INT8 codes are uniform → wastes codes
   in the empty middle. Draw clipping boundaries. Interactive: dropdown
   to switch format and watch the clipping line move.
4. **What does a MAC cost?**: gate-level anatomy of a multiply-accumulate.
   Dadda tree (AND gates → half/full adders → accumulator). Count the
   gates, show energy per toggle from 5nm cell data (~0.3–1.4 fJ).
   Component breakdown: multiplier ~7 fJ, shifter ~4 fJ, accumulator
   ~11 fJ, latches/control ~14 fJ → total ~45 fJ/MAC @0.75V 5nm.
   Voltage scaling to 0.4V → ~13 fJ/MAC. Process scaling 22nm → ~64 fJ.
   Source: h0-pe-8b/docs/area-power-estimate.md.
5. **The energy budget — where do the joules go?**: pie chart of compute
   vs data movement vs overhead. Fetch dominates: "moving a byte costs
   more than multiplying it." CRAM + NoC overhead is +30–50 fJ/OP on
   top of ~32 fJ/OP core. System total: 62–82 fJ/OP → 12–16 TOPS/W.
   Compare: 4-bit (h0-pe-4b) gets ~7 fJ/cycle → ~63 TOPS/W.
   Reference Boqueria/Speed AI MLPerf as existence proof.
6. **Why custom silicon**: the operation is uniform (MAC), precision is
   low (4–8 bits), volume is enormous. GPUs waste energy on flexibility
   the workload doesn't need. This is why purpose-built inference
   silicon can be 10–50× more efficient per watt.

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
