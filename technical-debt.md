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
| 4 | — | High Dimensions | — | Planned |
| 5 | — | Precision and Energy | — | Planned |
| 5a | — | Microscaling | — | Planned (deep-dive, may go to sim0) |
| 6 | — | PCA | — | Planned |
| 7 | — | Clustering | — | Planned |
| TBD | — | How Were These Vectors Trained? | — | Placeholder |
| TBD | — | Ideograms as Embeddings | — | Placeholder (essay?) |

Wiki companion pages: [How LLMs See Words](https://github.com/hepzibah-ai/general/wiki/How-LLMs-See-Words) links to notebooks 1–3.

---

## Notebook 4: "High Dimensions"

Next to build. The arc: high-D geometry sets the rules (§1–3), training
finds structure within those rules (§4–6).

### Sections:
1. **Random orthogonality**: 1000 random unit vectors in D dims, histogram
   of pairwise cosines. Slider for dimension — watch histogram sharpen.
   Compare random vs GloVe side by side.
2. **Sphere shell volume**: fraction within ε of surface vs D. Interactive
   slider. By D=50 it's ~1.0.
3. **Gaussian coordinates and heavy tails**: random unit vector in D dims
   has coords ≈ N(0, 1/√D). Show this, then show real GloVe coordinates —
   heavier tails. Two distinct mechanisms to disentangle:
   - **GloVe's heavy tails** come from language statistics (Zipf's law: a
     few words dominate co-occurrence counts, shaping the embedding
     geometry). GloVe has no dropout or L1/L2 regularization — the tails
     are structural. Reference: Pennington et al. 2014; for Zipf, point to
     Wikipedia "Zipf's law" and Powers 1998 "Applications and explanations
     of Zipf's law" for depth.
   - **Transformer activation outliers** are a different, more extreme
     phenomenon emerging at scale (~6.7B+ params): specific dimensions fire
     with much larger magnitude. Reference: Dettmers et al. 2022
     "LLM.int8()" (canonical); Anthropic "Toy Models of Superposition"
     2022 (partial theoretical explanation via outlier features).
   - **Visualization**: log-log survival plot (complementary CDF). Pool all
     GloVe coordinates, plot P(|x| > t) vs t on log-log axes alongside a
     matched Gaussian. Gaussian drops like a cliff; GloVe sits above it in
     the tails. The log abscissa has physical meaning: floating-point codes
     are logarithmically spaced, so the x-axis *is* the number line of an
     fp format. Draw a vertical line at the clipping boundary of fp8/int8 —
     everything to the right gets crushed. This foreshadows notebook 5
     directly: the plot shows *why* fp8's log spacing wastes fewer codes in
     the empty middle and keeps more in the populated tails. Skip kurtosis
     — the plot teaches more than a number.
   - Frame honestly: "GloVe shows mild heavy tails from Zipf; transformer
     activations show extreme outliers from a different mechanism — and
     that's the one that matters for hardware."
4. **Linear representation hypothesis**: why trained embeddings have
   structure when random vectors don't. Network learns to encode concepts
   as directions. King − man + woman = queen because gender is a direction.
   References: Mikolov 2013, Elhage et al. "Toy Models of Superposition".
5. **Superposition**: more concepts than dimensions, packed via
   near-orthogonal directions. Callback to embedding demo's orthogonality
   section.
6. **Nearest-neighbor collapse**: for random data, farthest/nearest
   distance ratio → 1 as D grows. But NOT for trained embeddings — show
   both.
7. **References**: Hamming ch.9, 3Blue1Brown higher-dimensions video,
   Elhage et al., Dettmers et al., Pennington et al. 2014, Powers 1998 /
   Wikipedia (Zipf's law).

---

## Notebook 5: "Precision and Energy"

### Sections:
1. **Quantize the dot product**: float32, float16, fp8, int8, int4 side by
   side on same GloVe word pair. Show cosine similarity barely moves at
   fp8/int8, gets noisy at int4.
2. **Distribution meets number format**: bring back the log-log survival
   plot from notebook 4. Overlay representable values for each format
   (fp32, fp16, fp8, int8, int4) as tick marks or code-density rug plot.
   fp8's codes are log-spaced → dense where the data is dense, sparse
   where it's sparse. int8's codes are uniform → wastes codes in the empty
   middle, runs out in the tails. Draw clipping boundaries. Interactive:
   slider or dropdown to switch format and watch the clipping line move.
   (Use standard E4M3/E5M2 fp8 — our specific implementation stays in
   sim0.)
3. **Energy table**: 50fJ/OP = 100fJ/MAC at 20TOPS/W. Untether "Boqueria"
   (Speed AI) MLPerf result as existence proof. "We're the inheritors."
4. **Why custom silicon**: operation is uniform (MAC), precision is low
   (4–8 bits), volume is enormous. General-purpose hardware wastes energy
   on flexibility.

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

## Notebook 7: "Clustering"

- k-means on embeddings, color by cluster in 3D plotly
- Elbow / silhouette — interactive slider for k
- Hierarchical clustering — dendrogram
- Maybe t-SNE/UMAP for "map of all words" (precompute on subset)

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

- Hamming, "The Art of Doing Science and Engineering" ch.9 (high-D)
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
