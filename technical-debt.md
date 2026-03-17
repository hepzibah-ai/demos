# Interactive Demos — Technical Debt and Roadmap

This is the main planning file for the demo notebook series.

**On session start**: Read this file first.

---

## Deployment

Currently: Linode 172.105.0.10:8081 via Docker Compose.
**Migrating to corporate servers** — briefing sent to Chris (2026-03-16),
see `deploy/SETUP-FOR-CHRIS.md`.

Deploy command (same on either host):
`cd ~/demos && git pull && docker-compose -f deploy/docker-compose.yml up -d --build`

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
| 6 | `pca_demo.py` | PCA | `/pca` | Deployed (beta) |
| 7 | — | Clustering & Search | `/clustering` | Building |
| 8 | — | RAG: From Search to Answers | `/rag` | Planned |
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

## Notebook 6: "PCA" — shipped (beta)

7 sections as built:
1. **2D warm-up**: rotatable correlated scatter with code snippet and
   expected PC2 = 6.6% derivation.
2. **PCA on GloVe**: 40 curated words in 6 categories. Dynamic PC axis
   descriptions (computed from category centroids, not hardcoded).
   Co-occurrence dot product table (king·his vs king·her etc.) with
   10-word sliding window explanation.
3. **Explained variance**: scree plot + cumulative variance on 5000 words.
   No sharp elbow — GloVe uses all 50 dims. ~11 PCs for 50%, ~27 for 80%.
4. **PCA = SVD = eigenvalues**: equivalence table, dot product connection.
5. **Big Five analogy**: Goldberg 1990 (1,710 subjects, 339 adjectives).
   Key point: names are after-the-fact interpretations, same as our §2
   PC labels.
6. **Truncation + quantization**: reconstruction error and cosine error
   vs number of PCs kept. 30-dim × 8-bit = 6.7× compression.
7. **References**: Pearson, Shlens tutorial, Goldberg 1990, 3Blue1Brown.

### Bridge to h0-applications
PCA raises the question: how do you *compute* SVD on custom silicon?
This is a natural entry point for h0-applications work — showing that
the same MAC array that does inference can also do iterative linear
algebra (power method, Lanczos, randomized SVD). Not a demos notebook
— belongs in sim0/tutorials where we can show the actual graph
compilation.

---

## Notebook 7: "Clustering and Search" — building

Every section has a live interactive element (slider, toggle, text input).
No passive textbook sections.

### Sections:
1. **Why search is hard**: text input for query word + slider for corpus
   size (100→10K). Brute-force nearest neighbors with visible timing.
   Shows linear scaling — motivates everything that follows.
2. **k-means**: slider for k (2→20). PCA scatter recolors live, table
   shows top-5 words per cluster. Elbow/silhouette plot — no magic k
   in a continuous embedding space.
3. **t-SNE vs PCA**: toggle between projections on same 1000-word subset.
   PCA preserves global structure, t-SNE preserves local neighborhoods.
   Precompute t-SNE during build. Needs scikit-learn dependency.
4. **IVF (inverted file index)**: clustering as search accelerator.
   Slider for nprobe (1→k). Scatter plot highlights searched cells.
   Recall % and speedup vs brute force.
5. **LSH (locality-sensitive hashing)**: slider for hash bits (4→64).
   Each bit = sign(random_vector · query) — dot product again.
   Recall vs selectivity tradeoff.
6. **HNSW**: build small graph (50–100 nodes), animate greedy walk
   from entry point to target. Slider for layers. Skip-list intuition.
7. **Bridge to RAG**: bag-of-words retrieval on curated sentences
   (average GloVe vectors). Text input for query. Honest about the
   limitation — sets up notebook 8.

### Dependencies:
- scikit-learn (t-SNE, k-means) — add to Dockerfile

---

## Notebook 8: "RAG — From Search to Answers" — planned

The full retrieval-augmented generation loop. Notebook 7 §7 is the
teaser; this is the real thing.

### Sections (draft):
1. **Word embeddings aren't enough**: bag-of-words loses word order,
   can't capture "not good" vs "good". Need sentence/paragraph embeddings.
2. **Sentence embeddings**: load a small model (all-MiniLM-L6-v2, 80MB,
   CPU-only). Compare sentence similarity — show that it captures meaning
   where bag-of-words fails.
3. **Chunking**: same document, different chunk sizes. Show that chunk
   size is a design choice (too small = no context, too big = dilutes).
   Interactive slider for chunk size with retrieval quality metric.
4. **The retrieval loop**: embed query → search index → top-k chunks.
   Build a small FAISS/IVF index on chunked documents. Interactive query.
5. **Retrieval + generation**: stuff retrieved chunks into a prompt.
   Show the assembled prompt (and optionally call Claude API to complete).
6. **What can go wrong**: retrieved chunk is irrelevant, chunk boundary
   splits a key sentence, embedding model doesn't understand domain jargon.
   Interactive examples of each failure mode.

### Dependencies (heavy):
- sentence-transformers → pulls PyTorch (~800MB). Container roughly
  doubles. Keep out of Docker build until ready.
- Optionally: anthropic SDK for live generation step.
- Alternative: use a lighter embedding model (e.g. onnxruntime + quantized
  MiniLM) to avoid PyTorch. Worth investigating.

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
