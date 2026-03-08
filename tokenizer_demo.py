# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "transformers",
# ]
# ///

import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        """
        # What's a Token?

        LLMs don't see characters or words — they see **tokens**, subword chunks
        chosen by a tokenizer. This notebook uses
        [DeepSeek's actual tokenizer](https://github.com/ser163/deepseek_v3_tokenizer_calc/blob/master/deepseek_tokenizer.py)
        so you can see exactly how a production LLM breaks text apart.

        The algorithm underneath is
        [Byte Pair Encoding](https://en.wikipedia.org/wiki/Byte_pair_encoding) (BPE):
        start with characters, repeatedly merge the most common pair, stop when you
        reach your vocabulary size. DeepSeek's tokenizer has ~100K merges learned from
        a massive training corpus. For the algorithm from scratch, see
        [nanochat](https://github.com/karpathy/nanochat).
        """
    )
    return


@app.cell
def _():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "deepseek-ai/DeepSeek-V3",
        trust_remote_code=True,
    )
    return (tokenizer,)


@app.cell
def _(mo):
    sample_text = mo.ui.text_area(
        value="The cat sat on the mat. The cat sat on the hat.",
        label="Enter text to tokenize:",
        full_width=True,
        rows=3,
    )
    sample_text
    return (sample_text,)


@app.cell
def _(sample_text, tokenizer, mo):
    _input_text = sample_text.value
    encoding = tokenizer.encode(_input_text)
    tokens = [tokenizer.decode([tid]) for tid in encoding]

    # Build a colored token display
    colors = [
        "#FFE0B2", "#B3E5FC", "#C8E6C9", "#F8BBD0",
        "#D1C4E9", "#FFCCBC", "#B2DFDB", "#FFF9C4",
        "#E1BEE7", "#DCEDC8",
    ]
    colored = " ".join(
        f'<span style="background:{colors[i % len(colors)]};'
        f'padding:2px 4px;border-radius:3px;margin:1px;'
        f'display:inline-block;font-family:monospace">{repr(t)}</span>'
        for i, t in enumerate(tokens)
    )

    mo.md(
        f"""
        ### Tokens: {len(encoding)}

        {colored}

        Characters: **{len(_input_text)}** → Tokens: **{len(encoding)}** (compression: **{len(_input_text)/len(encoding):.1f}x**)
        """
    )
    return (encoding, tokens)


@app.cell
def _(encoding, tokens, tokenizer, mo):
    # Token ID table
    _detail_rows = [
        f"| {i} | `{repr(t)}` | {tid} |"
        for i, (t, tid) in enumerate(zip(tokens, encoding))
    ]
    _detail_table = "\n".join(_detail_rows)

    mo.md(
        f"""
        ### Token details

        | # | Token | ID |
        |---|-------|----|
        {_detail_table}

        Each token maps to an integer ID — this is what the model actually sees.
        The vocabulary has **{tokenizer.vocab_size:,}** entries.
        """
    )
    return


@app.cell
def _(tokenizer, mo):
    mo.md(
        """
        ## Compare tokenization

        Tokenization efficiency varies dramatically by content type. The examples
        below show why:
        """
    )

    examples = [
        ("English prose", "The quick brown fox jumps over the lazy dog."),
        ("Python code", "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"),
        ("Rare words", "pneumonoultramicroscopicsilicovolcanoconiosis"),
        ("Japanese", "東京は日本の首都です。"),
        ("Repeated chars", "aaaaaaaaaaaaaaaaaaaaaaaaa"),
        ("Numbers", "3.14159265358979323846"),
        ("Mixed", "H0 tile uses a 4-bit MAC with int8 accumulator"),
    ]

    _cmp_rows = []
    for label, ex_text in examples:
        toks = tokenizer.encode(ex_text)
        ratio = len(ex_text) / len(toks) if toks else 0
        tok_strs = [repr(tokenizer.decode([t])) for t in toks]
        preview = " ".join(tok_strs[:8])
        if len(tok_strs) > 8:
            preview += " ..."
        _cmp_rows.append(f"| {label} | {len(ex_text)} chars | {len(toks)} tokens | {ratio:.1f}x | {preview} |")

    _cmp_table = "\n".join(_cmp_rows)

    mo.md(
        f"""
        | Type | Length | Tokens | Ratio | First tokens |
        |------|--------|--------|-------|-------------|
        {_cmp_table}
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## Why this matters

        Tokenization determines what the model can "see":

        - **Common English words** become single tokens — efficient
        - **Rare words** get split into subword pieces — expensive
        - **Code** tokenizes differently from prose (indentation, operators)
        - **Non-English text** often tokenizes less efficiently (more tokens per word)
        - **Numbers** are split digit-by-digit or in small groups — arithmetic is hard

        This explains some otherwise baffling LLM failures:
        - **Counting letters** in a word is hard because the model doesn't see individual letters
        - **Anagram puzzles** are nearly impossible for the same reason
        - **The model is better at English** partly because English gets more efficient tokenization

        ### Try it yourself

        Change the text above and watch the token count change:
        - Try a long common word vs a short rare one
        - Try the same sentence in English and another language
        - Try `"strawberry"` — can you see why letter-counting is hard?

        ### Going deeper

        - [DeepSeek's tokenizer source](https://github.com/ser163/deepseek_v3_tokenizer_calc/blob/master/deepseek_tokenizer.py) —
          the production code using HuggingFace `transformers`
        - [nanochat](https://github.com/karpathy/nanochat) tokenizer — builds BPE from scratch,
          handling unicode, special tokens, and regex pre-splitting
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
