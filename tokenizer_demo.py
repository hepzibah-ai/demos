# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "tokenizers",
#     "huggingface_hub",
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

        Type anything in the box below, then **click outside it** to update.

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
    from tokenizers import Tokenizer
    from huggingface_hub import hf_hub_download

    _path = hf_hub_download("deepseek-ai/DeepSeek-V3", "tokenizer.json")
    tokenizer = Tokenizer.from_file(_path)
    return (tokenizer,)


@app.cell
def _(mo):
    sample_text = mo.ui.text_area(
        value="The cat in the chartreuse Stetson sat on the mat.\n那只戴着黄绿色斯泰森帽的猫坐在垫子上。\nシャルトルーズのステットソンをかぶった猫がマットの上に座った。",
        label="Enter text to tokenize:",
        full_width=True,
        rows=5,
    )
    sample_text
    return (sample_text,)


@app.cell
def _(sample_text, tokenizer, mo):
    import html as _html

    _input_text = sample_text.value
    _enc = tokenizer.encode(_input_text)
    encoding = _enc.ids

    # Decode each token ID back to readable text (byte-level BPE tokens
    # are unreadable for CJK — this reverses the byte-to-unicode mapping)
    tokens = []
    for _tid in encoding:
        _decoded = tokenizer.decode([_tid])
        if _decoded.startswith(" "):
            _decoded = "·" + _decoded[1:]
        tokens.append(_decoded)

    # Build a colored token display
    colors = [
        "#FFE0B2", "#B3E5FC", "#C8E6C9", "#F8BBD0",
        "#D1C4E9", "#FFCCBC", "#B2DFDB", "#FFF9C4",
        "#E1BEE7", "#DCEDC8",
    ]
    colored = " ".join(
        f'<span style="background:{colors[i % len(colors)]};'
        f'padding:2px 4px;border-radius:3px;margin:1px;'
        f'display:inline-block;font-family:monospace">{_html.escape(t)}</span>'
        for i, t in enumerate(tokens)
    )

    mo.md(
        f"""
        ### Tokens: {len(encoding)}

        {colored}

        Characters: **{len(_input_text)}** → Tokens: **{len(encoding)}** (compression: **{len(_input_text)/len(encoding):.1f}x**)

        *The `·` marks a leading space — spaces are part of the token, not separate.*
        """
    )
    return (encoding, tokens)


@app.cell
def _(encoding, tokens, tokenizer, mo):
    # Token ID table
    _detail_rows = [
        f"| {i} | `{t}` | {tid} |"
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
        The vocabulary has **{tokenizer.get_vocab_size():,}** entries.
        """
    )
    return


@app.cell
def _(tokenizer, mo):
    mo.md(
        """
        ## Compare tokenization

        The same sentence in three languages — notice how token counts diverge:
        """
    )

    examples = [
        ("English", "The cat in the chartreuse Stetson sat on the mat."),
        ("Mandarin", "那只戴着黄绿色斯泰森帽的猫坐在垫子上。"),
        ("Japanese", "シャルトルーズのステットソンをかぶった猫がマットの上に座った。"),
        ("Python code", "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"),
        ("Rare words", "pneumonoultramicroscopicsilicovolcanoconiosis"),
        ("Numbers", "3.14159265358979323846"),
    ]

    _cmp_rows = []
    for label, ex_text in examples:
        _ex_enc = tokenizer.encode(ex_text)
        n_toks = len(_ex_enc.ids)
        ratio = len(ex_text) / n_toks if n_toks else 0
        _tok_strs = []
        for _tid in _ex_enc.ids[:8]:
            _d = tokenizer.decode([_tid])
            if _d.startswith(" "):
                _d = "·" + _d[1:]
            _tok_strs.append(_d)
        preview = " ".join(f"`{t}`" for t in _tok_strs)
        if len(_ex_enc.ids) > 8:
            preview += " ..."
        _cmp_rows.append(f"| {label} | {len(ex_text)} chars | {n_toks} tokens | {ratio:.1f}x | {preview} |")

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
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
