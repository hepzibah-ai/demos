# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
# ]
# ///

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        """
        # What's a Token?

        LLMs don't see characters or words — they see **tokens**, subword chunks
        chosen by a tokenizer. This notebook lets you see tokenization in action
        and understand why it matters.

        Production tokenizers (like
        [DeepSeek's](https://github.com/ser163/deepseek_v3_tokenizer_calc/blob/master/deepseek_tokenizer.py))
        use the Hugging Face `transformers` library and a pre-trained vocabulary.
        Under the hood, they all use the same core algorithm:
        [Byte Pair Encoding](https://en.wikipedia.org/wiki/Byte_pair_encoding) (BPE).

        Below we build BPE from scratch in pure Python so you can see exactly
        what's happening. The DeepSeek tokenizer does the same thing — just with
        100K+ merges learned from a massive corpus instead of our toy example.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## Step 1: Start with characters

        The simplest tokenizer splits text into individual characters (really,
        bytes). Try changing the text below and watch what happens.
        """
    )
    return


@app.cell
def _(mo):
    sample_text = mo.ui.text_area(
        value="The cat sat on the mat. The cat sat on the hat.",
        label="Enter text to tokenize:",
        full_width=True,
    )
    sample_text
    return (sample_text,)


@app.cell
def _(sample_text, mo):
    chars = list(sample_text.value)
    mo.md(
        f"""
        **Character-level tokens**: {len(chars)} tokens

        ```
        {chars}
        ```

        Every character is a separate token. This works, but it's inefficient —
        the model has to "spend" one token on every single character, and common
        words like "the" cost 3 tokens each time.
        """
    )
    return (chars,)


@app.cell
def _(mo):
    mo.md(
        """
        ## Step 2: Byte Pair Encoding (BPE)

        BPE finds the most common *pair* of adjacent tokens and merges them into
        a single new token. Repeat until you reach a target vocabulary size.

        This is how real tokenizers are trained — on a huge corpus, finding the
        most frequent pairs and merging them one by one.
        """
    )
    return


@app.cell
def _():
    from collections import Counter

    def get_pair_counts(token_list):
        """Count adjacent pairs in a list of tokens."""
        pairs = Counter()
        for i in range(len(token_list) - 1):
            pairs[(token_list[i], token_list[i + 1])] += 1
        return pairs

    def merge_pair(token_list, pair):
        """Merge all occurrences of a pair into a single token."""
        merged = pair[0] + pair[1]
        result = []
        i = 0
        while i < len(token_list):
            if i < len(token_list) - 1 and token_list[i] == pair[0] and token_list[i + 1] == pair[1]:
                result.append(merged)
                i += 2
            else:
                result.append(token_list[i])
                i += 1
        return result

    def run_bpe(text, num_merges):
        """Run BPE for a given number of merge steps. Return history."""
        tokens = list(text)
        history = [{"step": 0, "pair": None, "count": None, "tokens": list(tokens), "num_tokens": len(tokens)}]

        for step in range(1, num_merges + 1):
            pairs = get_pair_counts(tokens)
            if not pairs:
                break
            best_pair = pairs.most_common(1)[0]
            pair, count = best_pair
            tokens = merge_pair(tokens, pair)
            history.append({
                "step": step,
                "pair": pair,
                "count": count,
                "tokens": list(tokens),
                "num_tokens": len(tokens),
            })

        return history
    return run_bpe, get_pair_counts, merge_pair


@app.cell
def _(mo):
    num_merges = mo.ui.slider(
        start=0,
        stop=30,
        value=10,
        label="Number of BPE merges:",
        show_value=True,
    )
    num_merges
    return (num_merges,)


@app.cell
def _(sample_text, num_merges, run_bpe, mo):
    history = run_bpe(sample_text.value, num_merges.value)

    # Build the merge log
    merge_rows = []
    for entry in history:
        if entry["pair"] is not None:
            p = entry["pair"]
            merge_rows.append(
                f"| {entry['step']} | `{repr(p[0])}` + `{repr(p[1])}` → `{repr(p[0] + p[1])}` | {entry['count']} | {entry['num_tokens']} |"
            )

    merge_table = "\n".join(merge_rows) if merge_rows else "*(no merges yet — slide right)*"

    final = history[-1]
    token_display = " | ".join(f"`{t}`" for t in final["tokens"])

    mo.md(
        f"""
        ### Merge history

        | Step | Merge | Occurrences | Tokens remaining |
        |------|-------|-------------|-----------------|
        {merge_table}

        ### Result: {final['num_tokens']} tokens (down from {history[0]['num_tokens']} characters)

        {token_display}
        """
    )
    return (history,)


@app.cell
def _(history, mo):
    # Compression ratio
    initial = history[0]["num_tokens"]
    final = history[-1]["num_tokens"]
    ratio = initial / final if final > 0 else 0

    mo.md(
        f"""
        ### Compression

        - Characters: **{initial}**
        - Tokens after BPE: **{final}**
        - Compression ratio: **{ratio:.1f}x**

        Real tokenizers run thousands of merges on billions of words. GPT-4's
        tokenizer has ~100,000 tokens in its vocabulary. The principle is the same
        — just more merges on more data.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## Why this matters

        Tokenization determines what the model can "see":

        - **Common words** become single tokens (efficient — "the" is 1 token, not 3)
        - **Rare words** get split into pieces ("pneumonoultramicroscopicsilicovolcanoconiosis" → many tokens)
        - **Code** tokenizes differently from prose (indentation, brackets, operators)
        - **Non-English text** often tokenizes less efficiently (more tokens per word)

        This explains some otherwise baffling LLM failures:
        - Counting letters in a word is hard because the model doesn't see individual letters
        - Anagram puzzles are nearly impossible for the same reason
        - The model is better at English than other languages partly because English
          gets more efficient tokenization (more merges for common English patterns)

        ### Try it yourself

        Change the text above to see how different inputs tokenize:
        - Try `"hello"` vs `"pneumonia"` — common vs rare words
        - Try Python code: `"def hello(): return 42"`
        - Try non-English text and compare token counts
        - Try `"aaaaaaaaaa"` — what happens with repetition?

        ### Going deeper

        - [DeepSeek's tokenizer](https://github.com/ser163/deepseek_v3_tokenizer_calc/blob/master/deepseek_tokenizer.py) —
          a production tokenizer using HuggingFace `transformers`. Same BPE algorithm,
          but with a vocabulary trained on DeepSeek's full training corpus.
        - [nanochat](https://github.com/karpathy/nanochat) tokenizer code —
          builds a production-quality BPE tokenizer from scratch, handling the
          edge cases we've skipped here (unicode, special tokens, regex pre-splitting).
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
