package com.example.minicpm_v_demo

import android.content.Context

/**
 * BERT WordPiece tokenizer for Chinese models (e.g. chinese-macbert-base).
 *
 * Loads vocab.txt from assets and produces input_ids / attention_mask / token_type_ids
 * triplets suitable for ONNX Runtime inference.
 */
class BertTokenizer(
    private val vocab: Map<String, Int>,
    private val maxLength: Int = 64,
) {
    companion object {
        private const val TAG = "BertTokenizer"
        const val CLS_TOKEN = "[CLS]"
        const val SEP_TOKEN = "[SEP]"
        const val PAD_TOKEN = "[PAD]"
        const val UNK_TOKEN = "[UNK]"

        fun loadVocab(context: Context): Map<String, Int> {
            val vocab = HashMap<String, Int>()
            context.assets.open("vocab.txt").bufferedReader().useLines { lines ->
                lines.forEachIndexed { i, line -> vocab[line.trim()] = i }
            }
            return vocab
        }
    }

    private val clsId: Int = vocab[CLS_TOKEN] ?: 101
    private val sepId: Int = vocab[SEP_TOKEN] ?: 102
    private val padId: Int = vocab[PAD_TOKEN] ?: 0
    private val unkId: Int = vocab[UNK_TOKEN] ?: 100

    data class Encoding(
        val inputIds: LongArray,
        val attentionMask: LongArray,
        val tokenTypeIds: LongArray,
    )

    fun encode(text: String): Encoding {
        val tokens = tokenize(text)

        // Truncate to maxLength (accounting for [CLS] and [SEP])
        val maxTokens = maxLength
        val truncated = if (tokens.size > maxTokens) tokens.subList(0, maxTokens) else tokens

        val inputIds = LongArray(maxLength)
        val attentionMask = LongArray(maxLength)
        val tokenTypeIds = LongArray(maxLength)

        for (i in truncated.indices) {
            inputIds[i] = vocab[truncated[i]]?.toLong() ?: unkId.toLong()
            attentionMask[i] = 1L
        }

        return Encoding(inputIds, attentionMask, tokenTypeIds)
    }

    private fun tokenize(text: String): List<String> {
        val tokens = mutableListOf<String>()
        tokens.add(CLS_TOKEN)

        // Basic tokenization: clean + split on whitespace, then per-segment WordPiece
        val cleaned = cleanText(text)
        val segments = cleaned.split(Regex("\\s+")).filter { it.isNotEmpty() }

        for (segment in segments) {
            // Split segment into individual Chinese chars and non-Chinese runs
            val subSegments = splitByCjk(segment)
            for (sub in subSegments) {
                if (isCjkChar(sub.first())) {
                    // Each CJK character is its own token
                    for (ch in sub) {
                        tokens.add(ch.toString())
                    }
                } else {
                    // WordPiece on the non-CJK run
                    tokens.addAll(wordPieceTokenize(sub))
                }
            }
        }

        tokens.add(SEP_TOKEN)
        return tokens
    }

    private fun cleanText(text: String): String {
        val sb = StringBuilder()
        for (ch in text) {
            val cp = ch.code
            when {
                cp == 0 || cp == 0xFFFD -> { /* skip */ }
                isControlChar(ch) -> { /* skip */ }
                isWhitespace(ch) -> sb.append(' ')
                else -> sb.append(ch)
            }
        }
        return sb.toString()
    }

    private fun splitByCjk(text: String): List<String> {
        val result = mutableListOf<String>()
        val current = StringBuilder()
        var currentIsCjk = false

        for (ch in text) {
            val cjk = isCjkChar(ch)
            if (current.isNotEmpty() && cjk != currentIsCjk) {
                result.add(current.toString())
                current.clear()
            }
            current.append(ch)
            currentIsCjk = cjk
        }
        if (current.isNotEmpty()) result.add(current.toString())
        return result
    }

    private fun wordPieceTokenize(text: String): List<String> {
        val tokens = mutableListOf<String>()
        val chars = text.toCharArray()
        var start = 0

        while (start < chars.size) {
            var end = chars.size
            var curToken: String? = null

            while (start < end) {
                val sub = if (start == 0) {
                    String(chars, start, end - start)
                } else {
                    "##" + String(chars, start, end - start)
                }
                if (vocab.containsKey(sub)) {
                    curToken = sub
                    break
                }
                end--
            }

            if (curToken == null) {
                tokens.add(UNK_TOKEN)
                start++
            } else {
                tokens.add(curToken)
                start = end
            }
        }
        return tokens
    }

    private fun isCjkChar(ch: Char): Boolean {
        val cp = ch.code
        return (cp in 0x4E00..0x9FFF) ||
            (cp in 0x3400..0x4DBF) ||
            (cp in 0x20000..0x2A6DF) ||
            (cp in 0x2A700..0x2B73F) ||
            (cp in 0x2B740..0x2B81F) ||
            (cp in 0x2B820..0x2CEAF) ||
            (cp in 0xF900..0xFAFF) ||
            (cp in 0x2F800..0x2FA1F)
    }

    private fun isWhitespace(ch: Char): Boolean {
        return ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r' ||
            ch.code == 0x00A0 || ch.code == 0x1680 ||
            (ch.code in 0x2000..0x200A) ||
            ch.code == 0x202F || ch.code == 0x205F || ch.code == 0x3000
    }

    private fun isControlChar(ch: Char): Boolean {
        if (ch == '\t' || ch == '\n' || ch == '\r') return false
        return ch.code in 0x00..0x1F || ch.code in 0x7F..0x9F
    }
}
