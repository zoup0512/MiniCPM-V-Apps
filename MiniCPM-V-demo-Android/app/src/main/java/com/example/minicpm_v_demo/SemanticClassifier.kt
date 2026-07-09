package com.example.minicpm_v_demo

import android.content.Context
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.json.JSONObject
import java.io.File
import kotlin.math.exp

/**
 * Semantic intent classifier backed by an ONNX BERT model.
 *
 * Loads bert.onnx, vocab.txt, and id2label.json from assets.
 * Call [classify] to get a predicted label + confidence for a text input.
 */
class SemanticClassifier private constructor(
    private val session: OrtSession,
    private val tokenizer: BertTokenizer,
    private val id2label: Map<Int, String>,
) {
    companion object {
        private const val TAG = "SemanticClassifier"
        private const val MAX_LENGTH = 64

        fun create(context: Context): SemanticClassifier {
            val env = OrtEnvironment.getEnvironment()

            // Copy bert.onnx from assets to a temp file so ONNX Runtime can
            // mmap it instead of loading the entire ~400MB into JVM heap.
            val modelFile = File(context.cacheDir, "bert.onnx")
            if (!modelFile.exists()) {
                context.assets.open("bert.onnx").use { input ->
                    modelFile.outputStream().use { output ->
                        input.copyTo(output)
                    }
                }
            }
            val session = env.createSession(modelFile.absolutePath)

            val vocab = BertTokenizer.loadVocab(context)
            val tokenizer = BertTokenizer(vocab, MAX_LENGTH)

            val id2label = mutableMapOf<Int, String>()
            val json = context.assets.open("id2label.json").bufferedReader().use { it.readText() }
            val obj = JSONObject(json)
            for (key in obj.keys()) {
                id2label[key.toInt()] = obj.getString(key)
            }

            Log.i(TAG, "Loaded model: ${id2label.size} labels, vocab=${vocab.size}")
            return SemanticClassifier(session, tokenizer, id2label)
        }
    }

    data class Result(
        val label: String,
        val labelId: Int,
        val confidence: Float,
    )

    fun classify(text: String): Result {
        val env = OrtEnvironment.getEnvironment()
        val encoding = tokenizer.encode(text)

        val inputIdsTensor = OnnxTensor.createTensor(env, arrayOf(encoding.inputIds))
        val attentionMaskTensor = OnnxTensor.createTensor(env, arrayOf(encoding.attentionMask))
        val tokenTypeIdsTensor = OnnxTensor.createTensor(env, arrayOf(encoding.tokenTypeIds))

        val inputs = mapOf(
            "input_ids" to inputIdsTensor,
            "attention_mask" to attentionMaskTensor,
            "token_type_ids" to tokenTypeIdsTensor,
        )

        val output = session.run(inputs)
        @Suppress("UNCHECKED_CAST")
        val logits = (output[0].value as Array<FloatArray>)[0]

        inputIdsTensor.close()
        attentionMaskTensor.close()
        tokenTypeIdsTensor.close()
        output.close()

        // softmax
        val maxVal = logits.maxOrNull() ?: 0f
        var sumExp = 0.0
        val expVals = FloatArray(logits.size) { i ->
            val e = exp((logits[i] - maxVal).toDouble()).toFloat()
            sumExp += e
            e
        }
        val probs = FloatArray(logits.size) { i -> (expVals[i] / sumExp).toFloat() }

        var predId = 0
        var maxProb = probs[0]
        for (i in 1 until probs.size) {
            if (probs[i] > maxProb) {
                maxProb = probs[i]
                predId = i
            }
        }

        val label = id2label[predId] ?: "unknown"
        return Result(label = label, labelId = predId, confidence = maxProb)
    }

    fun close() {
        session.close()
    }
}
