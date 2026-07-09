package com.example.minicpm_v_demo

import android.content.Intent
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.ImageButton
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.updatePadding
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.appbar.AppBarLayout
import com.google.android.material.textfield.TextInputEditText
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class SemanticActivity : AppCompatActivity() {

    companion object {
        private val TAG = SemanticActivity::class.java.simpleName
    }

    private lateinit var recyclerChat: RecyclerView
    private lateinit var chatAdapter: ChatAdapter
    private lateinit var etInput: TextInputEditText
    private lateinit var btnSend: ImageButton
    private lateinit var btnClearChat: ImageButton
    private lateinit var btnModelManager: ImageButton
    private lateinit var cardInputBar: View
    private lateinit var appBarLayout: AppBarLayout
    private lateinit var tvStatus: TextView

    private var classifier: SemanticClassifier? = null
    private var messageIdCounter = 1L
    private val messages = mutableListOf<ChatMessage>()
    private var createdWithLocale: String? = null
    private var isLocaleRestart = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        createdWithLocale = LocaleManager.currentLanguage(this).tag

        setContentView(R.layout.activity_semantic)

        WindowCompat.setDecorFitsSystemWindows(window, false)
        val rootContent = findViewById<View>(android.R.id.content)
        ViewCompat.setOnApplyWindowInsetsListener(rootContent) { v, insets ->
            val sysBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            val ime = insets.getInsets(WindowInsetsCompat.Type.ime())
            v.updatePadding(
                left = sysBars.left,
                top = sysBars.top,
                right = sysBars.right,
                bottom = maxOf(sysBars.bottom, ime.bottom)
            )
            insets
        }

        initViews()
        setupRecyclerView()
        setupClickListeners()
        loadClassifier()
    }

    private fun initViews() {
        recyclerChat = findViewById(R.id.recycler_chat)
        etInput = findViewById(R.id.et_input)
        btnSend = findViewById(R.id.btn_send)
        btnClearChat = findViewById(R.id.btn_clear_chat)
        btnModelManager = findViewById(R.id.btn_model_manager)
        cardInputBar = findViewById(R.id.card_input_bar)
        appBarLayout = findViewById(R.id.appBarLayout)
        tvStatus = findViewById(R.id.tv_status)
    }

    private fun setupRecyclerView() {
        chatAdapter = ChatAdapter(io.noties.markwon.Markwon.create(this))
        recyclerChat.layoutManager = LinearLayoutManager(this)
        recyclerChat.adapter = chatAdapter

        cardInputBar.viewTreeObserver.addOnGlobalLayoutListener {
            recyclerChat.setPadding(
                recyclerChat.paddingLeft,
                recyclerChat.paddingTop,
                recyclerChat.paddingRight,
                cardInputBar.height
            )
        }

        messages.add(ChatMessage.WelcomeCard(isTextOnly = true))
        chatAdapter.submitList(messages.toList())
    }

    private fun setupClickListeners() {
        btnSend.setOnClickListener { handleInput() }
        btnClearChat.setOnClickListener { showClearDialog() }
        btnModelManager.setOnClickListener {
            startActivity(Intent(this, ModelManagerActivity::class.java))
        }
        etInput.setOnFocusChangeListener { _, hasFocus ->
            if (hasFocus) {
                appBarLayout.setExpanded(false, true)
                scrollToBottom()
            }
        }
    }

    private fun loadClassifier() {
        tvStatus.text = getString(R.string.semantic_loading)
        tvStatus.visibility = View.VISIBLE
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val clf = SemanticClassifier.create(applicationContext)
                classifier = clf
                withContext(Dispatchers.Main) {
                    tvStatus.visibility = View.GONE
                    Log.i(TAG, "SemanticClassifier ready")
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load SemanticClassifier", e)
                withContext(Dispatchers.Main) {
                    tvStatus.text = getString(R.string.semantic_load_failed, e.message)
                    tvStatus.visibility = View.VISIBLE
                }
            }
        }
    }

    private fun handleInput() {
        val text = etInput.text?.toString()?.trim().orEmpty()
        if (text.isEmpty()) {
            Toast.makeText(this, R.string.toast_empty_input, Toast.LENGTH_SHORT).show()
            return
        }
        if (classifier == null) {
            Toast.makeText(this, R.string.semantic_loading, Toast.LENGTH_SHORT).show()
            return
        }

        val userMsg = ChatMessage.UserMessage(id = messageIdCounter++, text = text)
        messages.add(userMsg)
        chatAdapter.submitList(messages.toList())
        etInput.text?.clear()
        scrollToBottom()

        val aiId = messageIdCounter++
        val aiMsg = ChatMessage.AiMessage(id = aiId, text = "", isGenerating = true)
        messages.add(aiMsg)
        chatAdapter.submitList(messages.toList())
        chatAdapter.setActiveAiMessage(aiId)
        scrollToBottom()

        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val result = classifier!!.classify(text)
                val response = buildString {
                    appendLine("**${result.label}**")
                    appendLine()
                    appendLine(getString(R.string.semantic_confidence, result.confidence))
                }
                withContext(Dispatchers.Main) {
                    val idx = messages.indexOfFirst { it.id == aiId }
                    if (idx >= 0) {
                        messages[idx] = (messages[idx] as ChatMessage.AiMessage).copy(
                            text = response,
                            isGenerating = false
                        )
                        chatAdapter.submitList(messages.toList())
                        chatAdapter.clearActiveAiMessage()
                        scrollToBottom()
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Classification failed", e)
                withContext(Dispatchers.Main) {
                    val idx = messages.indexOfFirst { it.id == aiId }
                    if (idx >= 0) {
                        messages[idx] = (messages[idx] as ChatMessage.AiMessage).copy(
                            text = getString(R.string.semantic_error, e.message),
                            isGenerating = false
                        )
                        chatAdapter.submitList(messages.toList())
                        chatAdapter.clearActiveAiMessage()
                    }
                }
            }
        }
    }

    private fun showClearDialog() {
        AlertDialog.Builder(this)
            .setTitle(R.string.clear_chat)
            .setMessage(R.string.clear_chat_confirm)
            .setPositiveButton(R.string.confirm) { _, _ -> clearChat() }
            .setNegativeButton(R.string.cancel, null)
            .show()
    }

    private fun clearChat() {
        messages.clear()
        messages.add(ChatMessage.WelcomeCard(isTextOnly = true))
        messageIdCounter = 1L
        chatAdapter.submitList(messages.toList())
        Toast.makeText(this, R.string.clear_chat_toast, Toast.LENGTH_SHORT).show()
    }

    private fun scrollToBottom() {
        recyclerChat.post {
            val count = chatAdapter.itemCount
            if (count > 0) {
                recyclerChat.scrollToPosition(count - 1)
            }
        }
    }

    override fun onResume() {
        super.onResume()
        val currentTag = LocaleManager.currentLanguage(this).tag
        if (createdWithLocale != null && createdWithLocale != currentTag) {
            isLocaleRestart = true
            LocaleManager.recreateSeamlessly(this)
            return
        }
        // If the user switched away from the semantic classifier while
        // this activity was in the background, redirect to MainActivity.
        val model = LlamaEngine.getSelectedModel(applicationContext)
        if (!model.isSemanticClassifier) {
            startActivity(Intent(this, MainActivity::class.java))
            finish()
            return
        }
    }

    override fun onDestroy() {
        if (isFinishing && !isLocaleRestart) {
            classifier?.close()
            classifier = null
        }
        super.onDestroy()
    }
}
