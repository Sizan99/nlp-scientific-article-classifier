document.addEventListener('DOMContentLoaded', () => {
    const chatWindow = document.getElementById('chat-window');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const clearBtn = document.getElementById('clear-btn');
    const uploadBtn = document.getElementById('upload-btn');
    const fileInput = document.getElementById('file-input');
    const charCount = document.getElementById('char-count');
    const pipelineMode = document.getElementById('pipeline-mode');
    
    // Character Counter
    userInput.addEventListener('input', () => {
        const count = userInput.value.length;
        charCount.textContent = `${count} / 2000 chars`;
        if (count > 2000) charCount.style.color = '#ef4444';
        else charCount.style.color = 'var(--text-muted)';
    });

    // Clear Chat
    clearBtn.addEventListener('click', () => {
        chatWindow.innerHTML = `
            <div class="message ai-message">
                <div class="avatar">
                    <svg viewBox="0 0 24 24" width="18" height="18" fill="white">
                        <path d="M12 2L14.4 9.6L22 12L14.4 14.4L12 22L9.6 14.4L2 12L9.6 9.6L12 2Z"/>
                    </svg>
                </div>
                <div class="bubble">
                    Chat history cleared. Please paste a scientific abstract for me to analyze.
                </div>
            </div>`;
    });

    // File Upload
    uploadBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;
        
        const reader = new FileReader();
        reader.onload = (e) => {
            userInput.value = e.target.result;
            // Trigger input event to update char counter
            userInput.dispatchEvent(new Event('input'));
        };
        reader.readAsText(file);
    });

    function appendMessage(sender, text, isHtml = false) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${sender}-message`;

        const avatarDiv = document.createElement('div');
        avatarDiv.className = 'avatar';
        
        const userSvg = `<svg viewBox="0 0 24 24" width="18" height="18" stroke="white" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>`;
        const aiSvg = `<svg viewBox="0 0 24 24" width="18" height="18" fill="white"><path d="M12 2L14.4 9.6L22 12L14.4 14.4L12 22L9.6 14.4L2 12L9.6 9.6L12 2Z"/></svg>`;
        
        avatarDiv.innerHTML = sender === 'user' ? userSvg : aiSvg;

        const bubbleDiv = document.createElement('div');
        bubbleDiv.className = 'bubble';
        if (isHtml) {
            bubbleDiv.innerHTML = text;
        } else {
            bubbleDiv.textContent = text;
        }

        msgDiv.appendChild(avatarDiv);
        msgDiv.appendChild(bubbleDiv);
        chatWindow.appendChild(msgDiv);
        
        // Scroll to bottom
        chatWindow.scrollTop = chatWindow.scrollHeight;
        return bubbleDiv; // Return bubble to update it later (for loading states)
    }

    async function sendMessage() {
        const text = userInput.value.trim();
        if (!text) return;

        // 1. Show user message
        appendMessage('user', text);
        userInput.value = '';

        // 2. Show AI loading message
        const aiBubble = appendMessage('ai', '', true);
        aiBubble.innerHTML = '<div class="loading"></div> Analyzing abstract...';

        try {
            // 3. Call FastAPI Backend
            const response = await fetch('/api/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    abstract: text,
                    pipeline_mode: pipelineMode.value
                })
            });

            if (!response.ok) throw new Error("Server Error");
            
            const data = await response.json();

            // 4. Update AI bubble with results
            const summaryLabel = pipelineMode.value === 'LLM' ? 'Generated Title (Abstractive):' : 'Extracted Summary (Extractive):';
            
            const resultHtml = `
                I have analyzed the abstract! Here are the results:
                <div class="results-box">
                    <p><span>Predicted Category:</span> ${data.category}</p>
                    <p style="margin-top: 8px;"><span>${summaryLabel}</span> ${data.title}</p>
                </div>
            `;
            aiBubble.innerHTML = resultHtml;

        } catch (error) {
            aiBubble.innerHTML = `⚠️ Sorry, an error occurred while processing the abstract. Please try again.`;
        }
    }

    sendBtn.addEventListener('click', sendMessage);

    // Allow Enter key to send (Shift+Enter for new line)
    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
});
