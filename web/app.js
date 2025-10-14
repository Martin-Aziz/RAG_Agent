const chatEl = document.getElementById('chat');
const form = document.getElementById('askForm');
const question = document.getElementById('question');
const modeSel = document.getElementById('mode');
const statusBar = document.getElementById('status');

function createMessageEl(who, text, meta) {
  const wrapper = document.createElement('div');
  wrapper.className = `message ${who}`;

  const avatar = document.createElement('div');
  avatar.className = `avatar ${who}`;
  avatar.setAttribute('aria-hidden', 'true');
  avatar.dataset.initial = who === 'user' ? 'U' : 'A';

  const body = document.createElement('div');
  body.className = 'body';

  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;

  body.appendChild(bubble);
  if (meta) {
    const m = document.createElement('div');
    m.className = 'meta';
    m.textContent = meta;
    body.appendChild(m);
  }

  if (who === 'user') {
    wrapper.appendChild(body);
    wrapper.appendChild(avatar);
  } else {
    wrapper.appendChild(avatar);
    wrapper.appendChild(body);
  }

  return wrapper;
}

function appendMessage(who, text, meta) {
  const el = createMessageEl(who, text, meta);
  chatEl.appendChild(el);
  chatEl.scrollTop = chatEl.scrollHeight;
  return el;
}

function parseStructuredResponse(answerText) {
  try {
    const parsed = JSON.parse(answerText);
    return {
      summary: parsed.summary || answerText,
      evidence: parsed.evidence || []
    };
  } catch (e) {
    // Fallback: treat the entire response as summary
    return {
      summary: answerText,
      evidence: []
    };
  }
}

function renderEvidence(evidence) {
  if (!evidence || !evidence.length) return null;
  
  const container = document.createElement('div');
  container.className = 'evidence-container';
  
  const title = document.createElement('div');
  title.style.fontWeight = '600';
  title.style.marginBottom = '8px';
  title.style.fontSize = '13px';
  title.style.color = '#fff';
  title.textContent = '📚 Supporting Evidence';
  container.appendChild(title);
  
  evidence.forEach(item => {
    const evidenceItem = document.createElement('div');
    evidenceItem.className = 'evidence-item';
    
    const docName = document.createElement('div');
    docName.className = 'document';
    docName.textContent = item.document || item.doc_id || 'Document';
    evidenceItem.appendChild(docName);
    
    const passage = document.createElement('div');
    passage.className = 'passage';
    const passageText = item.passage || item.text || '';
    passage.textContent = passageText.length > 200 
      ? passageText.substring(0, 197) + '...' 
      : passageText;
    evidenceItem.appendChild(passage);
    
    container.appendChild(evidenceItem);
  });
  
  return container;
}

function setStatus(message, show = true) {
  if (!statusBar) return;
  statusBar.textContent = message;
  statusBar.style.display = show ? 'block' : 'none';
}

function showTyping() {
  setStatus('🔍 Searching documents...');
  const el = appendMessage('bot', '...');
  el.dataset.typing = '1';
  const bubble = el.querySelector('.bubble');
  bubble.style.opacity = 0.6;
  return el;
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const q = question.value.trim();
  if (!q) return;
  
  appendMessage('user', q);
  question.value = '';

  const typingEl = showTyping();

  try {
    setStatus('🧠 Generating answer...');
    
    const payload = {
      user_id: 'web_user',
      session_id: String(Date.now()),
      query: q,
      mode: modeSel ? modeSel.value : 'parrag',
      context_ids: [],
      prefer_low_cost: true
    };
    
    const res = await fetch('/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    
    if (!res.ok) {
      throw new Error(`Server error: ${res.status}`);
    }
    
    const data = await res.json();
    setStatus('', false);

    // Parse structured response
    const structured = parseStructuredResponse(data.answer);
    
    // Update the typing bubble with the summary
    const bubble = typingEl.querySelector('.bubble');
    bubble.textContent = structured.summary || 'No answer available.';
    bubble.style.opacity = 1;
    
    // Render evidence from structured response if available
    let evidenceToRender = structured.evidence;
    
    // Fallback to data.evidence if structured response doesn't have evidence
    if ((!evidenceToRender || evidenceToRender.length === 0) && data.evidence && data.evidence.length > 0) {
      evidenceToRender = data.evidence;
    }
    
    if (evidenceToRender && evidenceToRender.length > 0) {
      const evidenceEl = renderEvidence(evidenceToRender);
      if (evidenceEl) {
        typingEl.querySelector('.body').appendChild(evidenceEl);
      }
    }
    
  } catch (err) {
    console.error('Error:', err);
    setStatus('', false);
    const bubble = typingEl.querySelector('.bubble');
    bubble.textContent = 'Error connecting to server. Please try again.';
    bubble.style.opacity = 1;
  }
});
