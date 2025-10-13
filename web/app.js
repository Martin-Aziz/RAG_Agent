const chatEl = document.getElementById('chat');
const form = document.getElementById('askForm');
const question = document.getElementById('question');
const modeSel = document.getElementById('mode');

function createMessageEl(who, text, meta) {
  const wrapper = document.createElement('div');
  wrapper.className = `message ${who}`;

  const avatar = document.createElement('div');
  avatar.className = `avatar ${who}`;
  avatar.textContent = who === 'user' ? 'U' : 'A';

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

function renderEvidence(evidence) {
  if (!evidence || !evidence.length) return;
  const list = evidence.map(e => `• ${e.text}`).join('\n');
  appendMessage('bot', list, 'Evidence');
}

function showTyping() {
  const el = appendMessage('bot', 'Thinking...');
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
    const data = await res.json();

    // replace typing
    const bubble = typingEl.querySelector('.bubble');
    if (data.answer) {
      bubble.textContent = data.answer;
      bubble.style.opacity = 1;
    } else {
      bubble.textContent = 'No answer';
      bubble.style.opacity = 1;
    }

    if (data.evidence && data.evidence.length) {
      renderEvidence(data.evidence);
    }
  } catch (err) {
    console.error(err);
    const bubble = typingEl.querySelector('.bubble');
    bubble.textContent = 'Error connecting to server';
    bubble.style.opacity = 1;
  }
});
