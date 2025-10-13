const chatEl = document.getElementById('chat');
const form = document.getElementById('askForm');
const question = document.getElementById('question');
const modeSel = document.getElementById('mode');

function appendMessage(who, text) {
  const div = document.createElement('div');
  div.className = `message ${who}`;
  const b = document.createElement('div');
  b.className = 'bubble';
  b.textContent = text;
  div.appendChild(b);
  chatEl.appendChild(div);
  chatEl.scrollTop = chatEl.scrollHeight;
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const q = question.value.trim();
  if(!q) return;
  appendMessage('user', q);
  question.value = '';
  appendMessage('bot', '...thinking');
  try {
    // include required fields expected by QueryRequest pydantic model
    const payload = {
      user_id: 'web_user',
      session_id: String(Date.now()),
      query: q,
      mode: modeSel.value,
      context_ids: [],
      prefer_low_cost: true
    };
    const res = await fetch('/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    // replace last bot 'thinking' with actual
    const lastBot = chatEl.querySelector('.message.bot:last-child .bubble');
    if (data.answer) {
      lastBot.textContent = data.answer;
    } else {
      lastBot.textContent = 'No answer';
    }
    if (data.evidence && data.evidence.length) {
      const ev = data.evidence.map(e => `- ${e.text}`).join('\n');
      appendMessage('bot', ev);
    }
  } catch (err) {
    console.error(err);
    appendMessage('bot', 'Error connecting to server');
  }
});
