(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);

  const API = {
    transcribe: '/v1/transcribe',
    me: '/v1/me',
    logout: '/v1/auth/logout',
    history: '/v1/transcripts',
    transcript: (id) => `/v1/transcripts/${id}`,
  };

  const MAX_FILE_MB = 200;
  const ACCEPTED_EXT = /\.(m4a|mp3|wav|ogg|opus|flac|webm|mp4|mpga|mpeg)$/i;

  const els = {
    upload: $('upload'), progress: $('progress'), result: $('result'),
    error: $('error'), history: $('history'),
    dropzone: $('dropzone'), fileInput: $('file-input'),
    submitBtn: $('submit-btn'), cancelBtn: $('cancel-btn'),
    hint: $('hint'),
    langSelect: $('lang'),
    progressText: $('progress-text'), progressSub: $('progress-sub'),
    metaLang: $('meta-lang'), metaEngine: $('meta-engine'),
    metaDuration: $('meta-duration'), metaFallback: $('meta-fallback'),
    transcript: $('transcript'),
    copyBtn: $('copy-btn'), downloadBtn: $('download-btn'), resetBtn: $('reset-btn'),
    errorText: $('error-text'), errorReset: $('error-reset'),
    headerActions: $('header-actions'),
    userPill: $('user-pill'), userBtn: $('user-btn'), userMenu: $('user-menu'),
    userName: $('user-name'),
    historyBtn: $('history-btn'),
    historyFromUpload: $('history-from-upload'),
    historyFromResult: $('history-from-result'),
    logoutBtn: $('logout-btn'),
    historyClose: $('history-close'), historyList: $('history-list'),
    historyEmpty: $('history-empty'),
  };

  // The web UI always asks the server to use its default engine (Parakeet on
  // a normal install). Whisper stays available via the CLI / API for power use,
  // but the web flow keeps things simple: upload → transcript.
  const state = {
    file: null,
    abort: null,
    user: null,
    transcriptText: '',
    transcriptName: 'transcript',
  };

  // ----- Auth bootstrap -----
  async function bootstrap() {
    try {
      const r = await fetch(API.me, { credentials: 'same-origin' });
      if (r.status === 401) { window.location.replace('/login.html'); return; }
      if (!r.ok) throw new Error('auth check failed');
      const u = await r.json();
      state.user = u;
      els.userName.textContent = u.display_name || u.username;
      if (els.headerActions) els.headerActions.hidden = false;
    } catch (e) {
      // Don't redirect on network error — let user try, fail visibly
      console.warn('me check failed:', e);
    }
  }

  // (Engine selection UI removed — server picks the default engine.)

  // ----- File handling -----
  function setFile(file) {
    if (!file) {
      state.file = null;
      els.dropzone.classList.remove('has-file');
      els.submitBtn.disabled = true;
      els.hint.textContent = 'Kies een audiobestand om te beginnen.';
      els.hint.classList.remove('has-file');
      resetDropzoneCopy();
      return;
    }
    if (!ACCEPTED_EXT.test(file.name) && !(file.type || '').startsWith('audio/')) {
      showError(`Dit bestandstype ondersteunen we niet: ${file.name}`);
      return;
    }
    if (file.size > MAX_FILE_MB * 1024 * 1024) {
      showError(`Het bestand is te groot. Max ${MAX_FILE_MB} MB.`);
      return;
    }
    state.file = file;
    els.dropzone.classList.add('has-file');
    els.submitBtn.disabled = false;
    const sizeMb = (file.size / (1024 * 1024)).toFixed(1);
    els.hint.textContent = `Klaar: ${file.name} · ${sizeMb} MB`;
    els.hint.classList.add('has-file');

    const title = els.dropzone.querySelector('.dropzone-title');
    const sub = els.dropzone.querySelector('.dropzone-sub');
    title.textContent = file.name;
    sub.innerHTML = `<strong>${sizeMb} MB</strong> · klaar voor transcriptie`;
  }

  function resetDropzoneCopy() {
    const title = els.dropzone.querySelector('.dropzone-title');
    const sub = els.dropzone.querySelector('.dropzone-sub');
    title.textContent = 'Sleep een audiobestand hierheen';
    sub.innerHTML = 'of <strong>klik om te kiezen</strong>';
  }

  els.fileInput.addEventListener('change', (e) => {
    const f = e.target.files[0];
    if (f) setFile(f);
  });

  ['dragenter', 'dragover'].forEach((ev) => {
    els.dropzone.addEventListener(ev, (e) => {
      e.preventDefault(); e.stopPropagation();
      els.dropzone.classList.add('is-drag');
    });
  });
  ['dragleave', 'drop'].forEach((ev) => {
    els.dropzone.addEventListener(ev, (e) => {
      e.preventDefault(); e.stopPropagation();
      els.dropzone.classList.remove('is-drag');
    });
  });
  els.dropzone.addEventListener('drop', (e) => {
    const file = e.dataTransfer?.files?.[0];
    if (file) setFile(file);
  });

  // ----- Transcribe -----
  els.submitBtn.addEventListener('click', () => transcribe());
  els.cancelBtn?.addEventListener('click', () => state.abort?.abort());

  async function transcribe() {
    if (!state.file) return;
    showOnly(els.progress);
    els.progressText.textContent = 'Een moment, ik luister mee…';
    els.progressSub.textContent = 'Dit duurt meestal kort';

    const fd = new FormData();
    fd.append('audio', state.file);
    fd.append('engine', 'auto');
    fd.append('language', els.langSelect.value);

    state.abort = new AbortController();
    const tStart = Date.now();
    const tick = setInterval(() => {
      const sec = Math.round((Date.now() - tStart) / 1000);
      els.progressSub.textContent = `${formatTime(sec)} bezig`;
    }, 1000);

    try {
      const res = await fetch(API.transcribe, {
        method: 'POST',
        body: fd,
        credentials: 'same-origin',
        signal: state.abort.signal,
      });
      clearInterval(tick);
      if (res.status === 401) { window.location.replace('/login.html'); return; }
      if (!res.ok) {
        let msg = `Server gaf ${res.status} terug`;
        try {
          const j = await res.json();
          if (j?.error || j?.detail) msg = j.error || j.detail;
        } catch (_) {}
        throw new Error(msg);
      }
      const data = await res.json();
      showResult(data, state.file.name);
    } catch (err) {
      clearInterval(tick);
      if (err.name === 'AbortError') {
        showOnly(els.upload);
        return;
      }
      showError(err.message || 'Onbekende fout. Probeer het opnieuw.');
    } finally {
      state.abort = null;
    }
  }

  function engineLabel(e) {
    return ({ parakeet: 'Snel', whisper: 'Zorgvuldig', voxtral: 'Voxtral' })[e] || e;
  }
  function langLabel(c) {
    return ({ nl: 'Nederlands', en: 'English', fr: 'Français', de: 'Deutsch', auto: 'Automatisch' })[c] || c;
  }
  function formatTime(sec) {
    if (sec < 60) return `${sec}s`;
    const m = Math.floor(sec / 60), s = sec % 60;
    return `${m}m ${s}s`;
  }
  function fmtDate(iso) {
    try {
      const d = new Date(iso);
      return d.toLocaleDateString('nl-NL', { day: 'numeric', month: 'short', year: 'numeric' })
        + ' · ' + d.toLocaleTimeString('nl-NL', { hour: '2-digit', minute: '2-digit' });
    } catch { return iso; }
  }

  function showOnly(section) {
    [els.upload, els.progress, els.result, els.error, els.history].forEach((s) => {
      s.hidden = (s !== section);
    });
  }

  function showResult(data, fileName) {
    showOnly(els.result);
    const text = (data.text || '').trim();
    state.transcriptText = text;
    state.transcriptName = (fileName || 'transcript').replace(/\.[^.]+$/, '');

    els.transcript.innerHTML = '';
    const paragraphs = (data.segments && data.segments.length)
      ? groupSegments(data.segments)
      : splitSentences(text);
    paragraphs.forEach((p, i) => {
      const el = document.createElement('p');
      el.textContent = p;
      el.style.opacity = '0';
      el.style.transform = 'translateY(6px)';
      el.style.transition = 'opacity 400ms ease, transform 400ms ease';
      el.style.transitionDelay = `${i * 60}ms`;
      els.transcript.appendChild(el);
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          el.style.opacity = '1';
          el.style.transform = 'none';
        });
      });
    });

    els.metaEngine.textContent = engineLabel(data.engine);
    els.metaLang.textContent = langLabel(data.language || els.langSelect.value);
    if (data.duration_sec) {
      els.metaDuration.textContent = formatTime(Math.round(data.duration_sec));
      els.metaDuration.style.display = '';
    } else {
      els.metaDuration.style.display = 'none';
    }
    els.metaFallback.hidden = !data.fallback_used;

    els.transcript.dataset.text = text;
    els.result.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function groupSegments(segments) {
    const out = [];
    let buf = [];
    let lastEnd = 0;
    segments.forEach((s) => {
      const text = (s.text || '').trim();
      if (!text) return;
      const gap = (s.start ?? 0) - lastEnd;
      if (buf.length && (gap > 1.0 || buf.length >= 5)) {
        out.push(buf.join(' '));
        buf = [];
      }
      buf.push(text);
      lastEnd = s.end ?? lastEnd;
    });
    if (buf.length) out.push(buf.join(' '));
    return out;
  }

  function splitSentences(text) {
    if (!text) return [''];
    const parts = text.split(/(?<=[.!?…])\s+(?=[A-ZÀ-ÖØ-Þ"'(])/u);
    const out = [];
    for (let i = 0; i < parts.length; i += 3) {
      out.push(parts.slice(i, i + 3).join(' ').trim());
    }
    return out.length ? out : [text];
  }

  function showError(msg) {
    showOnly(els.error);
    els.errorText.textContent = msg;
  }

  function reset() {
    state.file = null;
    els.fileInput.value = '';
    els.dropzone.classList.remove('has-file');
    els.submitBtn.disabled = true;
    els.hint.textContent = 'Kies een audiobestand om te beginnen.';
    els.hint.classList.remove('has-file');
    resetDropzoneCopy();
    els.transcript.innerHTML = '';
    showOnly(els.upload);
    els.upload.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  els.resetBtn.addEventListener('click', reset);
  els.errorReset.addEventListener('click', reset);

  // ----- Copy / Download -----
  els.copyBtn.addEventListener('click', async () => {
    const text = state.transcriptText || els.transcript.innerText;
    const restore = els.copyBtn.innerHTML;
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      const ta = document.createElement('textarea');
      ta.value = text; document.body.appendChild(ta); ta.select();
      try { document.execCommand('copy'); } catch (_) {}
      ta.remove();
    }
    els.copyBtn.classList.add('copied');
    els.copyBtn.innerHTML =
      '<svg viewBox="0 0 20 20" width="18" height="18" aria-hidden="true">' +
      '<path d="M4 10l4 4 8-8" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>' +
      '</svg><span>Gekopieerd</span>';
    setTimeout(() => {
      els.copyBtn.classList.remove('copied');
      els.copyBtn.innerHTML = restore;
    }, 1800);
  });

  els.downloadBtn.addEventListener('click', () => {
    const text = state.transcriptText || els.transcript.innerText;
    const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = (state.transcriptName || 'transcript') + '.txt';
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  });

  // ----- User menu -----
  els.userBtn?.addEventListener('click', (e) => {
    e.stopPropagation();
    const open = !els.userMenu.hidden;
    els.userMenu.hidden = open;
    els.userBtn.setAttribute('aria-expanded', open ? 'false' : 'true');
  });
  document.addEventListener('click', (e) => {
    if (!els.userPill?.contains(e.target)) {
      if (els.userMenu) {
        els.userMenu.hidden = true;
        els.userBtn?.setAttribute('aria-expanded', 'false');
      }
    }
  });

  els.logoutBtn?.addEventListener('click', async () => {
    try {
      await fetch(API.logout, { method: 'POST', credentials: 'same-origin' });
    } catch (_) {}
    window.location.replace('/login.html');
  });

  // ----- History -----
  els.historyBtn?.addEventListener('click', () => {
    if (els.userMenu) {
      els.userMenu.hidden = true;
      els.userBtn?.setAttribute('aria-expanded', 'false');
    }
    openHistory();
  });
  els.historyFromUpload?.addEventListener('click', openHistory);
  els.historyFromResult?.addEventListener('click', openHistory);
  els.historyClose?.addEventListener('click', () => showOnly(els.upload));

  async function openHistory() {
    showOnly(els.history);
    els.historyList.innerHTML = '<p class="history-empty">Even laden…</p>';
    try {
      const r = await fetch(API.history, { credentials: 'same-origin' });
      if (r.status === 401) { window.location.replace('/login.html'); return; }
      if (!r.ok) throw new Error('kon geschiedenis niet laden');
      const data = await r.json();
      renderHistory(data.items || []);
    } catch (e) {
      els.historyList.innerHTML = '';
      const p = document.createElement('p');
      p.className = 'history-empty';
      p.textContent = 'Kon geschiedenis niet laden.';
      els.historyList.appendChild(p);
    }
  }

  function renderHistory(items) {
    els.historyList.innerHTML = '';
    if (!items.length) {
      const p = document.createElement('p');
      p.className = 'history-empty';
      p.textContent = 'Nog niets om te tonen.';
      els.historyList.appendChild(p);
      return;
    }
    items.forEach((it) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'history-item';
      const meta = [
        fmtDate(it.created_at),
        langLabel(it.language || 'auto'),
        engineLabel(it.engine),
      ];
      if (it.duration_sec) meta.push(formatTime(Math.round(it.duration_sec)));
      const snippetHtml = it.snippet
        ? `<p class="history-item-snippet">${escapeHtml(it.snippet)}${(it.snippet || '').length >= 160 ? '…' : ''}</p>`
        : '';
      btn.innerHTML = `
        <div class="history-item-main">
          <p class="history-item-title">${escapeHtml(it.original_filename || 'Naamloos')}</p>
          <p class="history-item-meta">${meta.join(' · ')}</p>
          ${snippetHtml}
        </div>
        <svg class="history-item-arrow" viewBox="0 0 20 20" width="18" height="18" aria-hidden="true">
          <path d="M7 5l5 5-5 5" fill="none" stroke="currentColor" stroke-width="1.8"
                stroke-linecap="round" stroke-linejoin="round"/>
        </svg>`;
      btn.addEventListener('click', () => loadTranscript(it.id));
      els.historyList.appendChild(btn);
    });
  }

  async function loadTranscript(id) {
    try {
      const r = await fetch(API.transcript(id), { credentials: 'same-origin' });
      if (r.status === 401) { window.location.replace('/login.html'); return; }
      if (!r.ok) throw new Error('kon transcript niet laden');
      const data = await r.json();
      showResult(data, data.original_filename);
    } catch (e) {
      showError(e.message || 'Kon dit transcript niet laden.');
    }
  }

  function escapeHtml(s) {
    return String(s ?? '').replace(/[&<>"']/g, (c) =>
      ({ '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;' }[c]));
  }

  // ----- Keyboard: Enter submits when ready -----
  window.addEventListener('keydown', (e) => {
    if (e.key !== 'Enter' || e.shiftKey || e.metaKey || e.ctrlKey) return;
    if (els.submitBtn.disabled || els.upload.hidden) return;
    const tag = document.activeElement?.tagName;
    if (tag === 'BUTTON' || tag === 'SELECT' || tag === 'INPUT' || tag === 'TEXTAREA') return;
    els.submitBtn.click();
  });

  bootstrap();
})();
