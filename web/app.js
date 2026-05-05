(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);

  const API = {
    transcribe: '/v1/transcribe',
    jobs: '/v1/jobs',
    job: (id) => `/v1/jobs/${id}`,
    me: '/v1/me',
    password: '/v1/me/password',
    adminUsers: '/v1/admin/users',
    adminUser: (id) => `/v1/admin/users/${id}`,
    logout: '/v1/auth/logout',
    history: '/v1/transcripts',
    transcript: (id) => `/v1/transcripts/${id}`,
  };

  const MAX_FILE_MB = 200;
  const ACCEPTED_EXT = /\.(m4a|mp3|wav|ogg|opus|flac|webm|mp4|mpga|mpeg)$/i;
  const ACTIVE_STATUSES = new Set(['queued', 'running']);
  const TERMINAL_STATUSES = new Set(['done', 'failed', 'cancelled']);

  const els = {
    upload: $('upload'), result: $('result'), error: $('error'), history: $('history'), settings: $('settings'),
    activeJobs: $('active-jobs'), activeJobsList: $('active-jobs-list'), activeJobsCount: $('active-jobs-count'),
    dropzone: $('dropzone'), fileInput: $('file-input'), pasteAudioBtn: $('paste-audio-btn'),
    submitBtn: $('submit-btn'), hint: $('hint'), engineHelp: $('engine-help'), langSelect: $('lang'),
    metaLang: $('meta-lang'), metaEngine: $('meta-engine'), metaDuration: $('meta-duration'), metaFallback: $('meta-fallback'),
    transcript: $('transcript'), copyBtn: $('copy-btn'), downloadBtn: $('download-btn'), resetBtn: $('reset-btn'),
    errorText: $('error-text'), errorReset: $('error-reset'), headerActions: $('header-actions'),
    userPill: $('user-pill'), userBtn: $('user-btn'), userMenu: $('user-menu'), userName: $('user-name'),
    historyBtn: $('history-btn'), historyFromUpload: $('history-from-upload'), historyFromResult: $('history-from-result'),
    logoutBtn: $('logout-btn'), settingsBtn: $('settings-btn'), settingsClose: $('settings-close'),
    historyClose: $('history-close'), historyList: $('history-list'), historyEmpty: $('history-empty'),
    passwordForm: $('password-form'), currentPassword: $('current-password'), newPassword: $('new-password'),
    confirmPassword: $('confirm-password'), passwordMessage: $('password-message'), usersCard: $('users-card'),
    usersList: $('users-list'), refreshUsers: $('refresh-users'), newUserForm: $('new-user-form'),
    newUserUsername: $('new-user-username'), newUserDisplay: $('new-user-display'),
    newUserPassword: $('new-user-password'), newUserAdmin: $('new-user-admin'), usersMessage: $('users-message'),
  };

  const state = {
    file: null,
    engine: 'whisper',
    user: null,
    transcriptText: '',
    transcriptName: 'transcript',
    jobs: new Map(),
    pollingTimer: null,
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
      if (els.usersCard) els.usersCard.hidden = !u.is_admin;
      await refreshJobs({ once: true });
    } catch (e) {
      console.warn('me check failed:', e);
    }
  }

  // ----- Engine selection -----
  document.querySelectorAll('.seg').forEach((seg) => {
    seg.addEventListener('click', () => {
      if (seg.disabled) return;
      document.querySelectorAll('.seg').forEach((s) => {
        const active = s === seg;
        s.classList.toggle('active', active);
        s.setAttribute('aria-checked', active ? 'true' : 'false');
      });
      state.engine = seg.dataset.engine;
      if (els.engineHelp) els.engineHelp.textContent = engineHelpText(state.engine);
    });
  });

  function engineHelpText(e) {
    if (e === 'parakeet') return 'Parakeet · razendsnel, goed multilingual';
    if (e === 'whisper')  return 'Whisper · trager, sterk in Nederlands';
    if (e === 'voxtral')  return 'Voxtral · audio-LLM, experimenteel';
    return '';
  }

  function setNoEngineState(reason) {
    state.file = null;
    if (els.fileInput) els.fileInput.value = '';
    if (els.dropzone) {
      els.dropzone.classList.remove('has-file');
      els.dropzone.style.pointerEvents = 'none';
      els.dropzone.style.opacity = '0.55';
    }
    if (els.submitBtn) els.submitBtn.disabled = true;
    if (els.hint) {
      els.hint.textContent = reason;
      els.hint.classList.remove('has-file');
      els.hint.style.color = 'var(--accent-deep)';
    }
  }

  function clearNoEngineState() {
    if (els.dropzone) {
      els.dropzone.style.pointerEvents = '';
      els.dropzone.style.opacity = '';
    }
    if (els.hint) {
      els.hint.style.color = '';
      if (!state.file) els.hint.textContent = 'Kies een audiobestand om te beginnen.';
    }
  }

  async function refreshEngineAvailability() {
    let ready = null;
    try {
      const r = await fetch('/v1/health', { credentials: 'same-origin' });
      if (r.ok) ready = (await r.json())?.engines || {};
    } catch (_) {}

    if (!ready) {
      setNoEngineState('Server niet bereikbaar — probeer het later opnieuw.');
      return;
    }

    const segs = Array.from(document.querySelectorAll('.seg'));
    let anyReady = false;
    segs.forEach((seg) => {
      const name = seg.dataset.engine;
      const ok = !!ready?.[name]?.ready;
      if (ok) anyReady = true;
      seg.disabled = !ok;
      seg.title = ok ? '' : 'Niet beschikbaar — bouw image opnieuw met dit engine erbij';
      seg.style.opacity = ok ? '' : '0.45';
    });

    if (!anyReady) {
      setNoEngineState('Geen transcribeer-engine beschikbaar. Neem contact op met de beheerder.');
      return;
    }
    clearNoEngineState();

    const activeBtn = document.querySelector('.seg.active');
    if (activeBtn && activeBtn.disabled) {
      const fallback = segs.find((s) => !s.disabled);
      if (fallback) fallback.click();
    }
  }

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

  els.pasteAudioBtn?.addEventListener('click', pasteFromClipboard);
  window.addEventListener('paste', (e) => {
    const tag = document.activeElement?.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA' || document.activeElement?.isContentEditable) return;
    const file = fileFromDataTransferItems(e.clipboardData?.items);
    if (file) {
      e.preventDefault();
      setFile(file);
      toast('Audio uit klembord klaargezet.');
    }
  });

  async function pasteFromClipboard() {
    try {
      if (!navigator.clipboard?.read) {
        toast('Klembord lezen wordt niet ondersteund in deze browser.');
        return;
      }
      const items = await navigator.clipboard.read();
      for (const item of items) {
        const type = item.types.find((t) => t.startsWith('audio/'));
        if (!type) continue;
        const blob = await item.getType(type);
        setFile(new File([blob], clipboardFileName(type), { type }));
        toast('Audio uit klembord klaargezet.');
        return;
      }
      toast('Geen audio in klembord gevonden');
    } catch (e) {
      toast('Geen audio in klembord gevonden');
    }
  }

  function fileFromDataTransferItems(items) {
    for (const item of Array.from(items || [])) {
      if (item.kind === 'file' && item.type.startsWith('audio/')) {
        const file = item.getAsFile();
        if (!file) continue;
        return new File([file], file.name || clipboardFileName(item.type), { type: item.type || file.type });
      }
    }
    return null;
  }

  function clipboardFileName(type) {
    const ext = ({ 'audio/mpeg': 'mp3', 'audio/mp4': 'm4a', 'audio/wav': 'wav', 'audio/webm': 'webm', 'audio/ogg': 'ogg' })[type] || (type.split('/')[1] || 'audio');
    return `klembord-${new Date().toISOString().replace(/[:.]/g, '-')}.${ext}`;
  }

  // ----- Transcribe enqueue -----
  els.submitBtn.addEventListener('click', () => transcribe());

  async function transcribe() {
    if (!state.file) return;
    els.submitBtn.disabled = true;
    const fd = new FormData();
    fd.append('audio', state.file);
    fd.append('engine', state.engine);
    fd.append('language', els.langSelect.value);

    try {
      const res = await fetch(API.transcribe, { method: 'POST', body: fd, credentials: 'same-origin' });
      if (res.status === 401) { window.location.replace('/login.html'); return; }
      if (!res.ok) throw new Error(await responseMessage(res));
      const job = await res.json();
      upsertJobs([job]);
      renderJobs();
      ensurePolling();
      toast('Transcriptie gestart. Je kunt meteen verder.');
      clearSelectedFile();
      showOnly(els.upload);
    } catch (err) {
      showError(err.message || 'Onbekende fout. Probeer het opnieuw.');
      els.submitBtn.disabled = !state.file;
    }
  }

  function clearSelectedFile() {
    state.file = null;
    els.fileInput.value = '';
    els.dropzone.classList.remove('has-file');
    els.submitBtn.disabled = true;
    els.hint.textContent = 'Kies een audiobestand om te beginnen.';
    els.hint.classList.remove('has-file');
    resetDropzoneCopy();
  }

  // ----- Jobs panel -----
  async function refreshJobs({ once = false } = {}) {
    if (document.hidden && !once) return;
    try {
      const r = await fetch(API.jobs, { credentials: 'same-origin' });
      if (r.status === 401) { window.location.replace('/login.html'); return; }
      if (!r.ok) throw new Error('kon jobs niet laden');
      const data = await r.json();
      upsertJobs(data.items || []);
      renderJobs();
      ensurePolling();
    } catch (e) {
      console.warn('jobs refresh failed:', e);
    }
  }

  function upsertJobs(jobs) {
    jobs.forEach((job) => {
      const previous = state.jobs.get(job.id);
      state.jobs.set(job.id, { ...previous, ...job, seen_done_at: previous?.seen_done_at });
      const current = state.jobs.get(job.id);
      if (TERMINAL_STATUSES.has(current.status) && !current.seen_done_at) current.seen_done_at = Date.now();
    });
  }

  function renderJobs() {
    const jobs = Array.from(state.jobs.values())
      .filter((job) => ACTIVE_STATUSES.has(job.status) || (TERMINAL_STATUSES.has(job.status) && Date.now() - (job.seen_done_at || 0) < 10 * 60 * 1000))
      .sort((a, b) => (b.created_at || 0) - (a.created_at || 0));
    els.activeJobs.hidden = jobs.length === 0;
    els.activeJobsCount.textContent = String(jobs.filter((j) => ACTIVE_STATUSES.has(j.status)).length);
    els.activeJobsList.innerHTML = '';
    jobs.forEach((job) => els.activeJobsList.appendChild(jobRow(job)));
  }

  function jobRow(job) {
    const row = document.createElement('div');
    row.className = `job-row job-${job.status}`;
    const started = job.started_at || job.created_at || Date.now() / 1000;
    const elapsed = Math.max(0, Math.round(((job.finished_at || Date.now() / 1000) - started)));
    const pct = Math.max(0, Math.min(100, Math.round((job.progress || 0) * 100)));
    row.innerHTML = `
      <div class="job-main">
        <div class="job-title-line">
          <strong>${escapeHtml(job.filename || 'Audio')}</strong>
          <span class="badge">${engineLabel(job.engine_requested || job.engine_used)}</span>
        </div>
        <div class="job-meta">${jobStatusLabel(job)} · ${formatTime(elapsed)}</div>
        <div class="job-progress ${pct === 0 && ACTIVE_STATUSES.has(job.status) ? 'indeterminate' : ''}"><span style="width:${pct}%"></span></div>
      </div>
      <div class="job-actions"></div>`;
    const actions = row.querySelector('.job-actions');
    if (job.status === 'done' && job.transcript_id) {
      row.classList.add('job-ready');
      const btn = button('Klaar — Transcript bekijken', 'action small');
      btn.addEventListener('click', () => loadTranscript(job.transcript_id));
      actions.appendChild(btn);
      row.addEventListener('click', (e) => {
        if (e.target.closest('button')) return;
        loadTranscript(job.transcript_id);
      });
    } else if (ACTIVE_STATUSES.has(job.status)) {
      const btn = button('Annuleer', 'action ghost small');
      btn.addEventListener('click', () => cancelJob(job.id));
      actions.appendChild(btn);
    } else if (job.status === 'failed') {
      const msg = document.createElement('span');
      msg.className = 'job-error';
      msg.textContent = job.error || 'Mislukt';
      actions.appendChild(msg);
    }
    return row;
  }

  function button(text, className) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = className;
    btn.textContent = text;
    return btn;
  }

  async function cancelJob(id) {
    try {
      const r = await fetch(API.job(id), { method: 'DELETE', credentials: 'same-origin' });
      if (!r.ok) throw new Error(await responseMessage(r));
      upsertJobs([await r.json()]);
      renderJobs();
      ensurePolling();
    } catch (e) {
      toast(e.message || 'Kon job niet annuleren.');
    }
  }

  function ensurePolling() {
    const hasActive = Array.from(state.jobs.values()).some((job) => ACTIVE_STATUSES.has(job.status));
    if (!hasActive || document.hidden) {
      stopPolling();
      return;
    }
    if (!state.pollingTimer) state.pollingTimer = window.setInterval(() => refreshJobs(), 3000);
  }

  function stopPolling() {
    if (state.pollingTimer) {
      window.clearInterval(state.pollingTimer);
      state.pollingTimer = null;
    }
  }

  document.addEventListener('visibilitychange', () => {
    if (document.hidden) stopPolling();
    else refreshJobs({ once: true });
  });
  window.setInterval(() => renderJobs(), 1000);

  function jobStatusLabel(job) {
    if (job.status === 'queued') return 'In wachtrij';
    if (job.status === 'running') return 'Bezig';
    if (job.status === 'done') return 'Klaar!';
    if (job.status === 'cancelled') return 'Geannuleerd';
    return 'Mislukt';
  }

  function engineLabel(e) {
    return ({ parakeet: 'Snel', whisper: 'Zorgvuldig', voxtral: 'Voxtral' })[e] || e || 'Snel';
  }
  function langLabel(c) {
    return ({ nl: 'Nederlands', en: 'English', fr: 'Français', de: 'Deutsch', es: 'Español', it: 'Italiano', pt: 'Português', auto: 'Automatisch' })[c] || c;
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
    [els.upload, els.result, els.error, els.history, els.settings].forEach((s) => {
      if (s) s.hidden = (s !== section);
    });
  }

  function showResult(data, fileName) {
    showOnly(els.result);
    const text = (data.text || '').trim();
    state.transcriptText = text;
    state.transcriptName = (fileName || 'transcript').replace(/\.[^.]+$/, '');

    els.transcript.innerHTML = '';
    const paragraphs = (data.segments && data.segments.length) ? groupSegments(data.segments) : splitSentences(text);
    paragraphs.forEach((p, i) => {
      const el = document.createElement('p');
      el.textContent = p;
      el.style.opacity = '0';
      el.style.transform = 'translateY(6px)';
      el.style.transition = 'opacity 400ms ease, transform 400ms ease';
      el.style.transitionDelay = `${i * 60}ms`;
      els.transcript.appendChild(el);
      requestAnimationFrame(() => requestAnimationFrame(() => {
        el.style.opacity = '1';
        el.style.transform = 'none';
      }));
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
    for (let i = 0; i < parts.length; i += 3) out.push(parts.slice(i, i + 3).join(' ').trim());
    return out.length ? out : [text];
  }

  function showError(msg) {
    showOnly(els.error);
    els.errorText.textContent = msg;
  }

  function toast(msg) {
    els.hint.textContent = msg;
    els.hint.classList.remove('has-file');
  }

  function reset() {
    clearSelectedFile();
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
    if (!els.userPill?.contains(e.target)) closeUserMenu();
  });

  function closeUserMenu() {
    if (els.userMenu) {
      els.userMenu.hidden = true;
      els.userBtn?.setAttribute('aria-expanded', 'false');
    }
  }

  els.logoutBtn?.addEventListener('click', async () => {
    try { await fetch(API.logout, { method: 'POST', credentials: 'same-origin' }); } catch (_) {}
    window.location.replace('/login.html');
  });

  // ----- Settings -----
  els.settingsBtn?.addEventListener('click', () => {
    closeUserMenu();
    openSettings();
  });
  els.settingsClose?.addEventListener('click', () => showOnly(els.upload));
  els.refreshUsers?.addEventListener('click', () => loadUsers());

  async function openSettings() {
    showOnly(els.settings);
    clearMessage(els.passwordMessage);
    clearMessage(els.usersMessage);
    if (state.user?.is_admin) await loadUsers();
  }

  els.passwordForm?.addEventListener('submit', async (e) => {
    e.preventDefault();
    clearMessage(els.passwordMessage);
    if (els.newPassword.value !== els.confirmPassword.value) {
      setMessage(els.passwordMessage, 'Nieuwe wachtwoorden komen niet overeen.', true);
      return;
    }
    try {
      const r = await fetch(API.password, {
        method: 'POST', credentials: 'same-origin', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ current_password: els.currentPassword.value, new_password: els.newPassword.value }),
      });
      if (!r.ok) throw new Error(await responseMessage(r));
      els.passwordForm.reset();
      setMessage(els.passwordMessage, 'Wachtwoord opgeslagen.', false);
    } catch (err) {
      setMessage(els.passwordMessage, err.message || 'Kon wachtwoord niet opslaan.', true);
    }
  });

  async function loadUsers() {
    if (!state.user?.is_admin) return;
    els.usersList.innerHTML = '<p class="history-empty">Even laden…</p>';
    try {
      const r = await fetch(API.adminUsers, { credentials: 'same-origin' });
      if (!r.ok) throw new Error(await responseMessage(r));
      renderUsers(await r.json());
    } catch (err) {
      els.usersList.innerHTML = '';
      setMessage(els.usersMessage, err.message || 'Kon gebruikers niet laden.', true);
    }
  }

  function renderUsers(users) {
    els.usersList.innerHTML = '';
    users.forEach((user) => {
      const row = document.createElement('div');
      row.className = 'user-row';
      row.innerHTML = `
        <div>
          <strong>${escapeHtml(user.display_name || user.username)}</strong>
          <span>${escapeHtml(user.username)}</span>
        </div>
        <div class="user-row-actions">
          ${user.is_admin ? '<span class="badge">beheerder</span>' : ''}
        </div>`;
      const actions = row.querySelector('.user-row-actions');
      const del = button('Verwijder', 'action ghost small');
      del.disabled = user.id === state.user?.id;
      del.title = del.disabled ? 'Je kunt jezelf niet verwijderen' : '';
      del.addEventListener('click', () => deleteUser(user.id));
      actions.appendChild(del);
      els.usersList.appendChild(row);
    });
  }

  async function deleteUser(id) {
    if (id === state.user?.id) return;
    try {
      const r = await fetch(API.adminUser(id), { method: 'DELETE', credentials: 'same-origin' });
      if (!r.ok) throw new Error(await responseMessage(r));
      setMessage(els.usersMessage, 'Gebruiker verwijderd.', false);
      await loadUsers();
    } catch (err) {
      setMessage(els.usersMessage, err.message || 'Kon gebruiker niet verwijderen.', true);
    }
  }

  els.newUserForm?.addEventListener('submit', async (e) => {
    e.preventDefault();
    clearMessage(els.usersMessage);
    try {
      const r = await fetch(API.adminUsers, {
        method: 'POST', credentials: 'same-origin', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          username: els.newUserUsername.value,
          display_name: els.newUserDisplay.value,
          password: els.newUserPassword.value,
          is_admin: els.newUserAdmin.checked,
        }),
      });
      if (!r.ok) throw new Error(await responseMessage(r));
      els.newUserForm.reset();
      setMessage(els.usersMessage, 'Gebruiker toegevoegd.', false);
      await loadUsers();
    } catch (err) {
      setMessage(els.usersMessage, err.message || 'Kon gebruiker niet toevoegen.', true);
    }
  });

  function setMessage(el, msg, isError) {
    el.hidden = false;
    el.textContent = msg;
    el.classList.toggle('is-error', isError);
  }

  function clearMessage(el) {
    if (!el) return;
    el.hidden = true;
    el.textContent = '';
    el.classList.remove('is-error');
  }

  // ----- History -----
  els.historyBtn?.addEventListener('click', () => {
    closeUserMenu();
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
      const meta = [fmtDate(it.created_at), langLabel(it.language || 'auto'), engineLabel(it.engine)];
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

  async function responseMessage(res) {
    let msg = `Server gaf ${res.status} terug`;
    try {
      const j = await res.json();
      if (j?.error || j?.detail) msg = j.error || j.detail;
    } catch (_) {}
    return msg;
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

  bootstrap().then(() => { refreshEngineAvailability(); });
})();
