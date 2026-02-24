/* ============================================================
   DermaAI v2 — Single-Page Application Logic
   ============================================================ */

'use strict';

// ── State ──────────────────────────────────────────────────────────────────
const State = {
  sessionId: null,
  pendingQuestions: [],
  clarificationRound: 0,
  result: null,
  audit: null,
  pdfUrls: {},
  eventSource: null,
};

// ── View manager ───────────────────────────────────────────────────────────
const VIEWS = ['intake', 'clarification', 'analysis', 'review', 'complete'];

function showView(name) {
  VIEWS.forEach(v => {
    document.getElementById(`view-${v}`).classList.toggle('active', v === name);
  });
  // Update stepper
  document.querySelectorAll('.step').forEach(el => {
    const s = el.dataset.step;
    const idx = VIEWS.indexOf(s);
    const cur = VIEWS.indexOf(name);
    el.classList.toggle('active', s === name);
    el.classList.toggle('done', idx < cur);
  });
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ── Overlay helpers ────────────────────────────────────────────────────────
function showOverlay(msg = 'Processing…') {
  document.getElementById('overlay').classList.remove('hidden');
  document.getElementById('overlay-msg').textContent = msg;
}
function hideOverlay() {
  document.getElementById('overlay').classList.add('hidden');
}

// ── Form helpers ───────────────────────────────────────────────────────────
function csvToList(str) {
  if (!str || !str.trim()) return [];
  return str.split(',').map(s => s.trim()).filter(Boolean);
}

function buildProfileJson() {
  const v = id => document.getElementById(id)?.value.trim() || null;
  return JSON.stringify({
    name:                 v('f-name')       || 'Unknown',
    age:                  parseInt(v('f-age')) || null,
    sex:                  v('f-sex')        || null,
    gender:               v('f-gender')     || null,
    skin_tone:            v('f-skin-tone')  || null,
    occupation:           v('f-occupation') || null,
    caste:                v('f-caste')      || null,
    pincode:              v('f-pincode')    || null,
    known_allergies:      csvToList(document.getElementById('f-allergies')?.value),
    current_medications:  csvToList(document.getElementById('f-medications')?.value),
    past_skin_conditions: csvToList(document.getElementById('f-past-skin')?.value),
    family_skin_history:  v('f-family')     || null,
    notes:                v('f-notes')      || null,
  });
}

// ── Image drop-zone ────────────────────────────────────────────────────────
function initDropZone() {
  const zone    = document.getElementById('drop-zone');
  const input   = document.getElementById('f-image');
  const preview = document.getElementById('image-preview');
  const wrap    = document.getElementById('image-preview-wrap');
  const remove  = document.getElementById('image-remove');

  zone.addEventListener('click', e => {
    if (e.target !== remove && !wrap.contains(e.target)) input.click();
  });

  ['dragenter', 'dragover'].forEach(ev =>
    zone.addEventListener(ev, e => { e.preventDefault(); zone.classList.add('dragover'); })
  );
  ['dragleave', 'drop'].forEach(ev =>
    zone.addEventListener(ev, () => zone.classList.remove('dragover'))
  );
  zone.addEventListener('drop', e => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) setPreview(file);
  });

  input.addEventListener('change', () => {
    if (input.files[0]) setPreview(input.files[0]);
  });

  remove.addEventListener('click', () => {
    input.value = '';
    wrap.classList.add('hidden');
  });

  function setPreview(file) {
    const reader = new FileReader();
    reader.onload = e => {
      preview.src = e.target.result;
      wrap.classList.remove('hidden');
    };
    reader.readAsDataURL(file);

    // Sync to file input if dropped
    if (input.files.length === 0) {
      const dt = new DataTransfer();
      dt.items.add(file);
      input.files = dt.files;
    }
  }
}

// ── INTAKE FORM SUBMIT ─────────────────────────────────────────────────────
async function handleIntakeSubmit(e) {
  e.preventDefault();

  const name    = document.getElementById('f-name').value.trim();
  const age     = document.getElementById('f-age').value.trim();
  const sex     = document.getElementById('f-sex').value;
  const symptoms = document.getElementById('f-symptoms').value.trim();

  if (!name || !age || !sex || !symptoms) {
    alert('Please fill in all required fields (Name, Age, Sex, Symptom Description).');
    return;
  }

  showOverlay('Saving patient profile and running initial assessment…');

  const formData = new FormData();
  formData.append('profile_json', buildProfileJson());
  formData.append('symptom_text', symptoms);

  const imageFile = document.getElementById('f-image').files[0];
  if (imageFile) formData.append('image', imageFile);

  try {
    const res = await fetch('/api/start', { method: 'POST', body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Failed to start session');

    State.sessionId = data.session_id;
    State.pendingQuestions = data.questions || [];
    State.clarificationRound = 1;

    hideOverlay();

    if (State.pendingQuestions.length > 0) {
      renderClarificationQuestions(State.pendingQuestions, 1);
      showView('clarification');
    } else {
      await startAnalysis();
    }
  } catch (err) {
    hideOverlay();
    alert(`Error: ${err.message}`);
  }
}

// ── CLARIFICATION ──────────────────────────────────────────────────────────
function renderClarificationQuestions(questions, round) {
  document.getElementById('clarif-round').textContent = `Round ${round} of 2`;
  const container = document.getElementById('clarif-questions');
  container.innerHTML = '';
  questions.forEach((q, i) => {
    const div = document.createElement('div');
    div.className = 'clarif-question-item';
    div.innerHTML = `
      <span class="clarif-q-label">${escHtml(q)}</span>
      <input class="clarif-q-input" type="text" data-qi="${i}"
             placeholder="Your answer (press Enter to move on)" />
    `;
    container.appendChild(div);
  });
  // Focus first input
  const first = container.querySelector('input');
  if (first) setTimeout(() => first.focus(), 100);
}

async function handleClarificationSubmit() {
  const inputs = document.querySelectorAll('#clarif-questions .clarif-q-input');
  const answers = Array.from(inputs).map(inp => inp.value.trim());

  showOverlay('Processing your answers…');

  try {
    const res = await fetch(`/api/${State.sessionId}/clarify`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ answers }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Clarification error');

    const newQuestions = data.questions || [];
    hideOverlay();

    if (newQuestions.length > 0 && State.clarificationRound < 2) {
      State.clarificationRound++;
      State.pendingQuestions = newQuestions;
      renderClarificationQuestions(newQuestions, State.clarificationRound);
    } else {
      await startAnalysis();
    }
  } catch (err) {
    hideOverlay();
    alert(`Error: ${err.message}`);
  }
}

// ── PIPELINE ANALYSIS VIEW ─────────────────────────────────────────────────

// Ordered list matching the pipeline execution sequence
const AGENT_ORDER = [
  'Patient Profile Specialist',
  'Dermatology Colour Analyst',
  'Dermatology Texture and Lesion Surface Analyst',
  'Dermatology Morphology and Elevation Analyst',
  'Dermatology Border Analyst',
  'Dermatology Shape Analyst',
  'Clinical Symptom Decomposition Specialist',
  'Dermatology Research Analyst',
  'Dermatology Differential Diagnosis Specialist',
  'Clinical Mimic & Edge-Case Specialist',
  'Dermatology Treatment Protocol Specialist',
  'Chief Medical Officer (Dermatology)',
  'Medical Scribe & Patient Communicator',
];
const TOTAL_AGENTS = AGENT_ORDER.length;

// Display metadata per agent
const AGENT_META = {
  'Patient Profile Specialist':                    { icon: '👤', label: 'Patient Profile',       phase: 'Context' },
  'Dermatology Colour Analyst':                    { icon: '🎨', label: 'Colour Analysis',        phase: 'Visual Analysis' },
  'Dermatology Texture and Lesion Surface Analyst':{ icon: '🔬', label: 'Surface Texture',        phase: 'Visual Analysis' },
  'Dermatology Morphology and Elevation Analyst':  { icon: '📐', label: 'Elevation Profile',      phase: 'Visual Analysis' },
  'Dermatology Border Analyst':                    { icon: '🔍', label: 'Border Assessment',      phase: 'Visual Analysis' },
  'Dermatology Shape Analyst':                     { icon: '⬡', label: 'Shape Assessment',       phase: 'Visual Analysis' },
  'Clinical Symptom Decomposition Specialist':     { icon: '📋', label: 'Symptom Mapping',        phase: 'Clinical Intel' },
  'Dermatology Research Analyst':                  { icon: '📚', label: 'Literature Research',    phase: 'Clinical Intel' },
  'Dermatology Differential Diagnosis Specialist': { icon: '⚕️', label: 'Differential Dx',        phase: 'Diagnosis' },
  'Clinical Mimic & Edge-Case Specialist':         { icon: '🔄', label: 'Mimic Resolution',       phase: 'Diagnosis' },
  'Dermatology Treatment Protocol Specialist':     { icon: '💊', label: 'Treatment Protocol',     phase: 'Diagnosis' },
  'Chief Medical Officer (Dermatology)':           { icon: '🏥', label: 'CMO Final Decision',     phase: 'Synthesis' },
  'Medical Scribe & Patient Communicator':         { icon: '📝', label: 'Report Generation',      phase: 'Synthesis' },
};

// Pipeline counter (reset on each new run)
let _doneCount = 0;

// Fuzzy-match an incoming agent name to a key in AGENT_META
function _resolveAgent(name) {
  if (AGENT_META[name]) return name;
  return AGENT_ORDER.find(k =>
    name.toLowerCase().includes(k.toLowerCase()) ||
    k.toLowerCase().includes(name.toLowerCase())
  ) || null;
}

// Strip JSON syntax from a raw output snippet → short readable phrase
function _extractInsight(raw) {
  if (!raw) return '';
  // Pull first string value from JSON-like text: "key": "value"
  const jm = raw.match(/"[\w_]+":\s*"([^"]{4,80})"/);
  if (jm) return jm[1];
  // "Label: Value" line format (biodata text output)
  const lm = raw.match(/[A-Z][^:]{2,25}:\s*([^\n,{]{4,60})/);
  if (lm) return lm[1].trim();
  // Fallback: strip JSON chars
  return raw.replace(/[{}"`,\[\]\\]/g, '').replace(/\s+/g, ' ').trim().slice(0, 72) || '';
}

// Update the hero card
function _setHero(agentKey, opts = {}) {
  const hero    = document.getElementById('pipeline-hero');
  const phaseEl = document.getElementById('pipeline-hero-phase');
  const iconEl  = document.getElementById('pipeline-hero-icon');
  const nameEl  = document.getElementById('pipeline-hero-name');
  const dotsEl  = document.getElementById('pipeline-dots');

  hero.classList.remove('hero-complete', 'hero-error');

  if (opts.complete) {
    hero.classList.add('hero-complete');
    phaseEl.textContent = 'Complete';
    iconEl.textContent  = '✓';
    nameEl.textContent  = 'All stages finished — loading your results…';
    dotsEl.classList.add('hidden');
    return;
  }
  if (opts.error) {
    hero.classList.add('hero-error');
    phaseEl.textContent = 'Error';
    iconEl.textContent  = '⚠';
    nameEl.textContent  = opts.message || 'An error occurred.';
    dotsEl.classList.add('hidden');
    return;
  }

  const meta = AGENT_META[agentKey];
  if (!meta) return;
  phaseEl.textContent = meta.phase;
  iconEl.textContent  = meta.icon;
  nameEl.textContent  = meta.label;
  dotsEl.classList.remove('hidden');
}

// Update the progress bar and counter label
function _updateProgress() {
  const pct = Math.round((_doneCount / TOTAL_AGENTS) * 100);
  document.getElementById('pipeline-fill').style.width = `${pct}%`;
  document.getElementById('pipeline-count').textContent = `${_doneCount} / ${TOTAL_AGENTS}`;
}

// Append a completed-stage card to the grid
function _addCard(agentKey, insight) {
  const meta  = AGENT_META[agentKey] || { icon: '✓', label: agentKey };
  const grid  = document.getElementById('pipeline-grid');
  const label = document.getElementById('pipeline-section-label');

  label.classList.remove('hidden');

  const card = document.createElement('div');
  card.className = 'pipeline-card';
  card.innerHTML = `
    <span class="pipeline-card-icon">${meta.icon}</span>
    <span class="pipeline-card-name">${meta.label}</span>
    <span class="pipeline-card-badge">✓ Complete</span>
    ${insight ? `<p class="pipeline-card-insight">${escHtml(insight)}</p>` : ''}
  `;
  grid.appendChild(card);
}

// ── Public pipeline API ────────────────────────────────────────

// Called before each run to wipe the previous state
function resetAgentList() {
  _doneCount = 0;
  document.getElementById('pipeline-fill').style.width = '0%';
  document.getElementById('pipeline-count').textContent = `0 / ${TOTAL_AGENTS}`;
  document.getElementById('pipeline-grid').innerHTML = '';
  document.getElementById('pipeline-section-label').classList.add('hidden');
  document.getElementById('pipeline-hero').classList.remove('hero-complete', 'hero-error');
  document.getElementById('pipeline-dots').classList.remove('hidden');
  _setHero(AGENT_ORDER[0]);
}

// Called for every task_done SSE event
function markAgentDone(agentName, rawSummary) {
  const key     = _resolveAgent(agentName) || agentName;
  const insight = _extractInsight(rawSummary || '');

  _doneCount++;
  _updateProgress();
  _addCard(key, insight);

  // Advance hero to the next pending agent
  if (_doneCount < TOTAL_AGENTS) {
    _setHero(AGENT_ORDER[_doneCount]);
  }
}

// appendLog / clearLog: retained so existing call-sites compile; log panel removed.
function clearLog() { /* no-op */ }
function appendLog(msg, cls = '') {
  // Only surface error-class messages visually
  if (cls === 'error') _setHero(null, { error: true, message: msg });
}

// ── ANALYSIS ───────────────────────────────────────────────────────────────
async function startAnalysis() {
  showView('analysis');
  resetAgentList();

  try {
    const res = await fetch(`/api/${State.sessionId}/analyze`, { method: 'POST' });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Failed to start analysis');
  } catch (err) {
    _setHero(null, { error: true, message: `Could not start: ${err.message}` });
    return;
  }

  openEventStream();
}

function openEventStream() {
  if (State.eventSource) {
    State.eventSource.close();
    State.eventSource = null;
  }

  const es = new EventSource(`/api/${State.sessionId}/stream`);
  State.eventSource = es;

  es.onmessage = e => {
    let event;
    try { event = JSON.parse(e.data); } catch { return; }
    handleStreamEvent(event);
  };

  es.onerror = () => {
    es.close();
    setTimeout(fetchAndShowResult, 3000);
  };
}

function handleStreamEvent(event) {
  switch (event.type) {
    case 'connected':
      break;   // hero already shows first agent from resetAgentList

    case 'task_done':
      markAgentDone(event.agent, event.summary || '');
      break;

    case 'complete':
      _setHero(null, { complete: true });
      State.eventSource?.close();
      setTimeout(fetchAndShowResult, 1000);
      break;

    case 'error':
      _setHero(null, { error: true, message: event.message });
      State.eventSource?.close();
      alert(`Analysis error: ${event.message}`);
      break;
  }
}

// ── RESULT ─────────────────────────────────────────────────────────────────
async function fetchAndShowResult() {
  try {
    const res = await fetch(`/api/${State.sessionId}/result`);
    const data = await res.json();

    if (data.status === 'analyzing') {
      setTimeout(fetchAndShowResult, 3000);
      return;
    }
    if (data.status === 'error') {
      alert(`Analysis error: ${data.error}`);
      return;
    }

    State.result = data.result;
    State.audit  = data.audit;

    renderReviewView(State.result, State.audit);
    showView('review');
  } catch (err) {
    alert(`Failed to load results: ${err.message}`);
  }
}

// ── RENDER REVIEW ──────────────────────────────────────────────────────────
function renderReviewView(result, audit) {
  if (!result) return;

  // Run banner
  const runCount = audit?.run_count ?? 1;
  document.getElementById('run-count-badge').textContent = `Run #${runCount}`;
  const histLabel = document.getElementById('feedback-history-label');
  histLabel.textContent = runCount > 1 ? `${runCount - 1} feedback round(s) completed` : '';

  // ── Doctor strip ───────────────────────────────────────────────────────
  const finalDx = (result.primary_diagnosis && result.primary_diagnosis !== 'Unknown')
    ? result.primary_diagnosis
    : (audit?.differential_output?.primary_diagnosis || '—');
  const showFallback = (!result.primary_diagnosis || result.primary_diagnosis === 'Unknown') && audit?.differential_output?.primary_diagnosis;
  const stripValueEl = document.getElementById('doc-primary-diagnosis');
  stripValueEl.textContent = finalDx;
  stripValueEl.title = showFallback ? 'Scribe did not return a diagnosis; showing differential agent’s proposed primary' : '';
  document.getElementById('doc-confidence').textContent = `${(result.confidence || '').toUpperCase()} confidence`;
  setSeverityBadge('doc-severity', result.severity);

  const reBadge = document.getElementById('doc-rediagnosis-badge');
  if (result.re_diagnosis_applied) {
    reBadge.classList.remove('hidden');
  } else {
    reBadge.classList.add('hidden');
  }

  // Lesion profile table
  const lesionBody = document.querySelector('#doc-lesion-table tbody');
  lesionBody.innerHTML = '';
  const lp = result.lesion_profile || {};
  const lesionRows = [
    ['Colour',    audit?.colour_output?.lesion_colour  || lp.colour    || '—', audit?.colour_output?.reason    || ''],
    ['Surface',   audit?.texture_output?.surface       || lp.texture   || '—', audit?.texture_output?.reason   || ''],
    ['Elevation', audit?.levelling_output?.levelling   || lp.levelling || '—', audit?.levelling_output?.reason || ''],
    ['Border',    audit?.border_output?.border         || lp.border    || '—', audit?.border_output?.reason    || ''],
    ['Shape',     audit?.shape_output?.shape           || lp.shape     || '—', audit?.shape_output?.reason     || ''],
    ['Pattern',   audit?.pattern_output?.pattern       || lp.pattern   || '—', audit?.pattern_output?.reason   || ''],
  ];
  lesionRows.forEach(([label, val, reason]) => {
    const tr = document.createElement('tr');
    const reasonHtml = reason ? ` <span style="font-size:.75rem;color:var(--text-muted)">${escHtml(reason)}</span>` : '';
    tr.innerHTML = `<td>${label}</td><td>${escHtml(val)}${reasonHtml}</td>`;
    lesionBody.appendChild(tr);
  });

  // Decomposition
  const dc = document.getElementById('doc-decomp');
  const dec = audit?.decomposition_output;
  if (dec) {
    dc.innerHTML = `
      <table class="detail-table"><tbody>
        <tr><td>Symptoms</td><td>${escHtml((dec.symptoms || []).slice(0,5).join(', '))}</td></tr>
        <tr><td>Duration</td><td>${dec.time_days ?? '—'} days</td></tr>
        <tr><td>Onset</td><td>${escHtml(dec.onset || '—')}</td></tr>
        <tr><td>Progression</td><td>${escHtml(dec.progression || '—')}</td></tr>
        <tr><td>Location</td><td>${escHtml((dec.body_location || []).join(', ') || '—')}</td></tr>
        <tr><td>Occupation exposure</td><td>${escHtml((dec.occupational_exposure || []).join(', ') || '—')}</td></tr>
      </tbody></table>
    `;
  } else {
    dc.innerHTML = '<p class="muted">No decomposition data.</p>';
  }

  // Research
  const rc = document.getElementById('doc-research');
  const res2 = audit?.research_output;
  if (res2) {
    const findings = (res2.key_findings || []).map(f => `<li>${escHtml(f)}</li>`).join('');
    const pmids = (res2.cited_pmids || []).map(p => {
      const id = escHtml(String(p).trim());
      return `<a class="tag pmid-link" href="https://pubmed.ncbi.nlm.nih.gov/${id}/" target="_blank" rel="noopener noreferrer" title="Open on PubMed">PMID ${id}</a>`;
    }).join('');
    rc.innerHTML = `
      <table class="detail-table" style="margin-bottom:.75rem"><tbody>
        <tr><td>Search Query</td><td>${escHtml(res2.primary_search_query || '—')}</td></tr>
        <tr><td>Evidence Strength</td><td>${escHtml(res2.evidence_strength || '—')}</td></tr>
        <tr><td>Articles Found</td><td>${res2.articles_found ?? '—'}</td></tr>
      </tbody></table>
      <p style="font-size:.8rem;font-weight:600;color:var(--text-muted);margin-bottom:.4rem">Key Findings</p>
      <ul class="diff-list for">${findings}</ul>
      ${pmids ? `<p style="font-size:.8rem;font-weight:600;color:var(--text-muted);margin:.75rem 0 .4rem">Cited Articles</p><div class="tag-list">${pmids}</div>` : ''}
    `;
  } else {
    rc.innerHTML = '<p class="muted">No research data.</p>';
  }

  // Differential diagnosis
  renderDifferential(audit?.differential_output, audit?.mimic_resolution_output, audit?.visual_differential_review_output, result);

  // Treatment plan
  renderTreatment(audit?.treatment_output);

  // Clinical reasoning
  document.getElementById('doc-reasoning').textContent = result.clinical_reasoning || '—';
  const reBlock = document.getElementById('doc-rediagnosis-block');
  const reReason = document.getElementById('doc-rediagnosis-reason');
  const proposedDx = audit?.differential_output?.primary_diagnosis?.trim();
  const finalDxVal = (result.primary_diagnosis || '').trim();
  if (result.re_diagnosis_applied) {
    reBlock.classList.remove('hidden');
    let reContent = '';
    if (proposedDx && finalDxVal && proposedDx.toLowerCase() !== finalDxVal.toLowerCase()) {
      reContent = `<strong>Proposed:</strong> ${escHtml(proposedDx)} → <strong>Final:</strong> ${escHtml(finalDxVal)}<br><br>`;
    }
    reContent += escHtml(result.re_diagnosis_reason || '').replace(/\n/g, '<br>');
    reReason.innerHTML = reContent;
  } else {
    reBlock.classList.add('hidden');
  }

  // Feedback history
  renderFeedbackHistory(audit?.feedback_history || [], audit?.adapter_status || {}, audit?.adapter_errors || {});

  // ── Patient tab ────────────────────────────────────────────────────────
  const patDx = (result.primary_diagnosis && result.primary_diagnosis !== 'Unknown')
    ? result.primary_diagnosis
    : (audit?.differential_output?.primary_diagnosis || '—');
  document.getElementById('pat-diagnosis').textContent = patDx;
  setSeverityBadge('pat-severity', result.severity);
  document.getElementById('pat-summary').textContent = result.patient_summary || '';

  const recList = document.getElementById('pat-recs');
  recList.innerHTML = (result.patient_recommendations || [])
    .map(r => `<li>${escHtml(r)}</li>`).join('');

  document.getElementById('pat-urgency').textContent = result.when_to_seek_care || '';
  document.getElementById('pat-disclaimer').textContent = result.disclaimer || '';
}

function renderDifferential(diff, mimic, vdr, result) {
  const container = document.getElementById('doc-diff');
  if (!diff) { container.innerHTML = '<p class="muted">No differential data.</p>'; return; }

  // Normalise for case-insensitive comparison
  const finalDx   = (result?.primary_diagnosis || '').trim().toLowerCase();
  const rejectedMimic = (mimic?.rejected_mimic || '').trim().toLowerCase();
  const reApplied = !!result?.re_diagnosis_applied;

  // Helper: find VDR vote for a condition (case-insensitive)
  function vdrVote(condition) {
    if (!vdr?.votes?.length) return null;
    const key = (condition || '').trim().toLowerCase();
    return vdr.votes.find(v => (v.condition || '').trim().toLowerCase() === key) || null;
  }

  // Primary diagnosis row status chip
  let primaryStatus = '';
  if (reApplied) {
    primaryStatus = '<span class="diff-status-chip revised">CMO REVISED</span>';
  } else if (finalDx && diff.primary_diagnosis && finalDx === diff.primary_diagnosis.trim().toLowerCase()) {
    primaryStatus = '<span class="diff-status-chip confirmed">CONFIRMED</span>';
  }

  let html = `
    <table class="detail-table" style="margin-bottom:1rem"><tbody>
      <tr>
        <td>Proposed Primary</td>
        <td><strong>${escHtml(diff.primary_diagnosis || '—')}</strong> ${primaryStatus}</td>
      </tr>
      <tr><td>Confidence</td><td>${escHtml(diff.confidence_in_primary || '—')}</td></tr>
      <tr><td>Primary Reasoning</td><td>${escHtml(diff.primary_reasoning || '—')}</td></tr>
    </tbody></table>
  `;

  if (vdr?.visual_winner) {
    const winnerConf = (vdr.visual_confidence || '').toUpperCase();
    html += `
      <div class="vdr-summary">
        <span class="vdr-label">Image Assessment:</span>
        <strong>${escHtml(vdr.visual_winner)}</strong>
        <span class="diff-prob ${(vdr.visual_confidence||'').toLowerCase()}" style="margin-left:.4rem">${winnerConf}</span>
        ${vdr.visual_reasoning_summary ? `<p class="vdr-reasoning">${escHtml(vdr.visual_reasoning_summary)}</p>` : ''}
      </div>`;
  }

  if (diff.red_flags?.length) {
    html += `<p class="diff-section-label" style="margin-bottom:.4rem">Red Flags</p>
    <div class="red-flag-list">
      ${diff.red_flags.map(f => `<span class="red-flag-chip">⚠ ${escHtml(f)}</span>`).join('')}
    </div>`;
  }

  html += `<p class="diff-section-label" style="margin:1rem 0 .5rem">Differentials</p>`;

  (diff.differentials || []).forEach(entry => {
    const condLower = (entry.condition || '').trim().toLowerCase();
    const forItems  = (entry.key_features_matching || []).map(f => `<li>${escHtml(f)}</li>`).join('');
    const agstItems = (entry.key_features_against  || []).map(f => `<li>${escHtml(f)}</li>`).join('');

    // Determine outcome status for this entry
    let statusChip = '';
    if (finalDx && condLower === finalDx) {
      statusChip = '<span class="diff-status-chip confirmed">CONFIRMED</span>';
    } else if (rejectedMimic && condLower === rejectedMimic) {
      statusChip = '<span class="diff-status-chip ruled-out">RULED OUT</span>';
    }

    // Visual vote chip
    const vote = vdrVote(entry.condition);
    let voteChip = '';
    if (vote) {
      if (vote.visually_consistent) {
        voteChip = `<span class="diff-status-chip img-yes" title="${escHtml(vote.visual_reasoning || '')}">✓ Image Consistent</span>`;
      } else {
        voteChip = `<span class="diff-status-chip img-no" title="${escHtml(vote.visual_reasoning || '')}">✗ Not Consistent</span>`;
      }
    }

    html += `
      <div class="diff-entry">
        <div class="diff-entry-header">
          <span class="diff-condition">${escHtml(entry.condition)}</span>
          <span class="diff-prob ${entry.probability}">${entry.probability?.toUpperCase()}</span>
          ${statusChip}
          ${voteChip}
        </div>
        <div class="diff-section">
          <p class="diff-section-label">Supporting Findings</p>
          <ul class="diff-list for">${forItems || '<li>—</li>'}</ul>
        </div>
        <div class="diff-section">
          <p class="diff-section-label">Against</p>
          <ul class="diff-list against">${agstItems || '<li>—</li>'}</ul>
        </div>
        <div class="diff-test"><strong>Confirm with:</strong> ${escHtml(entry.distinguishing_test || '—')}</div>
      </div>
    `;
  });

  container.innerHTML = html;
}

function renderTreatment(treat) {
  const container = document.getElementById('doc-treatment');
  if (!treat) { container.innerHTML = '<p class="muted">No treatment data.</p>'; return; }

  const immediate = (treat.immediate_actions || []).map(a => `<li>${escHtml(a)}</li>`).join('');
  const nonPharm  = (treat.non_pharmacological || []).map(a => `<li>${escHtml(a)}</li>`).join('');
  const contra    = (treat.contraindications || []).map(c => `<span class="tag" style="background:var(--danger-bg);color:var(--danger)">${escHtml(c)}</span>`).join('');

  const meds = (treat.medications || []).map(m => `
    <div class="treatment-med-entry ${m.line || ''}">
      <span class="treatment-med-line">${(m.line || '').toUpperCase()}-LINE</span>
      <div class="treatment-med-name">${escHtml(m.treatment_name)}</div>
      <div class="treatment-med-detail">${escHtml(m.dose_or_protocol)} · ${escHtml(m.duration)}</div>
      ${m.rationale ? `<div class="treatment-med-detail" style="margin-top:.3rem;font-style:italic">${escHtml(m.rationale)}</div>` : ''}
      ${m.monitoring ? `<div class="treatment-med-detail" style="color:var(--warning)">Monitor: ${escHtml(m.monitoring)}</div>` : ''}
    </div>
  `).join('');

  container.innerHTML = `
    <table class="detail-table" style="margin-bottom:1rem"><tbody>
      <tr><td>For Diagnosis</td><td>${escHtml(treat.for_diagnosis || '—')}</td></tr>
      <tr><td>Evidence Level</td><td>${escHtml(treat.evidence_level || '—')}</td></tr>
      <tr><td>Follow-up</td><td>${escHtml(treat.follow_up || '—')}</td></tr>
      <tr><td>Referral Needed</td><td>${treat.referral_needed ? `Yes — ${escHtml(treat.referral_to || 'Specialist')}` : 'No'}</td></tr>
    </tbody></table>

    ${immediate ? `<p class="diff-section-label" style="margin-bottom:.4rem">Immediate Actions</p>
    <ul class="diff-list for" style="margin-bottom:1rem">${immediate}</ul>` : ''}

    <p class="diff-section-label" style="margin-bottom:.5rem">Medication Protocol</p>
    <div class="treatment-meds">${meds}</div>

    ${nonPharm ? `<p class="diff-section-label" style="margin-bottom:.4rem">Non-Pharmacological</p>
    <ul class="diff-list for" style="margin-bottom:.75rem">${nonPharm}</ul>` : ''}

    <p class="diff-section-label" style="margin-bottom:.35rem">Patient Instructions</p>
    <p class="prose-text" style="margin-bottom:.75rem">${escHtml(treat.patient_instructions || '—')}</p>

    ${contra ? `<p class="diff-section-label" style="margin-bottom:.35rem">Contraindications (Patient-Specific)</p>
    <div class="tag-list">${contra}</div>` : ''}
  `;
}

function renderFeedbackHistory(history, adapterStatus = {}, adapterErrors = {}) {
  const container = document.getElementById('doc-feedback-history');
  const adapterLines = Object.entries(adapterStatus)
    .filter(([, status]) => status && status !== 'ok')
    .map(([k, status]) => {
      const err = adapterErrors[k] ? ` — ${escHtml(String(adapterErrors[k]).slice(0, 140))}` : '';
      return `<div class="feedback-round"><p class="feedback-round-text"><strong>${escHtml(k)}</strong>: ${escHtml(status)}${err}</p></div>`;
    }).join('');

  if (!history.length && !adapterLines) {
    container.innerHTML = '<p class="muted">No feedback rounds yet.</p>';
    return;
  }
  container.innerHTML = adapterLines + history.map(h => {
    if (h.action === 'approved') {
      return `<div class="feedback-round">
        <div class="feedback-round-header">
          <span class="feedback-round-num">Round ${h.round}</span>
        </div>
        <span class="feedback-approved">✓ Doctor Approved</span>
      </div>`;
    }
    return `<div class="feedback-round">
      <div class="feedback-round-header">
        <span class="feedback-round-num">Round ${h.round}</span>
        <span class="feedback-round-scope">${h.rerun_scope || ''}</span>
      </div>
      <p class="feedback-round-text">${escHtml(h.feedback || '')}</p>
    </div>`;
  }).join('');
}

// ── SEVERITY BADGE ─────────────────────────────────────────────────────────
function setSeverityBadge(elId, severity) {
  const el = document.getElementById(elId);
  if (!el) return;
  const s = (severity || '').toLowerCase();
  el.textContent = severity || '—';
  el.className = el.className.replace(/\b(mild|moderate|severe)\b/g, '');
  el.classList.add(s);
}

// ── TABS ───────────────────────────────────────────────────────────────────
function initTabs() {
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const tab = btn.dataset.tab;
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b === btn));
      document.querySelectorAll('.tab-pane').forEach(p => {
        p.classList.toggle('active', p.id === `tab-${tab}`);
      });
    });
  });
}

// ── APPROVAL ───────────────────────────────────────────────────────────────
async function handleApprove() {
  showOverlay('Doctor approval recorded. Generating reports…');

  try {
    const res = await fetch(`/api/${State.sessionId}/approve`, { method: 'POST' });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Approval failed');

    State.pdfUrls = data.pdf_urls || {};
    hideOverlay();
    renderCompleteView();
    showView('complete');
  } catch (err) {
    hideOverlay();
    alert(`Error: ${err.message}`);
  }
}

async function handleReject() {
  const feedback = document.getElementById('reject-feedback').value.trim();
  if (!feedback) {
    alert('Please describe what needs to change before rejecting.');
    return;
  }

  const scope = document.querySelector('input[name="scope"]:checked')?.value || 'full';

  showOverlay('Submitting feedback and starting re-analysis…');

  try {
    const res = await fetch(`/api/${State.sessionId}/reject`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ feedback, scope }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Reject failed');

    hideOverlay();

    // Reset rejection UI
    document.getElementById('reject-panel').classList.add('hidden');
    document.getElementById('reject-feedback').value = '';

    // Go back to analysis view to show re-run progress
    showView('analysis');
    resetAgentList();
    clearLog();
    appendLog(`Re-running with scope: ${scope}. Doctor feedback injected.`);
    openEventStream();
  } catch (err) {
    hideOverlay();
    alert(`Error: ${err.message}`);
  }
}

// ── COMPLETE VIEW ──────────────────────────────────────────────────────────
function renderCompleteView() {
  const urls = State.pdfUrls;
  const set = (id, url) => {
    const el = document.getElementById(id);
    if (el) { el.href = url || '#'; el.style.opacity = url ? '1' : '.4'; }
  };
  set('dl-doctor',  urls.doctor);
  set('dl-patient', urls.patient);
  set('dl-audit',   urls.audit);
}

// ── LOAD SAVED PROFILE ─────────────────────────────────────────────────────
async function handleLoadProfile() {
  try {
    const res = await fetch('/api/profile');
    if (!res.ok) { alert('No saved profile found.'); return; }
    const p = await res.json();

    const set = (id, val) => {
      const el = document.getElementById(id);
      if (el && val != null) el.value = val;
    };
    const setList = (id, arr) => {
      const el = document.getElementById(id);
      if (el && Array.isArray(arr) && arr.length) el.value = arr.join(', ');
    };

    set('f-name',       p.name !== 'Unknown' ? p.name : '');
    set('f-age',        p.age);
    set('f-sex',        p.sex);
    set('f-gender',     p.gender);
    set('f-skin-tone',  p.skin_tone);
    set('f-occupation', p.occupation);
    set('f-caste',      p.caste);
    set('f-pincode',    p.pincode);
    setList('f-allergies',   p.known_allergies);
    setList('f-medications', p.current_medications);
    setList('f-past-skin',   p.past_skin_conditions);
    set('f-family',     p.family_skin_history);
    set('f-notes',      p.notes);
  } catch (err) {
    alert(`Could not load profile: ${err.message}`);
  }
}

// ── NEW SESSION ────────────────────────────────────────────────────────────
function newSession() {
  State.sessionId = null;
  State.pendingQuestions = [];
  State.clarificationRound = 0;
  State.result = null;
  State.audit = null;
  State.pdfUrls = {};
  if (State.eventSource) { State.eventSource.close(); State.eventSource = null; }

  // Reset intake form
  document.getElementById('intake-form').reset();
  document.getElementById('image-preview-wrap').classList.add('hidden');

  showView('intake');
}

// ── UTILITY ────────────────────────────────────────────────────────────────
function escHtml(str) {
  if (str == null) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ── INIT ───────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initDropZone();
  initTabs();

  // Intake form
  document.getElementById('intake-form').addEventListener('submit', handleIntakeSubmit);

  // Clarification
  document.getElementById('clarif-submit').addEventListener('click', handleClarificationSubmit);
  document.getElementById('clarif-skip').addEventListener('click', async () => {
    // Submit empty answers
    document.querySelectorAll('#clarif-questions .clarif-q-input').forEach(i => i.value = '');
    await handleClarificationSubmit();
  });

  // Approval
  document.getElementById('btn-approve').addEventListener('click', handleApprove);

  document.getElementById('btn-reject').addEventListener('click', () => {
    document.getElementById('reject-panel').classList.remove('hidden');
    document.getElementById('reject-panel').scrollIntoView({ behavior: 'smooth' });
  });
  document.getElementById('btn-reject-cancel').addEventListener('click', () => {
    document.getElementById('reject-panel').classList.add('hidden');
  });
  document.getElementById('btn-reject-confirm').addEventListener('click', handleReject);

  // Load saved profile
  document.getElementById('btn-load-profile').addEventListener('click', handleLoadProfile);

  // New session
  document.getElementById('btn-new-session').addEventListener('click', newSession);

  // Start on intake view
  showView('intake');
});
