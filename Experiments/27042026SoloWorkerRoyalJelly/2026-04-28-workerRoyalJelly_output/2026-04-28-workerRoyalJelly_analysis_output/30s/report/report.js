
const payload = window.REPORT_META;
const beeSelect = document.getElementById('beeSelect');
const modeSelect = document.getElementById('modeSelect');
const summaryCards = document.getElementById('summaryCards');
const shapeList = document.getElementById('shapeList');
const statusEl = document.getElementById('status');

function pathFor(kind, source, beeId){ return payload.chunk_index?.[kind]?.[source]?.[beeId] || null; }
function keyFor(kind, source, beeId){ return `${kind}|${source}|${beeId}`; }
function loadScript(relPath){
  return new Promise((resolve, reject) => {
    const existing = document.querySelector(`script[data-rel="${relPath}"]`);
    if (existing) { resolve(); return; }
    const s = document.createElement('script');
    s.src = relPath;
    s.dataset.rel = relPath;
    s.onload = () => resolve();
    s.onerror = () => reject(new Error(`Failed to load ${relPath}`));
    document.body.appendChild(s);
  });
}
async function ensureChunk(kind, source, beeId){
  const key = keyFor(kind, source, beeId);
  if (window.REPORT_CHUNKS[key]) return window.REPORT_CHUNKS[key];
  const relPath = pathFor(kind, source, beeId);
  if (!relPath) return [];
  await loadScript(relPath);
  return window.REPORT_CHUNKS[key] || [];
}
function rulerShapes(maxTime){
  const shapes = [];
  for(let t=600; t<=maxTime+1e-9; t+=600){ shapes.push({type:'line', x0:t, x1:t, y0:0, y1:1, xref:'x', yref:'paper', line:{color:'rgba(120,120,120,0.30)', width:1, dash:'dot'}}); }
  for(let t=3600; t<=maxTime+1e-9; t+=3600){ shapes.push({type:'line', x0:t, x1:t, y0:0, y1:1, xref:'x', yref:'paper', line:{color:'rgba(60,60,60,0.55)', width:2}}); }
  return shapes;
}
function arenaStimTraces(){
  const shapes = [];
  Object.values(payload.arenas || {}).forEach(shape => {
    if(shape.shape === 'rect'){ shapes.push({type:'rect', x0:shape.geom[0], y0:shape.geom[1], x1:shape.geom[0]+shape.geom[2], y1:shape.geom[1]+shape.geom[3], line:{color:'rgba(255,255,255,0.8)', width:2}}); }
    else if(shape.shape === 'circle'){ shapes.push({type:'circle', x0:shape.geom[0]-shape.geom[2], y0:shape.geom[1]-shape.geom[2], x1:shape.geom[0]+shape.geom[2], y1:shape.geom[1]+shape.geom[2], line:{color:'rgba(255,255,255,0.8)', width:2}}); }
  });
  Object.values(payload.stimuli || {}).forEach(shape => {
    if(shape.shape === 'rect'){ shapes.push({type:'rect', x0:shape.geom[0], y0:shape.geom[1], x1:shape.geom[0]+shape.geom[2], y1:shape.geom[1]+shape.geom[3], line:{color:'rgba(255,170,0,0.9)', width:2}}); }
    else if(shape.shape === 'circle'){ shapes.push({type:'circle', x0:shape.geom[0]-shape.geom[2], y0:shape.geom[1]-shape.geom[2], x1:shape.geom[0]+shape.geom[2], y1:shape.geom[1]+shape.geom[2], line:{color:'rgba(255,170,0,0.9)', width:2}}); }
  });
  return shapes;
}
function initControls(){
  payload.summary.bee_ids.forEach(id => {
    const opt = document.createElement('option'); opt.value = id; opt.textContent = id; beeSelect.appendChild(opt);
  });
  beeSelect.value = payload.summary.default_bee || payload.summary.bee_ids[0] || '';
  if (!payload.summary.available_sources.includes('filtered')) modeSelect.value = 'raw';
}
function renderStaticSummary(compare){
  const total = compare.length;
  const detected = compare.filter(r => r.state === 'detected').length;
  const predicted = compare.filter(r => r.state === 'predicted').length;
  const missing = compare.filter(r => r.state === 'missing').length;
  summaryCards.innerHTML = [ ['Frames', total], ['Detected', detected], ['Predicted', predicted], ['Missing', missing], ['FPS', payload.summary.fps], ['Scale (mm/px)', payload.summary.pixel_to_mm] ].map(([label, value]) => `<div class="card"><div class="metric-label">${label}</div><div class="metric-value">${value}</div></div>`).join('');
  const pills = [];
  Object.values(payload.arenas || {}).forEach(a => pills.push(`<div class="info-pill"><strong>${a.id}</strong> arena</div>`));
  Object.values(payload.stimuli || {}).forEach(s => pills.push(`<div class="info-pill"><strong>${s.id}</strong> stimulus</div>`));
  shapeList.innerHTML = pills.join('');
}
function transparentLayout(extra){ return Object.assign({paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)', font:{color:'#e5e7eb'}, margin:{l:50,r:20,t:20,b:45}}, extra||{}); }
async function refresh(){
  const beeId = beeSelect.value;
  const mode = modeSelect.value;
  statusEl.textContent = `Loading ${beeId} (${mode})...`;
  const compare = await ensureChunk('compare', 'compare', beeId);
  let primary = [];
  let raw = [];
  let filtered = [];
  if (mode === 'compare') {
    if (payload.summary.available_sources.includes('raw')) raw = await ensureChunk('framewise', 'raw', beeId);
    if (payload.summary.available_sources.includes('filtered')) filtered = await ensureChunk('framewise', 'filtered', beeId);
    primary = filtered.length ? filtered : raw;
  } else {
    primary = await ensureChunk('framewise', mode, beeId);
  }
  const binnedSource = mode === 'compare' ? (payload.summary.available_sources.includes('filtered') ? 'filtered' : 'raw') : mode;
  const binned = await ensureChunk('binned', binnedSource, beeId);
  renderStaticSummary(compare);
  statusEl.textContent = `Loaded ${beeId} (${mode}).`;
  const maxPrimary = primary.length ? primary[primary.length-1].time_s : 0;
  const maxCompare = compare.length ? compare[compare.length-1].time_s : 0;
  const traj = [];
  if (mode === 'compare') { if (raw.length) traj.push({x:raw.map(r=>r.x), y:raw.map(r=>r.y), mode:'lines', name:'Raw'}); if (filtered.length) traj.push({x:filtered.map(r=>r.x), y:filtered.map(r=>r.y), mode:'lines', name:'Filtered', line:{dash:'dot'}}); }
  else { traj.push({x:primary.map(r=>r.x), y:primary.map(r=>r.y), mode:'lines', name:mode}); }
  Plotly.newPlot('trajectoryChart', traj, transparentLayout({xaxis:{title:'X (px)', range:[0, payload.summary.frame_width]}, yaxis:{title:'Y (px)', range:[payload.summary.frame_height,0]}, shapes:arenaStimTraces()}), {responsive:true});
  const stateMap = {missing:0, predicted:1, detected:2};
  Plotly.newPlot('compareChart', [{x:compare.map(r=>r.time_s), y:compare.map(r=>r.raw_x), mode:'lines', name:'Raw X'}, {x:compare.map(r=>r.time_s), y:compare.map(r=>r.filtered_x), mode:'lines', name:'Filtered X', line:{dash:'dot'}}, {x:compare.map(r=>r.time_s), y:compare.map(r=>stateMap[r.state]), mode:'lines', name:'State', yaxis:'y2'}], transparentLayout({xaxis:{title:'Time (s)'}, yaxis:{title:'X position (px)'}, yaxis2:{title:'State', overlaying:'y', side:'right', tickvals:[0,1,2], ticktext:['missing','predicted','detected']}, shapes:rulerShapes(maxCompare)}), {responsive:true});
  Plotly.newPlot('positionChart', [{x:primary.map(r=>r.time_s), y:primary.map(r=>r.x), mode:'lines', name:'X'}, {x:primary.map(r=>r.time_s), y:primary.map(r=>r.y), mode:'lines', name:'Y'}], transparentLayout({xaxis:{title:'Time (s)'}, yaxis:{title:'Position (px)'}, shapes:rulerShapes(maxPrimary)}), {responsive:true});
  Plotly.newPlot('angleChart', [{x:primary.map(r=>r.time_s), y:primary.map(r=>r.ang), mode:'lines', name:'Angle'}, {x:primary.map(r=>r.time_s), y:primary.map(r=>r.angle_change_deg), mode:'lines', name:'Angle change', yaxis:'y2'}], transparentLayout({xaxis:{title:'Time (s)'}, yaxis:{title:'Angle (deg)'}, yaxis2:{title:'Δ angle (deg)', overlaying:'y', side:'right'}, shapes:rulerShapes(maxPrimary)}), {responsive:true});
  Plotly.newPlot('speedChart', [{x:primary.map(r=>r.time_s), y:primary.map(r=>r.speed_mm_s), mode:'lines', name:'Speed'}], transparentLayout({xaxis:{title:'Time (s)'}, yaxis:{title:'Speed (mm/s)'}, shapes:rulerShapes(maxPrimary)}), {responsive:true});
  const insideCols = primary.length ? Object.keys(primary[0]).filter(k => k.startsWith('inside_')) : [];
  const stateTraces = insideCols.map(col => ({x:primary.map(r=>r.time_s), y:primary.map(r=>r[col]), mode:'lines', name:col, line:{shape:'hv'}}));
  stateTraces.push({x:compare.map(r=>r.time_s), y:compare.map(r=>stateMap[r.state]), mode:'lines', name:'state', yaxis:'y2', line:{dash:'dot'}});
  Plotly.newPlot('stateChart', stateTraces, transparentLayout({xaxis:{title:'Time (s)'}, yaxis:{title:'Inside (0/1)', range:[-0.2,1.2]}, yaxis2:{title:'State', overlaying:'y', side:'right', tickvals:[0,1,2], ticktext:['missing','predicted','detected']}, shapes:rulerShapes(maxCompare)}), {responsive:true});
  const distCols = primary.length ? Object.keys(primary[0]).filter(k => k.startsWith('dist_to_')) : [];
  Plotly.newPlot('distanceChart', distCols.map(col => ({x:primary.map(r=>r.time_s), y:primary.map(r=>r[col]), mode:'lines', name:col})), transparentLayout({xaxis:{title:'Time (s)'}, yaxis:{title:'Distance (mm)'}, shapes:rulerShapes(maxPrimary)}), {responsive:true});
  const time = binned.map(r => r.Time_End_s);
  const binnedTraces = [];
  if (payload.selected_metrics.speed && binned.length && binned[0].AvgSpeed_mm_s !== undefined) binnedTraces.push({x:time, y:binned.map(r=>r.AvgSpeed_mm_s), mode:'lines+markers', name:'AvgSpeed_mm_s'});
  if (binned.length) Object.keys(binned[0]).forEach(key => { if (key.startsWith('AvgDistArenaCenter_') || key.startsWith('Duration_in_Stim_') || key.startsWith('AvgDistStimCenter_') || key.startsWith('Entries_to_Stim_') || key.startsWith('Exits_from_Stim_')) { const trace = {x:time, y:binned.map(r=>r[key]), mode:'lines+markers', name:key}; if (key.startsWith('Exits_from_Stim_')) trace.line = {dash:'dot'}; binnedTraces.push(trace); } });
  Plotly.newPlot('binnedChart', binnedTraces, transparentLayout({xaxis:{title:'Time (s)'}, yaxis:{title:'Binned value'}, shapes:rulerShapes(time.length ? time[time.length-1] : 0)}), {responsive:true});
}
initControls(); beeSelect.addEventListener('change', refresh); modeSelect.addEventListener('change', refresh); refresh();
