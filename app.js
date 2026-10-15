/**
 * RoboRoyale SoloQueen Observatory - Main Application Logic
 */

(function () {
  'use strict';

  // Elements
  const expListEl = document.getElementById('experimentList');
  const searchInput = document.getElementById('searchInput');
  const mobileMenuBtn = document.getElementById('mobileMenuBtn');
  const sidebar = document.getElementById('sidebar');

  const expTitleEl = document.getElementById('activeExpTitle');
  const expDurationBadge = document.getElementById('expDurationBadge');
  const expTypeBadge = document.getElementById('expTypeBadge');

  const heroTitle = document.getElementById('heroTitle');
  const heroDate = document.getElementById('heroDate');
  const heroDuration = document.getElementById('heroDuration');
  const heroFrames = document.getElementById('heroFrames');
  const heroAnalysisType = document.getElementById('heroAnalysisType');

  const modeFilteredBtn = document.getElementById('modeFilteredBtn');
  const modeRawBtn = document.getElementById('modeRawBtn');
  const prevExpBtn = document.getElementById('prevExpBtn');
  const nextExpBtn = document.getElementById('nextExpBtn');

  // Plots and Cards
  const analysisPlotImg = document.getElementById('analysisPlotImg');
  const analysisPlotLink = document.getElementById('analysisPlotLink');
  const analysisDownloadBtn = document.getElementById('analysisDownloadBtn');

  const heatmapImg = document.getElementById('heatmapImg');
  const heatmapLink = document.getElementById('heatmapLink');
  const heatmapDownloadBtn = document.getElementById('heatmapDownloadBtn');

  const trajectoryContainer = document.getElementById('trajectoryContainer');

  // Lightbox
  const lightboxModal = document.getElementById('lightboxModal');
  const lightboxImg = document.getElementById('lightboxImg');
  const lightboxTitle = document.getElementById('lightboxTitle');
  const lightboxCloseBtn = document.getElementById('lightboxCloseBtn');
  const zoomInBtn = document.getElementById('zoomInBtn');
  const zoomOutBtn = document.getElementById('zoomOutBtn');
  const zoomResetBtn = document.getElementById('zoomResetBtn');

  // State
  let experiments = typeof EXPERIMENTS_DATA !== 'undefined' ? EXPERIMENTS_DATA : [];
  let currentExpIndex = 6; // Default to Exp 7 (index 6: 7-28092026SoloQueen)
  let currentMode = 'filtered'; // 'filtered' | 'raw'

  // Lightbox zoom/pan state
  let currentScale = 1;
  let isPanning = false;
  let startX = 0, startY = 0;
  let translateX = 0, translateY = 0;

  // Initialize
  function init() {
    if (!experiments || experiments.length === 0) {
      console.error('No experiments data found in EXPERIMENTS_DATA.');
      return;
    }

    renderSidebarList(experiments);
    setupEventListeners();

    // Check URL hash (e.g. #exp-7 or #exp-1)
    const hash = window.location.hash.replace('#', '');
    if (hash) {
      const idx = experiments.findIndex(e => e.id === hash || `exp-${e.number}` === hash);
      if (idx !== -1) {
        currentExpIndex = idx;
      }
    }

    loadExperiment(currentExpIndex);
  }

  // Render Sidebar Experiment Items
  function renderSidebarList(list) {
    expListEl.innerHTML = '';
    list.forEach((exp, index) => {
      const btn = document.createElement('button');
      btn.className = `experiment-item ${index === currentExpIndex ? 'active' : ''}`;
      btn.dataset.id = exp.id;
      btn.dataset.index = index;

      const tagClass = exp.has_merged ? 'merged' : 'standard';
      const tagText = exp.has_merged ? '2-Part Merged' : 'Single Session';

      btn.innerHTML = `
        <div class="exp-info">
          <div class="exp-title-row">
            <span class="exp-badge">#${exp.number}</span>
            <span class="exp-name">${exp.display_title}</span>
          </div>
          <div class="exp-meta">
            <span>${exp.date}</span>
            <span>•</span>
            <span class="exp-tag ${tagClass}">${tagText}</span>
          </div>
        </div>
      `;

      btn.addEventListener('click', () => {
        const trueIndex = experiments.findIndex(e => e.id === exp.id);
        if (trueIndex !== -1) {
          selectExperiment(trueIndex);
        }
        if (window.innerWidth <= 1024) {
          sidebar.classList.remove('open');
        }
      });

      expListEl.appendChild(btn);
    });
  }

  // Load and Render Active Experiment
  function loadExperiment(index) {
    if (index < 0 || index >= experiments.length) return;
    currentExpIndex = index;
    const exp = experiments[index];

    // Update URL hash without jumping
    history.replaceState(null, null, `#${exp.id}`);

    // Update Sidebar Active state
    document.querySelectorAll('.experiment-item').forEach(item => {
      const itemIndex = parseInt(item.dataset.index, 10);
      const itemId = item.dataset.id;
      item.classList.toggle('active', itemId === exp.id);
    });

    // Update Top Header
    expTitleEl.textContent = `${exp.display_title} — ${exp.date}`;
    expDurationBadge.textContent = exp.duration !== 'N/A' ? `⏱ ${exp.duration}` : '⏱ N/A';
    
    if (exp.has_merged) {
      expTypeBadge.textContent = '2-Part Merged';
      expTypeBadge.className = 'badge-pill badge-duration exp-tag merged';
    } else {
      expTypeBadge.textContent = 'Standard';
      expTypeBadge.className = 'badge-pill badge-duration exp-tag standard';
    }

    // Update Hero Card
    heroTitle.textContent = `${exp.display_title} Analysis Report`;
    heroDate.textContent = exp.date;
    heroDuration.textContent = exp.duration;
    heroFrames.textContent = exp.total_frames ? Number(exp.total_frames).toLocaleString() : 'N/A';
    heroAnalysisType.textContent = exp.has_merged ? 'Merged Session (8h)' : 'Single Session (8h+)';

    // Update Navigation Buttons
    prevExpBtn.disabled = index === 0;
    nextExpBtn.disabled = index === experiments.length - 1;

    // Render Images
    updateImages();
  }

  // Update Visual Plots according to current mode (filtered / raw)
  function updateImages() {
    const exp = experiments[currentExpIndex];
    if (!exp) return;

    const plots = exp.plots[currentMode] || exp.plots.filtered;

    // 1. Analysis Plots
    if (plots.analysis) {
      analysisPlotImg.src = plots.analysis;
      analysisPlotLink.href = plots.analysis;
      analysisDownloadBtn.href = plots.analysis;
      analysisDownloadBtn.download = `${exp.id}_analysis_${currentMode}.png`;
      analysisPlotImg.parentElement.style.display = 'block';
    } else {
      analysisPlotImg.parentElement.style.display = 'none';
    }

    // 2. Heatmap
    if (plots.heatmap) {
      heatmapImg.src = plots.heatmap;
      heatmapLink.href = plots.heatmap;
      heatmapDownloadBtn.href = plots.heatmap;
      heatmapDownloadBtn.download = `${exp.id}_heatmap_${currentMode}.png`;
      heatmapImg.parentElement.style.display = 'block';
    } else {
      heatmapImg.parentElement.style.display = 'none';
    }

    // 3. Trajectories
    renderTrajectories(exp.trajectories);
  }

  // Render Trajectories (single or multi-part grid)
  function renderTrajectories(trajectories) {
    trajectoryContainer.innerHTML = '';

    if (!trajectories || trajectories.length === 0) {
      trajectoryContainer.innerHTML = '<div style="color:var(--text-muted); padding:20px;">No trajectory maps found.</div>';
      return;
    }

    if (trajectories.length === 1) {
      // Single Trajectory Map
      const traj = trajectories[0];
      const wrapper = document.createElement('div');
      wrapper.className = 'plot-image-wrapper';
      wrapper.innerHTML = `
        <img src="${traj.path}" alt="${traj.label}" class="plot-image" loading="lazy" />
        <div class="plot-hover-hint">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"></circle><path d="m21 21-4.35-4.35"></path></svg>
          Click to Zoom
        </div>
      `;
      wrapper.addEventListener('click', () => openLightbox(traj.path, traj.label));
      trajectoryContainer.appendChild(wrapper);
    } else {
      // Multi-part Trajectories (Part 1 and Part 2)
      const grid = document.createElement('div');
      grid.className = 'trajectory-grid';

      trajectories.forEach((traj, i) => {
        const item = document.createElement('div');
        item.className = 'trajectory-item';
        item.innerHTML = `
          <div class="trajectory-item-header">
            <span>${traj.label}</span>
            <a href="${traj.path}" download="${traj.filename}" class="card-action-btn" style="padding:2px 8px; font-size:0.7rem;">Download</a>
          </div>
          <div class="plot-image-wrapper">
            <img src="${traj.path}" alt="${traj.label}" class="plot-image" loading="lazy" />
            <div class="plot-hover-hint">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"></circle><path d="m21 21-4.35-4.35"></path></svg>
              Click to Zoom
            </div>
          </div>
        `;
        const wrap = item.querySelector('.plot-image-wrapper');
        wrap.addEventListener('click', () => openLightbox(traj.path, traj.label));
        grid.appendChild(item);
      });

      trajectoryContainer.appendChild(grid);
    }
  }

  // Event Listeners
  function setupEventListeners() {
    // Mode toggling (Filtered vs Raw)
    modeFilteredBtn.addEventListener('click', () => {
      if (currentMode !== 'filtered') {
        currentMode = 'filtered';
        modeFilteredBtn.classList.add('active');
        modeRawBtn.classList.remove('active');
        updateImages();
      }
    });

    modeRawBtn.addEventListener('click', () => {
      if (currentMode !== 'raw') {
        currentMode = 'raw';
        modeRawBtn.classList.add('active');
        modeFilteredBtn.classList.remove('active');
        updateImages();
      }
    });

    // Navigation buttons
    prevExpBtn.addEventListener('click', () => {
      if (currentExpIndex > 0) selectExperiment(currentExpIndex - 1);
    });

    nextExpBtn.addEventListener('click', () => {
      if (currentExpIndex < experiments.length - 1) selectExperiment(currentExpIndex + 1);
    });

    // Search filter
    searchInput.addEventListener('input', (e) => {
      const q = e.target.value.toLowerCase().trim();
      const filtered = experiments.filter(exp => 
        exp.display_title.toLowerCase().includes(q) ||
        exp.date.toLowerCase().includes(q) ||
        `exp ${exp.number}`.includes(q) ||
        exp.folder_name.toLowerCase().includes(q)
      );
      renderSidebarList(filtered);
    });

    // Mobile menu toggle
    mobileMenuBtn.addEventListener('click', () => {
      sidebar.classList.toggle('open');
    });

    // Lightbox image clicks
    analysisPlotImg.parentElement.addEventListener('click', () => {
      openLightbox(analysisPlotImg.src, `${experiments[currentExpIndex].display_title} - Analysis Plots (${currentMode.toUpperCase()})`);
    });

    heatmapImg.parentElement.addEventListener('click', () => {
      openLightbox(heatmapImg.src, `${experiments[currentExpIndex].display_title} - Spatial Density Heatmap (${currentMode.toUpperCase()})`);
    });

    // Lightbox modal controls
    lightboxCloseBtn.addEventListener('click', closeLightbox);
    lightboxModal.addEventListener('click', (e) => {
      if (e.target === lightboxModal || e.target.classList.contains('lightbox-body')) {
        closeLightbox();
      }
    });

    zoomInBtn.addEventListener('click', () => applyZoom(currentScale * 1.3));
    zoomOutBtn.addEventListener('click', () => applyZoom(currentScale / 1.3));
    zoomResetBtn.addEventListener('click', resetZoom);

    // Mouse wheel zoom inside lightbox
    const lightboxBody = document.querySelector('.lightbox-body');
    lightboxBody.addEventListener('wheel', (e) => {
      e.preventDefault();
      const factor = e.deltaY < 0 ? 1.15 : 0.85;
      applyZoom(currentScale * factor);
    });

    // Drag / Pan inside lightbox
    lightboxBody.addEventListener('mousedown', (e) => {
      if (currentScale > 1) {
        isPanning = true;
        startX = e.clientX - translateX;
        startY = e.clientY - translateY;
      }
    });

    window.addEventListener('mousemove', (e) => {
      if (isPanning) {
        translateX = e.clientX - startX;
        translateY = e.clientY - startY;
        updateTransform();
      }
    });

    window.addEventListener('mouseup', () => {
      isPanning = false;
    });

    // Keyboard navigation
    window.addEventListener('keydown', (e) => {
      if (lightboxModal.classList.contains('active')) {
        if (e.key === 'Escape') closeLightbox();
        return;
      }

      if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
        if (currentExpIndex > 0) selectExperiment(currentExpIndex - 1);
      } else if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
        if (currentExpIndex < experiments.length - 1) selectExperiment(currentExpIndex + 1);
      } else if (e.key.toLowerCase() === 'f') {
        if (currentMode === 'filtered') {
          modeRawBtn.click();
        } else {
          modeFilteredBtn.click();
        }
      }
    });
  }

  function selectExperiment(index) {
    loadExperiment(index);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  // Lightbox functions
  function openLightbox(src, title) {
    lightboxImg.src = src;
    lightboxTitle.textContent = title;
    resetZoom();
    lightboxModal.classList.add('active');
    document.body.style.overflow = 'hidden';
  }

  function closeLightbox() {
    lightboxModal.classList.remove('active');
    document.body.style.overflow = '';
  }

  function applyZoom(newScale) {
    currentScale = Math.min(Math.max(newScale, 0.5), 5);
    updateTransform();
  }

  function resetZoom() {
    currentScale = 1;
    translateX = 0;
    translateY = 0;
    updateTransform();
  }

  function updateTransform() {
    lightboxImg.style.transform = `translate(${translateX}px, ${translateY}px) scale(${currentScale})`;
  }

  // Start app on DOMContentLoaded
  document.addEventListener('DOMContentLoaded', init);
})();
