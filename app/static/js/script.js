/**
 * InformaticsAI — Main Script
 * Handles: predict form AJAX, dark mode sync, misc UI
 */

// ── Prevent double-submit ──────────────────────────────────────────────
// (The main predict logic lives inline in predict.html to access Chart.js
//  after it is defined in the block. This file is for global utilities.)

document.addEventListener('DOMContentLoaded', () => {

    // ── Auto-dismiss field error on re-focus ──
    document.querySelectorAll('.input-ai').forEach(el => {
        el.addEventListener('focus', () => {
            el.classList.remove('is-error');
            const errId = 'err-' + el.name;
            const errEl = document.getElementById(errId);
            if (errEl) { errEl.textContent = ''; errEl.classList.remove('show'); }
        });
    });

    // ── Topbar active link highlight (fallback for SPA-like feel) ──
    const path  = window.location.pathname;
    document.querySelectorAll('.nav-item, .drawer-item').forEach(link => {
        const href = link.getAttribute('href');
        if (href && href !== '/' && path.startsWith(href)) {
            link.classList.add('active');
        }
    });

    // ── Re-draw charts on theme change to update colors ──
    const observer = new MutationObserver(() => {
        // Charts on compare page are already rendered; page reload on theme
        // change would be expensive. Instead we do nothing — Chart.js colors
        // are set at render-time. Users can reload if needed.
    });
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });

});
