// static/app.js
// ------------------------------------------------------------
// Minimal JS:
// - AJAX form submit (fetch JSON)
// - toast notifications
// - render prediction results in a user-friendly way
// - assistant can explain LAST result automatically (no manual JSON copy)
// ------------------------------------------------------------

(function () {
  "use strict";

  // ---------- Helpers ----------
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  function ensureToastWrap() {
    let wrap = $(".toast-wrap");
    if (!wrap) {
      wrap = document.createElement("div");
      wrap.className = "toast-wrap";
      document.body.appendChild(wrap);
    }
    return wrap;
  }

  function toast(type, title, msg, timeoutMs = 3500) {
    const wrap = ensureToastWrap();
    const el = document.createElement("div");
    el.className = `toast ${type || ""}`.trim();
    el.innerHTML = `
      <div class="toast-title">${escapeHtml(title || "Сообщение")}</div>
      <div class="toast-msg">${escapeHtml(msg || "")}</div>
    `;
    wrap.appendChild(el);
    window.setTimeout(() => {
      el.style.opacity = "0";
      el.style.transform = "translateY(6px)";
      el.style.transition = "opacity .18s ease, transform .18s ease";
      window.setTimeout(() => el.remove(), 220);
    }, timeoutMs);
  }

  function escapeHtml(s) {
    return String(s ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function toNumberOrKeep(x) {
    if (x === null || x === undefined) return x;
    if (typeof x !== "string") return x;

    const trimmed = x.trim();
    if (trimmed === "") return trimmed;
    if (/^-?\d+(\.\d+)?$/.test(trimmed)) return Number(trimmed);
    return x;
  }

  function serializeForm(form) {
    const out = {};
    const elements = $$("input[name], select[name], textarea[name]", form);

    elements.forEach((el) => {
      const name = el.name;
      if (!name) return;

      if (el.type === "checkbox") {
        out[name] = !!el.checked;
        return;
      }

      const dtype = el.dataset.type;
      let value = el.value;

      if (dtype === "number") {
        const v = value.trim();
        out[name] = v === "" ? null : Number(v);
        return;
      }

      out[name] = toNumberOrKeep(value);
    });

    return out;
  }

  async function apiFetchJson(url, payload) {
	  const res = await fetch(url, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify(payload ?? {}),
	  });

	  let data = null;
	  const contentType = res.headers.get("content-type") || "";
	  if (contentType.includes("application/json")) {
		data = await res.json();
	  } else {
		const text = await res.text();
		data = { raw: text };
	  }

	  // ✅ ВАЖНО: если сервер вернул {ok:false,...} — это тоже ошибка приложения
	  const appError =
		data && typeof data === "object" && Object.prototype.hasOwnProperty.call(data, "ok") && data.ok === false;

	  if (!res.ok || appError) {
		const msg =
		  (data && (data.detail || data.error || data.message)) ||
		  `HTTP ${res.status} ${res.statusText}`;
		// полезно для диагностики
		console.log("API error payload:", data);
		throw new Error(msg);
	  }

	  return data;
	}

  function setLoading(btn, isLoading) {
    if (!btn) return;
    btn.disabled = !!isLoading;
    const orig = btn.dataset.origText || btn.textContent;
    if (!btn.dataset.origText) btn.dataset.origText = orig;
    btn.textContent = isLoading ? "Обработка..." : btn.dataset.origText;
  }

  function safeJsonParse(text) {
    try {
      return { ok: true, value: JSON.parse(text) };
    } catch {
      return { ok: false, value: null };
    }
  }

  // ---------- Rendering ----------
  function renderProgressBar(p) {
    const pct = Math.max(0, Math.min(100, Math.round((p || 0) * 100)));
    return `
      <div class="progress" aria-label="confidence">
        <div style="width:${pct}%"></div>
      </div>
      <div class="small mono" style="margin-top:8px;">Уверенность: ${pct}%</div>
    `;
  }

  function renderKeyValue(obj) {
    const pretty = JSON.stringify(obj, null, 2);
    return `<pre>${escapeHtml(pretty)}</pre>`;
  }

  function renderErrorBlock(msg) {
    return `
      <div class="result">
        <div class="badge badge-danger">Ошибка</div>
        <div class="small" style="margin-top:10px;">${escapeHtml(msg || "unknown")}</div>
      </div>
    `;
  }

  function renderSubmissionResult(data) {
    if (data && data.ok === false) {
      return renderErrorBlock(data.error || data.message || "Не удалось посчитать.");
    }

    const proba = Number(data.proba ?? data.probability ?? 0);
    const thr = Number(data.threshold ?? data.thr ?? 0.5);
    const label = Number(data.label ?? (proba >= thr ? 1 : 0));
    const labelName =
      data.label_name ||
      (label === 1 ? "Скорее примут" : "Скорее не примут");

    const badgeClass = label === 1 ? "badge-success" : "badge-warning";

    let factorsHtml = "";
    const top = data.explanation?.top_factors;
    if (Array.isArray(top) && top.length) {
      const rows = top.slice(0, 8).map((f) => {
        const name = escapeHtml(f.name ?? "");
        const val = escapeHtml(String(f.value ?? ""));
        const dir = escapeHtml(f.direction ?? "");
        const w = escapeHtml(String(f.weight ?? ""));
        return `<tr><td>${name}</td><td class="mono">${val}</td><td>${dir}</td><td class="mono">${w}</td></tr>`;
      }).join("");
      factorsHtml = `
        <div class="table-wrap" style="margin-top:12px;">
          <table>
            <thead><tr><th>Фактор</th><th>Значение</th><th>Влияние</th><th>Вес</th></tr></thead>
            <tbody>${rows}</tbody>
          </table>
        </div>
      `;
    }

    const notes = data.explanation?.notes || data.notes || "";

    return `
      <div class="result">
        <div class="inline" style="justify-content:space-between;">
          <div>
            <div class="badge ${badgeClass}">${escapeHtml(labelName)}</div>
            <div class="small" style="margin-top:8px;">
              Порог: <span class="mono">${Number.isFinite(thr) ? thr.toFixed(2) : "—"}</span>
            </div>
          </div>
          <div style="min-width:220px; max-width:320px; width:100%;">
            ${renderProgressBar(proba)}
          </div>
        </div>

        ${notes ? `<div class="small" style="margin-top:12px;">${escapeHtml(notes)}</div>` : ""}

        ${factorsHtml || ""}

        <div class="small" style="margin-top:12px;">
          Подсказка: ориентир надёжнее, когда есть реальные оценки/рецензии. Если данных мало — воспринимайте как подсказку.
        </div>
      </div>
    `;
  }

  function renderFocusResult(data) {
    if (data && data.ok === false) {
      return renderErrorBlock(data.error || data.message || "Не удалось проверить.");
    }

    const isFocus = !!(data.is_focus ?? data.focus ?? false);
    const reason = data.reason || data.comment || "";
    const badgeClass = isFocus ? "badge-success" : "badge-warning";
    const title = isFocus ? "Относится к фокусу МУИВ" : "Не относится к фокусу МУИВ";

    let rulesHtml = "";
    if (Array.isArray(data.matched_rules) && data.matched_rules.length) {
      const items = data.matched_rules.slice(0, 8).map((r) => `<li>${escapeHtml(String(r))}</li>`).join("");
      rulesHtml = `<ul class="small" style="margin:10px 0 0 18px;">${items}</ul>`;
    }

    return `
      <div class="result">
        <div class="badge ${badgeClass}">${escapeHtml(title)}</div>
        ${reason ? `<div class="small" style="margin-top:10px;">${escapeHtml(reason)}</div>` : ""}
        ${rulesHtml}
        <div class="small" style="margin-top:12px;">
          Результат объясняется правилами/признаками — без “чёрного ящика”.
        </div>
      </div>
    `;
  }

  function renderEventForecastResult(data) {
    if (data && data.ok === false) {
      return renderErrorBlock(data.error || data.message || "Не удалось посчитать прогноз.");
    }

    const pred = data.reg_count_pred ?? data.pred ?? null;
    const baseline = data.baseline ?? null;

    const kpi = `
      <div class="kpi">
        <div class="kpi-item">
          <div class="kpi-label">Оценка регистраций</div>
          <div class="kpi-value">${pred === null ? "—" : escapeHtml(String(Math.round(pred)))}</div>
        </div>
        <div class="kpi-item">
          <div class="kpi-label">Бейзлайн</div>
          <div class="kpi-value">${baseline === null ? "—" : escapeHtml(String(Math.round(baseline)))}</div>
        </div>
        <div class="kpi-item">
          <div class="kpi-label">Комментарий</div>
          <div class="kpi-value" style="font-size:14px; font-weight:650; color: var(--muted);">
            ${escapeHtml(data.note || "Оценка для планирования")}
          </div>
        </div>
      </div>
    `;

    return `
      <div class="result">
        ${kpi}
        <div class="small" style="margin-top:12px;">
          Это ориентир для планирования ресурсов, а не гарантированное значение.
        </div>
      </div>
    `;
  }

  function renderGenericJson(data) {
    return `
      <div class="result">
        <div class="small" style="margin-bottom:10px;">Ответ сервера (JSON):</div>
        ${renderKeyValue(data)}
      </div>
    `;
  }

  function renderByKind(form, data) {
    const kind = (form.dataset.kind || "").toLowerCase();
    const id = (form.id || "").toLowerCase();

    if (kind === "submission" || id.includes("submission")) return renderSubmissionResult(data);
    if (kind === "focus" || id.includes("focus")) return renderFocusResult(data);
    if (kind === "event" || id.includes("event")) return renderEventForecastResult(data);

    return renderGenericJson(data);
  }

  // ---------- Wiring ----------
  async function handleApiFormSubmit(form) {
    const apiUrl = form.dataset.api;
    const targetSel = form.dataset.target;
    const target = targetSel ? $(targetSel) : null;
    const submitBtn = $("button[type='submit']", form);

    if (!apiUrl) {
      toast("error", "Форма не настроена", "Не задан data-api у формы.");
      return;
    }

    const payload = serializeForm(form);

    try {
      setLoading(submitBtn, true);
      if (target) target.innerHTML = `<div class="small">Обработка запроса…</div>`;

      const data = await apiFetchJson(apiUrl, payload);

      if (target) {
        target.innerHTML = renderByKind(form, data);
        // сохраняем “последний JSON” для ассистента
        target.dataset.lastJson = JSON.stringify(data);
      }

      if (data && data.ok === false) {
        toast("warn", "Готово, но есть замечание", data.error || data.message || "Ответ ok=false");
      } else {
        toast("success", "Готово", "Результат получен.");
      }
    } catch (err) {
      const msg = err?.message || String(err);
      if (target) target.innerHTML = renderErrorBlock(msg);
      toast("error", "Ошибка", msg, 5200);
    } finally {
      setLoading(submitBtn, false);
    }
  }

  function markActiveNav() {
    const path = window.location.pathname || "/";
    $$(".nav a").forEach((a) => {
      const href = a.getAttribute("href") || "";
      const isActive = href === path || (href !== "/" && path.startsWith(href));
      if (isActive) {
        a.classList.add("is-active");
        a.classList.add("active"); // на всякий случай
      }
    });
  }

  function initThemeToggle() {
    const btn = $("[data-action='toggle-theme']");
    if (!btn) return;

    const key = "ui_theme";
    const saved = localStorage.getItem(key);
    if (saved) document.documentElement.setAttribute("data-theme", saved);

    btn.addEventListener("click", () => {
      const cur = document.documentElement.getAttribute("data-theme") || "light";
      const next = cur === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      localStorage.setItem(key, next);
      toast("success", "Тема", next === "dark" ? "Тёмная тема включена" : "Светлая тема включена");
    });
  }

  function initApiForms() {
    $$("form[data-api]").forEach((form) => {
      form.addEventListener("submit", (e) => {
        e.preventDefault();
        handleApiFormSubmit(form);
      });
    });
  }

  function fillDemoForForm(form) {
    const kind = (form.dataset.kind || "").toLowerCase();

    // Демо-наборы по ключам name=""
    const demo = {
      submission: {
        n_reviews: 2,
        score_mean: 7.5,
        conf_mean: 4.0,
        score_std: 0.7,
        role: "master",
        university: "МГУ",
        country: "Россия",
        city: "Москва, Россия",
      },
      focus: {
        title_clean: "Виттевские чтения - 2026",
        city: "Москва, Россия",
        venue: "МУ имени С. Ю. Витте",
      },
      event: {
        title_clean: "Ломоносовские чтения-2026",
        city: "Москва, Россия",
        venue: "МГУ",
        is_online_flag: 0,
        cost_is_free: 1,
        cost_mentions_fee: 0,
        event_month: 4,
        event_dow: 2,
        description_len: 900,
        event_duration_days: 2,
        tracks_count_clean: 3,
        n_links: 5,
        n_calendar: 1,
        n_external: 2,
        n_pages: 1,
        deadline_days_before_event: 30,
      }
    };

    const pack = demo[kind];
    if (!pack) return false;

    Object.entries(pack).forEach(([k, v]) => {
      const el = $(`[name="${CSS.escape(k)}"]`, form);
      if (!el) return;
      el.value = String(v);
    });
    return true;
  }

  function initDemoButtons() {
    $$(`[data-action="fill-demo"], [data-action="fill-demo-focus"]`).forEach((btn) => {
      btn.addEventListener("click", () => {
        const sel = btn.dataset.targetForm;
        const form = sel ? $(sel) : null;
        if (!form) {
          toast("error", "Пример", "Не найден form по data-target-form");
          return;
        }
        const ok = fillDemoForForm(form);
        toast(ok ? "success" : "warn", "Пример", ok ? "Поля заполнены примером." : "Для этой формы нет демо-набора.");
      });
    });
  }

  function extractContextFromSource(el) {
    if (!el) return { raw: "" };

    // 1) если мы ранее сохраняли JSON
    if (el.dataset && el.dataset.lastJson) {
      const parsed = safeJsonParse(el.dataset.lastJson);
      if (parsed.ok) return parsed.value;
      return { raw: el.dataset.lastJson };
    }

    // 2) попробуем найти <pre> и распарсить
    const pre = $("pre", el);
    if (pre) {
      const t = pre.textContent || "";
      const parsed = safeJsonParse(t);
      if (parsed.ok) return parsed.value;
      return { raw: t };
    }

    // 3) fallback — просто текст
    return { raw: (el.innerText || el.textContent || "").trim() };
  }

  function initAssistantButtons() {
    $$(`[data-action="assistant"]`).forEach((btn) => {
      btn.addEventListener("click", async () => {
        const apiUrl = btn.dataset.api || "/api/assistant/report";
        const outSel = btn.dataset.target || "#assistant_result";
        const outEl = $(outSel);

        // контекст либо из textarea (data-context-target), либо из блока результата (data-context-source)
        let ctx = null;

        const ctxTargetSel = btn.dataset.contextTarget;
        const ctxSourceSel = btn.dataset.contextSource;

        if (ctxTargetSel) {
          const ctxEl = $(ctxTargetSel);
          const txt = ctxEl ? (ctxEl.value || "") : "";
          const parsed = safeJsonParse(txt);
          ctx = parsed.ok ? parsed.value : { raw: txt };
        } else if (ctxSourceSel) {
          const srcEl = $(ctxSourceSel);
          ctx = extractContextFromSource(srcEl);
        } else {
          // если ничего не задано — попробуем рядом найти ближайший .result/.card
          ctx = { raw: "no_context" };
        }

        const payload = { context: ctx };

        try {
          btn.disabled = true;
          if (!btn.dataset.origText) btn.dataset.origText = btn.textContent;
          btn.textContent = "Генерирую…";
          if (outEl) outEl.innerHTML = `<div class="small">Пишу пояснение…</div>`;

          const data = await apiFetchJson(apiUrl, payload);

          const text = data.report || data.text || data.message || JSON.stringify(data);
          if (outEl) {
            outEl.innerHTML = `
              <div class="result">
                <div class="badge">Пояснение</div>
                <div style="margin-top:10px; line-height:1.55; color: var(--text);">
                  ${escapeHtml(text).replaceAll("\n", "<br>")}
                </div>
              </div>
            `;
          }
          toast("success", "Ассистент", "Пояснение сформировано.");
        } catch (err) {
          toast("error", "Ассистент", err?.message || String(err), 5200);
          if (outEl) outEl.innerHTML = renderErrorBlock(err?.message || String(err));
        } finally {
          btn.disabled = false;
          btn.textContent = btn.dataset.origText || "Пояснить";
        }
      });
    });
  }

  // ---------- Boot ----------
  document.addEventListener("DOMContentLoaded", () => {
    markActiveNav();
    initThemeToggle();
    initApiForms();
    initDemoButtons();
    initAssistantButtons();
  });
})();
