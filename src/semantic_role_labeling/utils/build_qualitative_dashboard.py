import argparse
import json
from pathlib import Path


DEFAULT_INPUT = (
    "artifacts/xlm-roberta-large/baseline/seed120/"
    "confusion_analysis/qualitative_examples.json"
)
DEFAULT_OUTPUT = "artifacts/qualitative_dashboard/index.html"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a self-contained dashboard for qualitative SRL analysis."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_instances(path):
    with path.open("r", encoding="utf-8") as input_file:
        qualitative_data = json.load(input_file)

    categories = ("numbered_successes", "modifier_successes", "total_failures")
    missing_categories = [
        category for category in categories if category not in qualitative_data
    ]
    if missing_categories:
        raise ValueError(
            f"Missing qualitative categories: {', '.join(missing_categories)}"
        )

    instances = [
        instance
        for category in categories
        for instance in qualitative_data[category]
    ]
    return {"criteria": qualitative_data.get("criteria", {}), "instances": instances}


HTML_TEMPLATE = r"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SRL · Análise qualitativa</title>
  <style>
    :root {
      --ink: #18201d;
      --muted: #66706b;
      --paper: #f7f8f5;
      --surface: #ffffff;
      --line: #d9ded9;
      --strong-line: #aab4ad;
      --accent: #006b5f;
      --accent-soft: #d9eee9;
      --gold: #946200;
      --gold-soft: #fff2c9;
      --error: #b83b32;
      --error-soft: #fde4df;
      --correct: #28784d;
      --pred: #315c9b;
      --pred-soft: #e4edfb;
      --shadow: 0 10px 30px rgba(24, 32, 29, 0.08);
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      background:
        linear-gradient(rgba(24, 32, 29, 0.035) 1px, transparent 1px),
        var(--paper);
      background-size: 100% 32px;
      font-family: "Trebuchet MS", "Gill Sans", sans-serif;
    }
    button, input, select { font: inherit; }
    button, select { cursor: pointer; }
    button:focus-visible, input:focus-visible, select:focus-visible {
      outline: 3px solid rgba(0, 107, 95, 0.24);
      outline-offset: 2px;
    }

    .topbar {
      min-height: 72px;
      padding: 14px 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 24px;
      color: #fff;
      background: #18352f;
      border-bottom: 4px solid #d5a124;
    }
    .brand h1 {
      margin: 0;
      font: 700 24px/1.05 Georgia, serif;
      letter-spacing: 0;
    }
    .brand p { margin: 5px 0 0; color: #c9d8d2; font-size: 13px; }
    .dataset-stats { display: flex; gap: 24px; }
    .stat { min-width: 86px; text-align: right; }
    .stat strong { display: block; font: 700 21px/1 Georgia, serif; }
    .stat span { display: block; margin-top: 5px; color: #c9d8d2; font-size: 11px; text-transform: uppercase; }

    .filters {
      padding: 12px 24px;
      display: grid;
      grid-template-columns: minmax(240px, 1fr) repeat(2, minmax(150px, 210px)) auto;
      gap: 10px;
      align-items: center;
      background: rgba(255, 255, 255, 0.96);
      border-bottom: 1px solid var(--line);
      position: sticky;
      top: 0;
      z-index: 5;
    }
    .control {
      width: 100%;
      height: 40px;
      padding: 0 12px;
      color: var(--ink);
      background: #fff;
      border: 1px solid var(--strong-line);
      border-radius: 4px;
    }
    .clear-button {
      height: 40px;
      padding: 0 16px;
      color: var(--accent);
      background: transparent;
      border: 1px solid var(--accent);
      border-radius: 4px;
      font-weight: 700;
    }

    .workspace {
      display: grid;
      grid-template-columns: 330px minmax(0, 1fr);
      min-height: calc(100vh - 137px);
    }
    .result-pane {
      background: rgba(246, 248, 245, 0.96);
      border-right: 1px solid var(--line);
      min-width: 0;
    }
    .result-header {
      height: 46px;
      padding: 0 16px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      color: var(--muted);
      border-bottom: 1px solid var(--line);
      font-size: 13px;
    }
    .result-list {
      height: calc(100vh - 183px);
      overflow-y: auto;
    }
    .result-item {
      width: 100%;
      min-height: 94px;
      padding: 13px 16px;
      display: block;
      text-align: left;
      color: var(--ink);
      background: transparent;
      border: 0;
      border-bottom: 1px solid var(--line);
      border-left: 4px solid transparent;
    }
    .result-item:hover { background: #fff; }
    .result-item.active { background: #fff; border-left-color: var(--accent); }
    .item-top { display: flex; justify-content: space-between; gap: 8px; font-size: 11px; color: var(--muted); }
    .item-predicate { color: var(--accent); font-weight: 800; }
    .item-sentence {
      margin-top: 8px;
      display: -webkit-box;
      overflow: hidden;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
      font: 15px/1.35 Georgia, serif;
    }
    .item-foot { margin-top: 8px; display: flex; gap: 8px; font-size: 11px; }
    .error-count { color: var(--error); font-weight: 700; }
    .correct-count { color: var(--correct); font-weight: 700; }

    .detail-pane { min-width: 0; padding: 26px clamp(20px, 4vw, 56px) 48px; }
    .detail-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 24px; }
    .eyebrow { margin: 0 0 8px; color: var(--accent); font-size: 12px; font-weight: 800; text-transform: uppercase; }
    .sentence {
      max-width: 1000px;
      margin: 0;
      font: 600 clamp(22px, 2.3vw, 34px)/1.32 Georgia, serif;
      letter-spacing: 0;
    }
    .sentence mark { padding: 1px 4px; color: inherit; background: var(--gold-soft); border-bottom: 3px solid #d5a124; }
    .line-number { color: var(--muted); font-size: 12px; white-space: nowrap; }

    .metric-strip {
      margin: 24px 0 26px;
      display: grid;
      grid-template-columns: repeat(4, minmax(100px, 1fr));
      border-block: 1px solid var(--line);
    }
    .metric { padding: 14px 16px; border-right: 1px solid var(--line); }
    .metric:last-child { border-right: 0; }
    .metric strong { display: block; font: 700 23px/1 Georgia, serif; }
    .metric span { display: block; margin-top: 6px; color: var(--muted); font-size: 11px; text-transform: uppercase; }

    .section-title { margin: 28px 0 12px; font: 700 17px/1.2 Georgia, serif; }
    .legend { display: flex; flex-wrap: wrap; gap: 12px 20px; margin-bottom: 14px; color: var(--muted); font-size: 12px; }
    .legend span { display: inline-flex; align-items: center; gap: 6px; }
    .legend i { width: 10px; height: 10px; display: inline-block; border-radius: 2px; }

    .token-flow { display: flex; flex-wrap: wrap; align-items: flex-start; gap: 8px; }
    .token {
      min-width: 54px;
      padding: 7px 8px 6px;
      background: var(--surface);
      border: 1px solid var(--line);
      border-bottom: 3px solid var(--strong-line);
      border-radius: 4px;
      text-align: center;
      box-shadow: 0 2px 4px rgba(24, 32, 29, 0.04);
    }
    .token.argument { border-bottom-color: var(--gold); background: #fffdf6; }
    .token.predicate { border-bottom-color: var(--accent); background: var(--accent-soft); }
    .token.error { border-color: #e6aaa4; border-bottom-color: var(--error); background: var(--error-soft); }
    .token.ignored { display: none; }
    .token-text { display: block; min-height: 20px; font: 700 15px/1.2 Georgia, serif; }
    .token-role { display: block; margin-top: 5px; color: var(--muted); font-size: 9px; font-weight: 800; }

    .error-table { width: 100%; border-collapse: collapse; background: #fff; box-shadow: var(--shadow); }
    .error-table th, .error-table td { padding: 11px 13px; text-align: left; border-bottom: 1px solid var(--line); }
    .error-table th { color: var(--muted); background: #f1f3f0; font-size: 11px; text-transform: uppercase; }
    .error-table td { font-size: 13px; }
    .role-gold { color: var(--gold); font-weight: 800; }
    .role-pred { color: var(--pred); font-weight: 800; }
    .empty { padding: 40px 16px; color: var(--muted); text-align: center; font: 18px/1.4 Georgia, serif; }

    .pager { display: flex; gap: 6px; }
    .pager button { width: 28px; height: 28px; padding: 0; color: var(--ink); background: #fff; border: 1px solid var(--line); border-radius: 3px; }
    .pager button:disabled { opacity: 0.35; cursor: default; }

    @media (max-width: 900px) {
      .topbar { align-items: flex-start; }
      .dataset-stats { gap: 12px; }
      .stat { min-width: 62px; }
      .filters { grid-template-columns: 1fr 1fr; }
      .workspace { grid-template-columns: 1fr; }
      .result-pane { border-right: 0; border-bottom: 1px solid var(--line); }
      .result-list { height: 260px; }
      .detail-head { display: block; }
      .line-number { margin-top: 10px; display: block; }
    }
    @media (max-width: 560px) {
      .topbar { display: block; padding: 14px 16px; }
      .dataset-stats { margin-top: 16px; justify-content: space-between; }
      .stat { text-align: left; }
      .filters { padding: 10px 12px; grid-template-columns: 1fr; position: static; }
      .metric-strip { grid-template-columns: 1fr 1fr; }
      .metric:nth-child(2) { border-right: 0; }
      .metric:nth-child(-n+2) { border-bottom: 1px solid var(--line); }
      .detail-pane { padding-inline: 16px; }
      .sentence { font-size: 22px; }
    }
  </style>
</head>
<body>
  <header class="topbar">
    <div class="brand">
      <h1>SRL · Análise qualitativa</h1>
      <p>Inspeção de papéis semânticos por predicado</p>
    </div>
    <div class="dataset-stats">
      <div class="stat"><strong id="totalInstances">—</strong><span>exemplos</span></div>
      <div class="stat"><strong id="totalSuccesses">—</strong><span>sucessos</span></div>
      <div class="stat"><strong id="totalFailures">—</strong><span>falhas</span></div>
    </div>
  </header>

  <section class="filters" aria-label="Filtros">
    <input id="search" class="control" type="search" placeholder="Buscar sentença, ID ou predicado">
    <select id="categoryFilter" class="control" aria-label="Filtrar por categoria">
      <option value="all">Todas as categorias</option>
      <option value="numbered_success">Sucessos · argumentos numerados</option>
      <option value="modifier_success">Sucessos · modificadores</option>
      <option value="total_failure">Falhas totais</option>
    </select>
    <select id="roleFilter" class="control" aria-label="Filtrar por papel">
      <option value="all">Todos os papéis</option>
    </select>
    <button id="clearFilters" class="clear-button" type="button">Limpar filtros</button>
  </section>

  <main class="workspace">
    <aside class="result-pane">
      <div class="result-header">
        <span id="resultCount">0 resultados</span>
      </div>
      <div id="resultList" class="result-list"></div>
    </aside>

    <article id="detail" class="detail-pane"></article>
  </main>

  <script id="dataset" type="application/json">__DATA__</script>
  <script>
    const payload = JSON.parse(document.getElementById('dataset').textContent);
    const categoryLabels = {
      numbered_success: 'Sucesso · argumentos numerados',
      modifier_success: 'Sucesso · modificadores',
      total_failure: 'Falha total'
    };
    let filtered = [];
    let selectedIndex = null;

    const elements = {
      search: document.getElementById('search'),
      category: document.getElementById('categoryFilter'),
      role: document.getElementById('roleFilter'),
      list: document.getElementById('resultList'),
      detail: document.getElementById('detail'),
      count: document.getElementById('resultCount')
    };

    function escapeHtml(value) {
      return String(value).replace(/[&<>'"]/g, character => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'
      })[character]);
    }

    function markPredicate(sentence, predicate) {
      const safeSentence = escapeHtml(sentence);
      const safePredicate = escapeHtml(predicate);
      return safeSentence.replace(safePredicate, `<mark>${safePredicate}</mark>`);
    }

    function percent(value, digits = 1) {
      if (value === null || value === undefined) return '—';
      return new Intl.NumberFormat('pt-BR', {
        style: 'percent', minimumFractionDigits: digits, maximumFractionDigits: digits
      }).format(value);
    }

    const data = payload.instances.map((instance, index) => ({
      ...instance,
      index,
      roles: [...new Set(instance.arguments.flatMap(argument =>
        [argument.true_role, argument.predicted_role].filter(role => role !== 'O')
      ))]
    }));
    const allRoles = [...new Set(data.flatMap(instance => instance.roles))].sort((a, b) => a.localeCompare(b));

    function populateSummary() {
      document.getElementById('totalInstances').textContent = data.length.toLocaleString('pt-BR');
      document.getElementById('totalSuccesses').textContent = data.filter(item => item.category !== 'total_failure').length.toLocaleString('pt-BR');
      document.getElementById('totalFailures').textContent = data.filter(item => item.category === 'total_failure').length.toLocaleString('pt-BR');
      allRoles.forEach(role => {
        const option = document.createElement('option');
        option.value = role;
        option.textContent = role;
        elements.role.appendChild(option);
      });
    }

    function applyFilters() {
      const query = elements.search.value.trim().toLocaleLowerCase('pt-BR');
      const category = elements.category.value;
      const role = elements.role.value;
      filtered = data.filter(instance => {
        const searchable = [
          instance.sentence, instance.sentence_id, instance.predicate.token,
          instance.predicate.lemma, instance.predicate.frame
        ];
        const matchesQuery = !query || searchable.some(value =>
          String(value || '').toLocaleLowerCase('pt-BR').includes(query)
        );
        const matchesCategory = category === 'all' || instance.category === category;
        const matchesRole = role === 'all' || instance.roles.includes(role);
        return matchesQuery && matchesCategory && matchesRole;
      });
      if (!filtered.some(instance => instance.index === selectedIndex)) {
        selectedIndex = filtered[0]?.index ?? null;
      }
      renderList();
      renderDetail();
    }

    function renderList() {
      elements.count.textContent = `${filtered.length.toLocaleString('pt-BR')} exemplos`;
      elements.list.innerHTML = filtered.length ? filtered.map(instance => `
        <button class="result-item ${instance.index === selectedIndex ? 'active' : ''}" data-index="${instance.index}" type="button">
          <span class="item-top"><span>${escapeHtml(instance.sentence_id)}</span><span class="item-predicate">${escapeHtml(instance.predicate.token)}</span></span>
          <span class="item-sentence">${escapeHtml(instance.sentence)}</span>
          <span class="item-foot">
            <span class="${instance.category === 'total_failure' ? 'error-count' : 'correct-count'}">${categoryLabels[instance.category]}</span>
            <span>${instance.arguments.length} argumentos</span>
          </span>
        </button>
      `).join('') : '<div class="empty">Nenhum exemplo encontrado.</div>';
    }

    function renderDetail() {
      const instance = data.find(item => item.index === selectedIndex);
      if (!instance) {
        elements.detail.innerHTML = '<div class="empty">Selecione um exemplo para inspecionar.</div>';
        return;
      }
      const metrics = instance.instance_metrics;
      const argumentTokens = instance.arguments.map(argument => {
        const roleText = argument.correct
          ? argument.true_role
          : `${argument.true_role} → ${argument.predicted_role}`;
        return `<span class="token argument ${argument.correct ? '' : 'error'}" title="Confiança: ${percent(argument.confidence)}">
          <span class="token-text">${escapeHtml(argument.token)}</span>
          <span class="token-role">${escapeHtml(roleText)} · ${percent(argument.confidence)}</span>
        </span>`;
      }).join('');
      const argumentRows = instance.arguments.map(argument => `
        <tr>
          <td><strong>${escapeHtml(argument.token)}</strong><br><small>${escapeHtml(argument.subtoken)}</small></td>
          <td class="role-gold">${escapeHtml(argument.true_role)}</td>
          <td class="role-pred">${escapeHtml(argument.predicted_role)}</td>
          <td>${percent(argument.confidence)}</td>
          <td class="${argument.correct ? 'correct-count' : 'error-count'}">${argument.correct ? 'Correto' : 'Divergência'}</td>
        </tr>
      `).join('');
      elements.detail.innerHTML = `
        <header class="detail-head">
          <div>
            <p class="eyebrow">${categoryLabels[instance.category]}</p>
            <h2 class="sentence">${markPredicate(instance.sentence, instance.predicate.token)}</h2>
          </div>
          <span class="line-number">${escapeHtml(instance.sentence_id)}</span>
        </header>
        <section class="metric-strip" aria-label="Resumo da instância">
          <div class="metric"><strong>${percent(metrics.identification_precision)}</strong><span>precisão identificação</span></div>
          <div class="metric"><strong>${percent(metrics.identification_recall)}</strong><span>recall identificação</span></div>
          <div class="metric"><strong>${percent(metrics.identification_f1)}</strong><span>F1 identificação</span></div>
          <div class="metric"><strong>${percent(metrics.mean_confidence_errors ?? metrics.mean_confidence_correct)}</strong><span>confiança média</span></div>
        </section>
        <div class="legend">
          <span><strong>Predicado:</strong> ${escapeHtml(instance.predicate.token)}</span>
          <span><strong>Lema:</strong> ${escapeHtml(instance.predicate.lemma || '—')}</span>
          <span><strong>Frame:</strong> ${escapeHtml(instance.predicate.frame || '—')}</span>
          <span><strong>Índice:</strong> ${instance.predicate.token_index}</span>
        </div>
        <h3 class="section-title">Argumentos relevantes</h3>
        <div class="token-flow">${argumentTokens}</div>
        <h3 class="section-title">Comparação gold × predição</h3>
        <table class="error-table">
          <thead><tr><th>Token / subtoken</th><th>Gold</th><th>Predito</th><th>Confiança</th><th>Resultado</th></tr></thead>
          <tbody>${argumentRows}</tbody>
        </table>
      `;
    }

    elements.list.addEventListener('click', event => {
      const button = event.target.closest('[data-index]');
      if (!button) return;
      selectedIndex = Number(button.dataset.index);
      renderList();
      renderDetail();
      if (window.innerWidth <= 900) elements.detail.scrollIntoView({ behavior: 'smooth' });
    });
    elements.search.addEventListener('input', () => applyFilters());
    elements.category.addEventListener('change', () => applyFilters());
    elements.role.addEventListener('change', () => applyFilters());
    document.getElementById('clearFilters').addEventListener('click', () => {
      elements.search.value = '';
      elements.category.value = 'all';
      elements.role.value = 'all';
      applyFilters();
    });

    populateSummary();
    applyFilters();
  </script>
</body>
</html>
"""


def build_dashboard(instances):
    serialized = json.dumps(instances, ensure_ascii=False, separators=(",", ":"))
    serialized = serialized.replace("</", "<\\/")
    return HTML_TEMPLATE.replace("__DATA__", serialized)


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.is_file():
      raise FileNotFoundError(f"Qualitative examples not found: {input_path}")

    qualitative_data = load_instances(input_path)
    if not qualitative_data["instances"]:
      raise ValueError(f"Qualitative examples are empty: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_dashboard(qualitative_data), encoding="utf-8")
    print(
      f"Dashboard saved to {output_path} "
      f"({len(qualitative_data['instances'])} examples)."
    )


if __name__ == "__main__":
    main()