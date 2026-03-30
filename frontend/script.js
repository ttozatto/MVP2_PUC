const API_URL = "http://localhost:5000";

const selectFields = [
  "genero",
  "continente",
  "industria",
  "cargo",
  "regime_trabalho",
  "faixa_salarial",
];

async function loadOptions() {
  try {
    const res = await fetch(`${API_URL}/options`);
    const options = await res.json();

    for (const field of selectFields) {
      const select = document.getElementById(field);
      for (const value of options[field]) {
        const opt = document.createElement("option");
        opt.value = value;
        opt.textContent = value;
        select.appendChild(opt);
      }
    }

    const container = document.getElementById("problemas_saude");
    for (const issue of options.problemas_saude) {
      const lbl = document.createElement("label");
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.name = "problemas_saude";
      cb.value = issue;
      lbl.appendChild(cb);
      lbl.appendChild(document.createTextNode(issue));
      container.appendChild(lbl);
    }
  } catch {
    showError("Não foi possível carregar as opções. Verifique se o backend está rodando.");
  }
}

function setupRangeInputs() {
  const pairs = [
    { range: "equilibrio_trabalho_vida", display: "equilibrio_valor" },
    { range: "isolamento_social", display: "isolamento_valor" },
  ];
  for (const { range, display } of pairs) {
    const input = document.getElementById(range);
    const span = document.getElementById(display);
    input.addEventListener("input", () => {
      span.textContent = input.value;
    });
  }
}

function collectFormData() {
  const checked = document.querySelectorAll(
    'input[name="problemas_saude"]:checked'
  );
  return {
    idade: parseInt(document.getElementById("idade").value, 10),
    genero: document.getElementById("genero").value,
    continente: document.getElementById("continente").value,
    industria: document.getElementById("industria").value,
    cargo: document.getElementById("cargo").value,
    regime_trabalho: document.getElementById("regime_trabalho").value,
    faixa_salarial: document.getElementById("faixa_salarial").value,
    horas_trabalho_semana: parseInt(
      document.getElementById("horas_trabalho_semana").value,
      10
    ),
    equilibrio_trabalho_vida: parseInt(
      document.getElementById("equilibrio_trabalho_vida").value,
      10
    ),
    isolamento_social: parseInt(
      document.getElementById("isolamento_social").value,
      10
    ),
    problemas_saude: Array.from(checked).map((cb) => cb.value),
  };
}

function showError(msg) {
  const el = document.getElementById("erro");
  el.textContent = msg;
  el.classList.remove("hidden");
  document.getElementById("resultado").classList.add("hidden");
}

function showResult(data) {
  document.getElementById("erro").classList.add("hidden");
  const box = document.getElementById("resultado");
  box.classList.remove("hidden");

  const nivelEl = document.getElementById("resultado-nivel");
  nivelEl.textContent = `${data.label} (${data.nivel_burnout})`;
  nivelEl.className = `resultado-nivel nivel-${data.nivel_burnout}`;

  const detalhes = document.getElementById("resultado-detalhes");
  let html = `<p>Média dos modelos: <strong>${data.media_modelos}</strong></p>`;
  html += '<div style="margin-top:0.5rem">';
  for (const [modelo, pred] of Object.entries(data.previsoes_individuais)) {
    html += `<div class="modelo-row">
      <span class="modelo-nome">${modelo}</span>
      <span>${pred}</span>
    </div>`;
  }
  html += "</div>";
  detalhes.innerHTML = html;
}

async function handleSubmit(e) {
  e.preventDefault();
  const btn = document.getElementById("btn-submit");
  btn.disabled = true;
  btn.textContent = "Calculando...";

  try {
    const body = collectFormData();
    const res = await fetch(`${API_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();

    if (!res.ok) {
      showError(data.error || "Erro desconhecido na API.");
      return;
    }
    showResult(data);
  } catch {
    showError("Erro ao conectar com o servidor. Verifique se o backend está rodando.");
  } finally {
    btn.disabled = false;
    btn.textContent = "Obter Previsão";
  }
}

document.getElementById("burnout-form").addEventListener("submit", handleSubmit);
setupRangeInputs();
loadOptions();
