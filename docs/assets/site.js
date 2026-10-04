// Copy BibTeX
const copyButton = document.querySelector("[data-copy-bib]");
const bibtex = document.querySelector("#bibtex");

if (copyButton && bibtex) {
  copyButton.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(bibtex.textContent.trim());
      const original = copyButton.textContent;
      copyButton.textContent = "Copied";
      window.setTimeout(() => {
        copyButton.textContent = original;
      }, 1400);
    } catch {
      copyButton.textContent = "Select text";
    }
  });
}

// FID / CMMD toggle; bold the best (lowest) value per column within each group
const fidTable = document.querySelector("#fid-table");
const tabs = document.querySelectorAll(".seg [data-metric]");

function renderMetric(metric) {
  if (!fidTable) return;
  const rows = [...fidTable.tBodies[0].rows];
  const groups = [[]];
  rows.forEach((row) => {
    if (row.classList.contains("group")) groups.push([]);
    else groups[groups.length - 1].push(row);
  });
  groups.forEach((group) => {
    if (!group.length) return;
    const ncol = group[0].cells.length;
    for (let c = 1; c < ncol; c++) {
      const cells = group.map((r) => r.cells[c]);
      let best = null;
      cells.forEach((cell) => {
        cell.textContent = cell.dataset[metric];
        cell.classList.remove("best");
        if (!best || parseFloat(cell.dataset[metric]) < parseFloat(best.dataset[metric])) best = cell;
      });
      best.classList.add("best");
    }
  });
  tabs.forEach((t) => t.setAttribute("aria-selected", String(t.dataset.metric === metric)));
}

tabs.forEach((t) => t.addEventListener("click", () => renderMetric(t.dataset.metric)));
renderMetric("fid");
