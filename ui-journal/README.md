# Веб-интерфейс «Журнал контроля качества»

Второй интерфейс DXA QC, рядом с классическим (`../ui`). Тот же API, другой визуальный мир: главный экран
с изометрическим денситометром (рис. 1), контрольный листок пакета, результаты валидации. Описание
дизайн-системы — [DESIGN.md](DESIGN.md).

React + Vite + TypeScript. Сборка кладётся в `../src/dxaqc/webui_journal_dist`, `dxaqc serve` отдаёт её
по адресу `/journal/` (классический интерфейс остаётся на `/`). Node на время работы не нужен.

```
bun install
bun run dev            # http://localhost:5173/journal/, API проксируется на dxaqc serve --port 8766
bun run build          # → src/dxaqc/webui_journal_dist
```

Изометрический кит в `src/iso/` — из скилла Anatomy, лицензия MIT (`src/iso/LICENSE`).
