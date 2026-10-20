# DXA QC — веб-интерфейс

React + Vite + TypeScript. Сборка кладётся в `../src/dxaqc/webui_dist` и раздаётся `dxaqc serve`; Node на время работы не нужен.

```sh
bun install
bun run build          # → src/dxaqc/webui_dist
bun run dev            # разработка; API проксируется на dxaqc serve --port 8766
```

Визуальная система — `../DESIGN.md`.
