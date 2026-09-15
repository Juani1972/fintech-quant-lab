# Capturas de pantalla

Guarda aquí las capturas (o maquetas) que aparecen en el README principal.

## Convención de nombres

| Archivo | Página |
|---|---|
| `01_dashboard.png` | Portada |
| `02_garch.png` | Página GARCH |
| `03_cointegration.png` | Página Cointegración |
| `04_famafrench.png` | Página Fama-French |
| `05_risk.png` | Página Riesgo |
| `06_backtest.png` | Página Backtest |
| `07_walkforward.png` | Página Walk-Forward |
| `08_optimization.png` | Página Optimización |
| `09_robustness.png` | Página Robustez |
| `10_historico.png` | Página Histórico |

## Opción A: generarlas con matplotlib (sin navegador)

`generate_mockups.py`, en esta misma carpeta, genera las 10 imágenes a
partir de cálculos reales de `app.core` sobre un par de tickers
sintético (no descarga datos de Yahoo Finance). Útil si no tienes forma
de tomar capturas reales de la app en ejecución:

```bash
pip install matplotlib
python docs/screenshots/generate_mockups.py
```

Son maquetas fieles al cálculo (GARCH, cointegración, backtest, etc. son
reales), pero no capturas literales de la interfaz Streamlit — no
reflejan el layout exacto de cada página.

## Opción B: capturas reales de la app

1. Ejecuta la app: `streamlit run app/main.py`.
2. Navega a la página correspondiente.
3. Ejecuta un análisis con datos de ejemplo (ej: `KO, PEP`).
4. Captura la pantalla completa (Cmd+Shift+4 en macOS, Win+Shift+S en Windows).
5. Guarda el PNG en esta carpeta con el nombre indicado en la tabla.

## Tamaño recomendado

- Ancho: 1600-2000 px.
- Formato: PNG.
- Peso: < 500 KB por imagen (comprime con [TinyPNG](https://tinypng.com/)).
