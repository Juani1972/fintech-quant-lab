# 🌍 Mercados y cómo buscar cualquier empresa

`app/core/data_loader.py` no impone ninguna restricción de mercado: el
campo de tickers acepta cualquier símbolo que reconozca Yahoo Finance
(la fuente de datos vía `yfinance`), sea de EE.UU. o de cualquier otra
bolsa. Este documento explica cómo encontrar el ticker correcto de una
empresa concreta y qué universos predefinidos ya trae la app.

## Universos predefinidos con empresas de mercados internacionales

Además de los universos de EE.UU. (Magnificent 7, sectores SPDR,
bancos, energía...), el selector de universos incluye 5 mercados
principales fuera de EE.UU., con 10 tickers representativos cada uno
(no el índice completo):

| Universo | Mercado | Ejemplos |
|---|---|---|
| `IBEX 35 (España)` | Bolsa de Madrid | Santander (`SAN.MC`), Inditex (`ITX.MC`), Iberdrola (`IBE.MC`) |
| `DAX 40 (Alemania)` | Frankfurt | Siemens (`SIE.DE`), Allianz (`ALV.DE`), SAP (`SAP.DE`) |
| `CAC 40 (Francia)` | Euronext París | LVMH (`MC.PA`), L'Oréal (`OR.PA`), TotalEnergies (`TTE.PA`) |
| `FTSE 100 (Reino Unido)` | Londres | HSBC (`HSBA.L`), AstraZeneca (`AZN.L`), Shell (`SHEL.L`) |
| `Nikkei 225 (Japón)` | Tokio | Toyota (`7203.T`), Sony (`6758.T`), SoftBank (`9984.T`) |

Elige el universo en la portada y el campo de tickers se rellena solo
— después puedes editarlo, quitar o añadir tickers libremente.

## Cómo encontrar el ticker de CUALQUIER empresa

Si la empresa que buscas no está en ningún universo predefinido:

1. Ve a [finance.yahoo.com](https://finance.yahoo.com) y busca el
   nombre de la empresa en el buscador.
2. Copia el símbolo que aparece junto a su nombre — ese es el ticker
   que hay que escribir en la app, exactamente igual (con su sufijo si
   lo tiene).
3. Si la empresa cotiza en varias bolsas a la vez (p. ej. muchas
   grandes tecnológicas europeas cotizan también en EE.UU. como ADR),
   Yahoo mostrará varios resultados — cualquiera funciona, pero elige
   el de la bolsa cuya moneda/horario te interese analizar.

## Sufijos de bolsa más comunes en Yahoo Finance

Si conoces el ticker "local" de una empresa (el que usan en su propio
país) pero no el formato exacto de Yahoo, añade el sufijo de su bolsa:

| Sufijo | Bolsa | País |
|---|---|---|
| *(sin sufijo)* | NYSE / NASDAQ | Estados Unidos |
| `.MC` | Bolsa de Madrid | España |
| `.L` | London Stock Exchange | Reino Unido |
| `.PA` | Euronext París | Francia |
| `.DE` | Xetra / Frankfurt | Alemania |
| `.MI` | Borsa Italiana (Milán) | Italia |
| `.AS` | Euronext Ámsterdam | Países Bajos |
| `.LS` | Euronext Lisboa | Portugal |
| `.BR` | Euronext Bruselas | Bélgica |
| `.SW` | SIX Swiss Exchange | Suiza |
| `.ST` | Nasdaq Estocolmo | Suecia |
| `.CO` | Nasdaq Copenhague | Dinamarca |
| `.OL` | Oslo Børs | Noruega |
| `.T` | Tokyo Stock Exchange | Japón |
| `.HK` | Hong Kong Stock Exchange | Hong Kong / China |
| `.SS` | Shanghai Stock Exchange | China |
| `.SZ` | Shenzhen Stock Exchange | China |
| `.KS` | Korea Exchange (KOSPI) | Corea del Sur |
| `.TW` | Taiwan Stock Exchange | Taiwán |
| `.AX` | Australian Securities Exchange | Australia |
| `.TO` | Toronto Stock Exchange | Canadá |
| `.SA` | B3 (Bolsa de São Paulo) | Brasil |
| `.MX` | Bolsa Mexicana de Valores | México |
| `.BA` | Bolsa de Buenos Aires | Argentina |
| `.NS` / `.BO` | NSE / BSE | India |

## Cosas a tener en cuenta al mezclar mercados

- **Fechas de cotización**: si combinas dos tickers de bolsas con
  festivos distintos (p. ej. un ticker de EE.UU. y uno de Japón),
  habrá días en los que uno cotiza y el otro no. `load_prices` alinea
  ambas series y aplica la `missing_policy` elegida (por defecto,
  relleno hacia delante) para esos huecos — pero ten en cuenta que
  eso introduce un pequeño desfase artificial en días de mercado
  cerrado en una de las dos bolsas.
- **Divisas**: los precios se descargan en la moneda de cotización de
  cada bolsa (EUR en `.MC`/`.PA`/`.DE`, GBP en `.L`, JPY en `.T`...).
  La app no hace conversión de divisa — si comparas o combinas
  tickers de distintas monedas (p. ej. para cointegración o pairs
  trading), los resultados reflejan también el movimiento del tipo de
  cambio entre ambas divisas, no solo el de las empresas.
- **Cobertura de datos**: Yahoo Finance no siempre tiene el mismo
  histórico de profundidad para todas las bolsas — algunos mercados
  emergentes tienen menos años de datos disponibles que EE.UU.
