# Fontes físicas preservadas · Portugal REC

Reference year: **2023**. Retrieved 25 September 2026. Individual buildings,
occupancy and equipment are synthetic; the files here are independent inputs.

- [E-REDES consumption profiles](https://www.e-redes.pt/sites/eredes/files/2023-01/E-REDES_Perfil_Consumo_2023.xlsx): original workbook and chronological CSV, including 92/100 intervals on Lisbon DST dates. BTN-C is a shape prior, not measured individual households.
- [PVGIS API](https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/using-pvgis-5/api-non-interactive-service_en): Lisbon 38.72,-9.14, SARAH3, 2023; archived response includes all parameters. Hourly satellite irradiance and reanalysis temperature/wind are interpolated to 15 minutes. No claim of observed quarter-hour clouds.
- [OMIE day-ahead prices](https://www.omie.es/pt/market-results/daily/daily-market/day-ahead-price): `omie_2023_market_periods.parquet` preserves the raw market days. `omie_2023_daily_provenance.csv` records source URLs/hashes. `omie_2023_hourly.parquet` now assigns each period to **Europe/Madrid**, then converts to UTC. The first 2024 market period supplies the final UTC hour of 2023. The former one-hour UTC misalignment is corrected.
- [OMIE definitive adjustment](https://www.omie.es/es/market-results/daily/average-final-prices/hourly-price-consumers?date=2023-01-01&scope=daily): 365 daily snapshots in `raw_2023/omie_adjustment_*.json.gz`, with source URLs and HTTP payload hashes. Includes signed, nonzero adjustments. [DGEG expiry](https://www.dgeg.gov.pt/pt/estatistica/energia/precos-de-energia/mecanismo-iberico/): 31 December 2023.
- [Coopérnico tariff, January 2023](https://www.coopernico.org/files/documents/Coopernico_Tarifario_Indexado.pdf): BASE supply formula uses CGS 0.004, margin 0.010 EUR/kWh and a 15% loss uplift. This historical PDF is archived; the modern 2.0 formula is not substituted for 2023. BTE uses the same **benchmark** supply convention, not a quoted Coopérnico BTE contract. The adjustment is applied to procurement energy before losses.
- [ERSE H1 2023](https://www.erse.pt/media/vueeumz4/tarifas-e-pre%C3%A7os-2023.pdf) and [H2 2023](https://www.erse.pt/media/xjeks20k/tarifas-e-pre%C3%A7os-se-2sem2023.pdf): regulated access energy and contracted-power rates. H2 begins at 00:00 Lisbon on 1 July. Negative access terms are preserved. [2023 tariff regulation](https://www.erse.pt/media/y02kc00v/regulamento-9-2023_rt_site.pdf), article 43: peak power is Ep/Hp; daily BTE has four peak hours per day, so its 0.4854 EUR/kW/day becomes 0.12135 EUR/kWh in peak intervals. Winter peak 09:00–10:30 and 18:00–20:30; summer 10:30–13:00 and 19:30–21:00. Daily off-peak 22:00–08:00, BTE super-off-peak 02:00–06:00. [ERSE period tables](https://www.erse.pt/atividade/regulacao/tarifas-e-precos-eletricidade).
- [REN API documentation](https://datahub.ren.pt/en/api-instructions): daily `ElectricityProductionBreakdownDaily` snapshots under `raw_2023/ren_*.json.gz`, MW values at Lisbon quarter-hour intervals. Clock labels are checked against physical DST-aware timestamps.
- [APA emission factors](https://apambiente.pt/clima/fator-de-emissao-de-gases-de-efeito-de-estufa-para-eletricidade-produzida-em-portugal): archived 2025 report, mainland **2023 annual production 0.092 kg CO2e/kWh**. This is not the all-Portugal value or a five-year average.
- [REE API](https://www.ree.es/en/datos/apidata): Spanish peninsular daily operational production intensity, `no-renovables-detalle-emisiones-CO2`; exact request URLs inside the archived JSON. Daily values are held within their physical Spanish day.

## Derived inputs and scientific scope

`python scripts/prepare_portugal_reference.py` reconstructs prices and carbon
offline, without requesting any website. `aligned_reference_metadata.json`
describes formulas and exclusions. `price_components_*.parquet` decomposes the
five tariffs; `pricing_*.parquet` supplies actual and causal forecast columns.
All prices are EUR/kWh, before VAT. IEC 0.001 EUR/kWh is included, without tax
exemptions. Fixed power charges, audiovisual/DGEG fees, VAT and supplier fixed
margins are not part of the marginal control KPI; fixed access rates are
reported separately in each dataset. This does not reproduce a full invoice.

OMIE forecasts use publication at a conservative 14:00 Madrid. The definitive
MIBEL adjustment is estimated from past observations, not revealed from the
future. Access rates and their known calendar are used for the target period.
The archived E-REDES loss profile is background material; this version uses
the historical Coopérnico flat 15% convention consistently.

`carbon_components_2023.parquet` contains the REN mix, REE imports proxy,
production estimate, delivered estimate and balance residual. A common fossil
coefficient is calibrated so generation-weighted Portuguese production equals
the APA annual factor. Import blending assumes Spain's daily production mix;
the delivery factor is 1.15. **Derived average operational emissions, not
measured 15-minute emissions, marginal emissions, lifecycle emissions or a
full cross-border flow tracing result.** Biomass biogenic CO2 and stored-hydro
carbon attribution are not independently modelled. The annual calibration
reconciles different accounting boundaries and is not independent validation.

SHA-256 hashes of all source files, including daily archives, are in the
generated dataset manifests. Raw provider data remain attributed; public
access does not establish unrestricted redistribution rights.


## Native-runtime recovery

The physical inputs and source tariff series above are retained. The current dataset schemas all select the same BTN bi-hourly reference, `shared/pricing_btn_bi.parquet`; other tariff files are provenance only. Dataset prices are full variable grid purchase prices. The 0.8 local-market factor belongs in the experiment schema and uses the original simulator settlement. Dataset schemas leave this market disabled. Source provenance is not evidence of longitudinal EV simulation, calibrated DR or added inverter constraints. No new usage evaluation accompanies this tariff revision.
