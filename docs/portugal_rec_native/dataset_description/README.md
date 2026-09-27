# Descriptive report - supporting data

The English report `../Portugal_REC_Dataset_Report.pdf` describes dataset edition
`portugal-native-2023-2024-v3.1`. Its editable Markdown companion contains the same
prose, tables and captions. Figures are available as PNG and editable SVG in
`../figures/descriptive/`.

All statistics come from the published datasets. No CityLearn environment or
controller is executed by the report builders. Dataset files are read only.

## Reading the tables

- `inventory.csv`: full static-base inventories and annual energy totals.
- `members.csv`: descriptive member and asset fields; narrative prose is omitted.
- `annual_member_energy.csv`: non-shiftable load and PV energy by member/year.
- `sessions.csv`: visits reconstructed from connector files selected by Normal
  schemas. Beginning/end censoring is explicit. Parking crossing New Year remains
  one session, attributed to the year of its start.
- `cycles.csv`, `cycle_profiles.csv`: controllable requests and their programme
  energy/duration. Fixed appliance energy is already inside non-shiftable demand.
- `electrical_limits.csv`: connection and total-limit settings in Normal schemas.
- `variants.csv`: configuration factors, active inventories and lifecycle counts.
- `topology_events.csv`, `topology_timeline.csv`: schema events and the resulting
  active inventory. Internal template registration is excluded from real events.
- `health.csv`: declared development/confirmation channel rows. Missing category
  metadata is classified for description from the event mode; source files are
  unchanged. Several channel rows can describe one logical event.
- `outages.csv`: scheduled site-level outage episodes, not energy not served.
- `dr.csv`: explicit requested service, not delivered response or settlement.
- `statistics.json`: source checksum identities, aggregation conventions,
  descriptive statistics and 34 construction/aggregation consistency checks.
- `report_structure.json`: section and figure index.
- `report_qa.json`: final PDF identity, consistency-check count and layout review.

Inventories are alternatives across bases, not additive independent samples.
UTC years define annual totals; daily schedules and clock-hour summaries use
Europe/Lisbon. Physical interval energy is kWh/15 min; multiply by four for
interval-average kW. SOC remains a fraction in the underlying data.

## Reproduce

From the Simulator repository, use a Python environment with the packages in
`scripts/requirements-portugal-report.txt`:

```bash
python scripts/build_portugal_dataset_description.py
python scripts/render_portugal_dataset_report.py
```

The first script reads the four current bases and produces the CSV tables,
statistics and figures. The second composes the PDF and Markdown. The saved
checksums identify the inputs described. The figures are descriptive data plots;
they do not reproduce any previous controller experiment or audit outcome.
