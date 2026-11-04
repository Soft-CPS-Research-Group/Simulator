# Validation and performance

Use these tools to check a simulation, reproduce a reported result or investigate
service shortfalls. Run reports belong outside the repository. The scripts check
the cases they execute; they do not establish empirical equipment calibration.

## Choose a check

| Question | Guide |
|---|---|
| Do electricity flows, local sharing and cash reconcile? | [Independent balance audit](independent_balance_audit.md) |
| Were EV departure requirements observed and met? | [EV departure service](ev_departure_audit.md) |
| Were appliance requests completed, missed or cancelled? | [Deferrable service](deferrable_service_audit.md) |
| Are thermal service, equity and demand response accounted for? | [KPI family audits](kpi_family_audit.md) |
| Where did consumed and stored energy originate? | [Energy provenance](energy_provenance.md) |
| Do timestamps, inputs and initialization agree? | [Temporal reproducibility](temporal_reproducibility.md) |
| Are two exports comparable? | [Paired comparisons](experiment_comparisons.md) |
| Does a continuous run reconcile, and how much time/RAM does it use? | [Long-run validation](long_run_validation.md) |

## Checks for a source change

The [developer guide](developer_guide.md#core-validation) describes unit tests,
strict auditors, fixed-command regression references and installed-package checks.
For KPI meanings and undefined values, use the [KPI reference](kpis_reference.md).

## Published evidence

The [current release notes](releases.md) retain the measured memory, dispatch and
evaluation improvements, test counts, input versions and scope limits. Earlier
results remain in the [release archive](releases_archive.md).
Keep input versions and controller definitions matched when reproducing a result;
a release's validation does not automatically cover another dataset edition.
