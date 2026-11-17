====
KPIs
====

KPIs measure the completed simulation: energy, cost, emissions, service,
comfort and electrical performance. They are available from Python and CSV
exports; using the UI is optional.

Choose what to read
*******************

* :doc:`guides/kpis_reference` — definitions, units, equations and metric families.
* :doc:`guides/KPI_V2_TREE` — names grouped by district/building and family.
* :doc:`guides/baselines` — the reference behind normalized results.
* :doc:`guides/scorecard` — a compact set for comparing controllers.
* :doc:`guides/scorecard_kpi_names` — the simulator's exact default selection.

List the KPIs in your run
*************************

After completing an episode:

.. code-block:: python

   kpis = env.evaluate_v2()
   names = kpis[["cost_function", "level"]].drop_duplicates()
   print(names.sort_values("cost_function").to_string(index=False))
   kpis.to_csv("kpis.csv", index=False)

The table contains ``cost_function``, ``value``, ``name`` and ``level``.
``name`` identifies the building or district; ``level`` identifies the scope.
Feature-specific rows follow the scenario configuration.

Main families
*************

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Family
     - Measures
   * - Cost and grid energy
     - Operating cost, imports, exports, net exchange, peaks and ramping.
   * - Emissions and PV
     - Carbon emissions, generation and local solar self-consumption.
   * - EVs and batteries
     - Departure service, target accuracy, charging, V2G, throughput and degradation.
   * - Electrical service
     - Residual violations, requested pressure and phase indicators.
   * - Flexible loads and escalators
     - Completed/missed cycles, delays, energy and passenger service.
   * - Comfort and resilience
     - Thermal discomfort and service during outages.
   * - Community market and equity
     - Local exchange, settlement costs, savings and their distribution.
   * - Demand response
     - Requested, delivered and credited response, shortfall and settlement.
   * - Robustness
     - Measurement, action and availability perturbation counters.

``evaluate()`` retains historical cost-function names. ``evaluate_v2()`` is the
structured KPI API used by CityLearn v3; its method name is not a software
version number.

Audited appliance service
*************************

The additive ``building_deferrable_appliance_audit_*`` and
``district_deferrable_appliance_audit_*`` families use
``deferrable-service-audit-v1``. They distinguish pending, running, completed,
missed and cancelled requests and expose excluded populations. Delivered energy
counts executed intervals, including partial cycles, rather than future profile
reservations. Requested energy equals served, final unserved and outstanding
energy. Completion ratios must be read with resolved coverage; empty denominators
are undefined.

``env.get_deferrable_service_audit()`` exposes individual requests. Final KPI exports
include an adjacent ``.deferrable_service.json`` and versioned manifest metadata.
Historical appliance service KPIs retain their previous definitions.


Paired comparison evidence
***************************

Final KPI manifests include ``paired-experiment-v1`` conditions, actual native
reference parameters and explicit controller identity when supplied. The comparison
API/CLI distinguishes paired, incompatible and unknown conditions. Controller and
sharing interventions have separate pairing rules. Cost differences must be read
with audited service, coverage and final storage inventory. Historical exports
without the required evidence cannot be silently treated as paired.


Executed intervals and independent balances
*******************************************

Iteration 4 (included from v1.9.7) reconciles energy, source/destination
cohorts, acquisition carbon and member/operator cash with a separate flow-network
reference. Energy/cost/PV totals include only executed active intervals, preserving
completed flows through member removal and re-entry. Terminal observations do not
add another dispatched interval.

Provenance definition ``1.1`` adds ``curtailment`` for islanded unused PV. It is
available generation, excluded from useful load and instantaneous absorption.
BESS/charger totals retain gross charge/discharge before interval netting.
Affected experiments require rerunning; historical 1.0 exports cannot reconstruct
these corrected boundaries. Initial/final inventory remains separate from useful
service and cash.

See :doc:`guides/independent_balance_audit` for the bounded CLI, reference method,
explicit tolerances, regression cases and limits. The audit is optional offline
instrumentation; it does not provide empirical calibration or a universal
certificate for every dataset/controller.


Audited KPI populations
***********************

Iteration 5, included from v1.9.8, adds ``kpi-families-audit-v1`` thermal service
and equity coverage, plus ``demand-response-audit-v1`` reference quality.
``env.get_kpi_family_audit()`` reports executed active samples, occupied/known
populations and energy shortfalls by service. District audit ratios pool exposure
and energy. Native comfort bands and conditional severity statistics are corrected.

DR preparation rounds up to the requested duration and excludes structurally
changed references, even after caching. Explicit service/stress/unspecified groups
retain separate denominators. Member signed delivery and cash are additive;
individual credit/shortfall diagnostics are not district totals. Missing reference
or no applicable events give undefined audit rates with counts and coverage.

The export manifest records definitions and reward evidence without changing
the reward. See :doc:`guides/kpi_family_audit` for formulas, migration and limits.


Temporal evidence and daily rates
*********************************

Local iteration 6 after v1.9.8 adds ``temporal-evidence-v1``: canonical UTC sample
clocks, initialization/reset seeds, loaded input hashes and dependency versions.
Portuguese DST transitions preserve the declared sample spacing. A control-step
override is explicitly distinguished from real resampling.

Member daily averages use executed active participation, excluding the current
unexecuted observation and removal/re-entry gaps. Zero elapsed duration gives an
undefined rate. The manifest separates sample indexes and executed intervals;
terminal padding is explicit. See :doc:`guides/temporal_reproducibility` for replay,
resampling conventions and limitations.

To explore the same exported results graphically, continue to :doc:`ui`.
