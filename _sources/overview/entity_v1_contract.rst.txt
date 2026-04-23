==================
Entity v1 Contract
==================

CityLearn provides two interfaces:

* ``interface="flat"``: legacy fixed-size vectors (backward compatible).
* ``interface="entity"``: canonical table/edge payload for graph, hierarchical and transformer agents.

Entity mode contract
--------------------

At each step, ``reset()`` and ``step()`` return:

.. code-block:: python

   {
       "tables": {
           "district": np.ndarray,  # [n_district, n_features]
           "building": np.ndarray,  # [n_buildings, n_features]
           "charger": np.ndarray,   # [n_chargers, n_features]
           "ev": np.ndarray,        # [n_evs, n_features]
           "storage": np.ndarray,   # [n_storage, n_features]
           "pv": np.ndarray,        # [n_pv, n_features]
       },
       "edges": {
           "district_to_building": np.ndarray,      # [n_buildings, 2]
           "building_to_charger": np.ndarray,       # [n_chargers, 2]
           "building_to_storage": np.ndarray,       # [n_storage, 2]
           "building_to_pv": np.ndarray,            # [n_pv, 2]
           "charger_to_ev_connected": np.ndarray,   # [n_chargers, 2]
           "charger_to_ev_connected_mask": np.ndarray,
           "charger_to_ev_incoming": np.ndarray,    # [n_chargers, 2]
           "charger_to_ev_incoming_mask": np.ndarray,
       },
       "meta": {
           "time_step": int,
           "endogenous_time_step": int,
           "spec_version": "entity_v1",
           "temporal_semantics": {
               "exogenous": "t",
               "endogenous": "t_minus_1_settled",
           },
           "topology_version": int,
       },
   }

``entity_specs`` provides stable metadata for tooling/model builders:

* table IDs and feature names,
* per-feature unit/bundle/legacy tags,
* action schemas,
* edge schemas,
* topology metadata,
* normalization/encoding policy.

Temporal semantics
------------------

In ``entity_v1``:

* Exogenous observations are read at current control index ``t``.
* Endogenous observations are read from settled transition ``t-1`` (clamped at 0 on reset).
* In dynamic topology mode, events at ``time_step=k`` apply after transition ``k-1 -> k`` and before observation ``k``.

Dynamic topology notes
----------------------

When ``topology_mode="dynamic"`` (entity mode only), table sizes can grow/shrink during the episode.

* IDs are canonical and stable while entities are active.
* Relation masks identify valid EV-charger relations each step.
* Simulator outputs raw values; normalization should be handled externally (running stats per feature).
