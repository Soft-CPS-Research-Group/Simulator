======
Agents
======

CityLearn supports centralized, decentralized-independent and decentralized-coordinated control architectures. In the centralized architecture, 1 agent controls all storage, cooling and heating devices i.e. provides as many actions as storage, cooling and heating devices in the district. In the decentralized-independent architecture, each building has it's own unique agent and building agents do not share information i.e. each agent acts in isolation and provides as many actions as storage, cooling and heating devices in the building it controls. The decentralized-coordinated architecture is similar to the decentralized-independent architecture with the exception of information sharing amongst agents.

CityLearn provides implementations of rule-based control (RBC) and reinforcement learning control (RLC) algorithms. The rule-based control algorithms are in the :py:mod:`citylearn.agents.rbc` module, while there are two reinforcement learning modules: :py:mod:`citylearn.agents.sac` and :py:mod:`citylearn.agents.marlisa`. The table below summarizes which control architectures the control algorithms support:

.. csv-table::
   :file: ../../../assets/tables/citylearn_agents.csv
   :header-rows: 1

Dynamic operational baselines
-----------------------------

``NormalPolicy`` and the other ``BusinessAsUsualAgent`` policies refresh their
action layout when members or equipment change. In decentralized dynamic runs,
``agent.action_history_ids`` identifies the stable rows of ``agent.actions``:
departed members retain their history, new members receive new rows, and steps
without an action contain an empty vector. These historical rows need not match
the current active-member order. Centralized runs retain one ``central_agent``
history row. Calling the policy after an environment reset starts a fresh history.

The ``learn`` loop preserves entity observations and reports dynamic reward
summaries by member identity, using the members that executed each transition.
