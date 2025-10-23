.. _a_share_monitor:

==============================
A-share Monitoring Scenario
==============================

The A-share monitoring stack couples RD-Agent China's data ingestion, rules
engine, alert dispatcher, and Streamlit dashboard into an end-to-end workflow
for daily supervision of mainland equities.  This guide walks through the core
commands, configuration surfaces, and sample assets that ship with the
repository.

.. contents:: Contents
   :depth: 2
   :local:

Quick start
===========

1. Install and configure RD-Agent (see :doc:`../installation_and_configuration`).
2. Copy the sample configs so you can customise them locally:

   .. code-block:: bash

      mkdir -p monitor-config
      cp examples/ashare_monitoring/monitor_rules.yaml monitor-config/rules.yaml
      cp examples/ashare_monitoring/alert_subscriptions.yaml monitor-config/alerts.yaml
      cp examples/ashare_monitoring/universe_custom.json monitor-config/universe.json

3. Point the CLI at your copies (or edit the defaults in
   ``rdagent_china/config.py``):

   .. code-block:: bash

      export RDC_MONITOR_RULES_PATH="$(pwd)/monitor-config/rules.yaml"
      export RDC_MONITOR_ALERT_SUBSCRIPTIONS_PATH="$(pwd)/monitor-config/alerts.yaml"

Data ingestion
==============

Daily prices can be ingested in bulk or incrementally using the ``rdc`` CLI:

.. code-block:: bash

   # Full backfill for the CSI300 constituents
   rdc ingest --universe CSI300 --start 2024-01-01 --end 2024-06-30

   # Broad A-share coverage with a custom limit
   rdc ingest --universe ALL --limit 200

   # Incremental sync that respects per-symbol checkpoints
   rdc sync-price-daily --end 2024-06-30

The ingestion commands initialise the DuckDB database referenced by
``RDC_DUCKDB_PATH`` (defaults to ``rdagent_china/data/market.duckdb``).  You can
inspect the stored tables with standard DuckDB tooling if needed.

Running monitoring cycles
=========================

The ``monitor`` command evaluates your rules against freshly pulled data and
persists triggered alerts.  Two primary modes are supported:

* ``--run-once`` executes a single monitoring pass – ideal for cron jobs or
  validation.
* Without ``--run-once`` the command launches an APScheduler loop that runs at
  the configured end-of-day time (``RDC_MONITOR_EOD_TIME``) and optionally at an
  intraday cadence when ``--intraday`` is supplied.

Examples:

.. code-block:: bash

   # Evaluate the CSI300 universe once using the default rules
   rdc monitor --run-once --universe CSI300

   # Focus on a comma separated watchlist defined inline
   rdc monitor --run-once --watchlist 000001.SZ,600519.SH --start 2024-06-01

   # Launch the continuous scheduler with intraday refreshes every N minutes
   rdc monitor --intraday --universe CSI300

Watchlists can also be supplied via files.  The JSON payload in
``examples/ashare_monitoring/universe_custom.json`` illustrates how to bind a
named universe to bespoke include/exclude lists and reusable watchlists.

Backtesting
===========

Use ``rdc backtest`` to sanity check a rules configuration offline before
shipping it to the live loop:

.. code-block:: bash

   rdc backtest --symbols 000001.SZ 600519.SH --start 2024-01-01 --end 2024-03-31 \
       --strategy sma --report-path reports/ashare_backtest.html

The generated HTML report summarises returns, drawdown, and signal timing for the
selected period.

Alert routing
=============

Alert delivery is handled by :class:`rdagent_china.alerts.dispatcher.AlertDispatcher`.
Subscriptions can be embedded directly in the environment settings or loaded
from the YAML file referenced by ``RDC_MONITOR_ALERT_SUBSCRIPTIONS_PATH``.  The
example file demonstrates several patterns:

.. literalinclude:: ../../examples/ashare_monitoring/alert_subscriptions.yaml
   :language: yaml

In summary:

* ``rule`` accepts explicit names or ``"*"`` for wildcards.
* ``universes`` and ``symbols`` filter destinations.
* ``channels`` restrict delivery to configured webhooks (Feishu, WeCom, Slack,
  Email).  Channel webhooks are configured through the ``RDC_MONITOR_ALERT_*``
  environment variables.

Dashboard
=========

Launch the Streamlit dashboard to explore persisted alerts, breadth metrics, and
trace logs:

.. code-block:: bash

   rdc dashboard --port 19555

Use ``--log-dir`` to point at an RD-Agent session folder if you want to enable
trace inspection, and ``--universe`` to seed default filters.

.. figure:: /_static/a_share_monitor_overview.svg
   :alt: A-share monitor overview tab with summary metrics and market breadth plots
   :width: 640px

   Overview tab highlighting key monitors.

.. figure:: /_static/a_share_monitor_detail.svg
   :alt: Stock detail tab illustrating candlestick overlays and trace integration
   :width: 640px

   Stock-specific drill down with optional ``mplfinance`` rendering and trace hooks.

Sample fixtures
===============

Automated tests use ``rdagent_china/tests/fixtures/price_daily_sample.csv`` and
match the example rules in ``examples/ashare_monitoring/monitor_rules.yaml``.  The
combination produces a ``volume_breakout_combo`` alert for ``000001.SZ``, which
is then routed to Feishu and Email in the sample subscriptions.  You can reuse
these assets to smoke test environment changes or to experiment with alternative
rule definitions.

Further reference
=================

+ :mod:`rdagent_china.data.adapters` – Akshare and Tushare data adapters used by
  the monitor.
+ :mod:`rdagent_china.signals.rules` – config-driven rules engine powering
  signal evaluation.
+ :mod:`rdagent_china.monitor.loop` – orchestration loop tying data, signals,
  persistence, and alerting together.
