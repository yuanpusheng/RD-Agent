.. _a_share_monitor:

==============================
A-share Monitoring Scenario
==============================

.. note::
   The A-share monitoring workflow is in an early, scaffolding stage.  It ships with
   a minimal loop that exercises the RDLoop plumbing so you can extend it with real
   data providers and alerting logic.

Quick start
===========

Run the live monitoring loop with default settings:

.. code-block:: bash

   rdagent a-monitor run

To replay a historical window via the same automation pipeline:

.. code-block:: bash

   rdagent a-monitor backtest --start 2024-01-01 --end 2024-03-31

Dashboard UI
============

Launch the Streamlit dashboard to explore persisted signals, market breadth, and the
underlying log traces:

.. code-block:: bash

   rdagent a-monitor ui --log-dir path/to/logs --universe CSI300

The UI exposes four tabs tailored to the monitoring workflow:

* **Overview** – aggregate metrics, market breadth visualisation, and a returns heatmap.
* **Watchlist & Filters** – interactive universe selection, severity/rule filters, and highlighted rules.
* **Signal Feed** – sortable feed with CSV export and integrated trace inspector hooked to RD-Agent's logging storage.
* **Stock Detail** – Plotly and optional ``mplfinance`` charts with signal timelines for the focused symbol.

Use ``--log-dir`` to point at an RD-Agent session folder if you want to inspect trace
messages, and ``--session`` to preselect a run nested under that directory. The
``--universe`` option seeds the default universe selector.

.. figure:: /_static/a_share_monitor_overview.svg
   :alt: A-share monitor overview tab with summary metrics and market breadth plots
   :width: 640px

   Overview tab highlighting key monitors.

.. figure:: /_static/a_share_monitor_detail.svg
   :alt: Stock detail tab illustrating candlestick overlays and trace integration
   :width: 640px

   Stock-specific drill down with optional ``mplfinance`` rendering and trace hooks.

Configuration
=============

Settings can be supplied through environment variables (``ASHARE_MONITOR_*``) or by
modifying the templates in :code:`configs/a_share_monitor/`.  Common options include:

``ASHARE_MONITOR_SYMBOLS``
    Comma-separated list of instruments to monitor (e.g. ``000300.SH,600519.SS``).

``ASHARE_MONITOR_LOOKBACK_DAYS``
    Rolling window length (defaults to 30).

``ASHARE_MONITOR_MODE``
    ``live`` or ``backtest``.  The CLI subcommands wire this up automatically.

Alert delivery can be configured through the ``RDC_MONITOR_ALERT_*`` environment family. Typical values include:

``RDC_MONITOR_ALERT_CHANNELS_ENABLED``
    Comma-separated list of enabled channels (``feishu``, ``wecom``, ``slack``, ``email``).

``RDC_MONITOR_ALERT_<CHANNEL>_WEBHOOK``
    Webhook endpoint for the given channel (for example ``RDC_MONITOR_ALERT_FEISHU_WEBHOOK``).

``RDC_MONITOR_ALERT_<CHANNEL>_SECRET``
    Optional signing secret or bearer token applied when posting alerts.

``RDC_MONITOR_ALERT_NOTIFICATION_COOLDOWN_MINUTES``
    Global rate-limit window for duplicate notifications.

``RDC_MONITOR_ALERT_SUBSCRIPTIONS_PATH``
    Path to a YAML file describing per-rule and per-symbol alert subscriptions.

Where to extend
===============

The scenario scaffolds the following components under
``rdagent/scenarios/a_share_monitor/``:

- ``scenario.py`` defines the human-readable background information.
- ``hypothesis.py`` wires hypothesis generation and experiment creation.
- ``developer.py`` implements placeholder coder/runner stages.
- ``feedback.py`` emits lightweight feedback objects for the loop trace.

Use these modules as touch points when connecting real data services, analytics, or
alerting destinations.
