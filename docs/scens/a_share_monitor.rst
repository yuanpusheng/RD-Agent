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
