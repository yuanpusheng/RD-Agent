# A-share Monitoring Examples

This directory contains ready-to-use configuration snippets that accompany the
RD-Agent China A-share monitoring stack.

* `monitor_rules.yaml` – baseline rule set combining EMA crossovers, volume
  pressure, and gap detection.
* `alert_subscriptions.yaml` – sample channel routing showing per-rule and
  per-symbol subscriptions.
* `universe_custom.json` – example of constraining the CSI300 universe and
  declaring reusable watchlists for the monitor CLI.

These files are referenced from the documentation (`docs/scens/a_share_monitor.rst`)
so you can copy them into your own workspace and adapt them as needed.
