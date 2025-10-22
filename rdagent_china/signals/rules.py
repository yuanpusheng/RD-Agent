from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import yaml

from .base import IndicatorOutput, SignalRecord
from .events import EVENT_REGISTRY
from .indicators import INDICATOR_REGISTRY


class RulesEngine:
    """Config-driven rules engine for composing and emitting trading signals."""

    def __init__(
        self,
        config: Mapping[str, Any],
        indicator_registry: Mapping[str, Any] | None = None,
        event_registry: Mapping[str, Any] | None = None,
    ):
        self.config = dict(config)
        self.indicator_registry = dict(indicator_registry or INDICATOR_REGISTRY)
        self.event_registry = dict(event_registry or EVENT_REGISTRY)
        self.signal_definitions = self.config.get("signals", {})
        self.rule_definitions = self.config.get("rules", [])
        if not isinstance(self.signal_definitions, Mapping):
            raise ValueError("'signals' section must be a mapping of signal name to definition")
        if not isinstance(self.rule_definitions, Iterable):
            raise ValueError("'rules' section must be an iterable of rule definitions")

    @classmethod
    def from_dict(cls, config: Mapping[str, Any]) -> "RulesEngine":
        return cls(config=config)

    @classmethod
    def from_file(cls, path: str | Path) -> "RulesEngine":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in {".yaml", ".yml"}:
            config = yaml.safe_load(text)
        else:
            config = json.loads(text)
        return cls.from_dict(config)

    def evaluate(
        self,
        data: pd.DataFrame,
        universe: str | None = None,
        config_version: str | None = None,
        run_version: str | None = None,
    ) -> list[SignalRecord]:
        if "symbol" not in data.columns:
            raise ValueError("Input data must include a 'symbol' column for rules evaluation")

        working = data.copy()
        time_column = next((col for col in ("date", "datetime", "timestamp", "dt") if col in working.columns), None)
        if time_column:
            working[time_column] = pd.to_datetime(working[time_column])

        records: list[SignalRecord] = []
        grouped = working.groupby("symbol")
        for symbol, frame in grouped:
            if time_column:
                frame = frame.sort_values(time_column)
                indexed = frame.set_index(time_column)
            else:
                frame = frame.sort_index()
                if not isinstance(frame.index, pd.DatetimeIndex):
                    raise ValueError(
                        "Input data must have a datetime column or index for rules evaluation"
                    )
                indexed = frame
            indexed.index = pd.to_datetime(indexed.index)
            computed = self._compute_signals(indexed)
            symbol_records = self._evaluate_rules_for_symbol(
                symbol=symbol,
                signals=computed,
                universe=universe,
                config_version=config_version,
                run_version=run_version,
            )
            records.extend(symbol_records)
        return records

    def _compute_signals(self, frame: pd.DataFrame) -> dict[str, IndicatorOutput]:
        computed: dict[str, IndicatorOutput] = {}
        for name, definition in self.signal_definitions.items():
            if not isinstance(definition, Mapping):
                raise ValueError(f"Signal definition for '{name}' must be a mapping")
            source_type = definition.get("type", "indicator")
            params = definition.get("params", {})
            if not isinstance(params, Mapping):
                raise ValueError(f"Signal '{name}' params must be a mapping")
            if source_type == "indicator":
                func_name = definition.get("indicator", name)
                func = self.indicator_registry.get(func_name)
            elif source_type == "event":
                func_name = definition.get("event", name)
                func = self.event_registry.get(func_name)
            else:
                raise ValueError(f"Unsupported signal type '{source_type}' for '{name}'")
            if func is None:
                raise KeyError(f"Signal function '{func_name}' not found in registry")
            result = func(frame.copy(), **dict(params))
            if not isinstance(result, IndicatorOutput):
                raise TypeError(f"Signal '{name}' produced unsupported output type: {type(result)}")
            computed[name] = result
        return computed

    def _evaluate_rules_for_symbol(
        self,
        symbol: str,
        signals: Mapping[str, IndicatorOutput],
        universe: str | None,
        config_version: str | None,
        run_version: str | None,
    ) -> list[SignalRecord]:
        records: list[SignalRecord] = []
        for rule in self.rule_definitions:
            if not isinstance(rule, Mapping):
                raise ValueError("Each rule definition must be a mapping")
            rule_name = rule.get("name")
            if not rule_name:
                raise ValueError("Rule definition missing 'name'")
            triggered_series = self._evaluate_rule(rule, signals)
            triggered_series = triggered_series[triggered_series.fillna(False)]
            for timestamp, triggered in triggered_series.items():
                if not bool(triggered):
                    continue
                ts = pd.to_datetime(timestamp)
                payload = self._collect_signal_payload(signals, ts)
                value = self._resolve_rule_value(rule, signals, ts)
                emit = rule.get("emit", {})
                record = SignalRecord(
                    universe=universe,
                    symbol=symbol,
                    timestamp=ts,
                    rule=rule_name,
                    label=emit.get("label", rule_name),
                    severity=emit.get("severity", "info"),
                    triggered=True,
                    value=value,
                    config_version=config_version,
                    run_version=run_version,
                    signals=payload,
                )
                records.append(record)
        return records

    def _evaluate_rule(self, rule: Mapping[str, Any], signals: Mapping[str, IndicatorOutput]) -> pd.Series:
        conditions_all = rule.get("all")
        conditions_any = rule.get("any")
        if conditions_all is None and conditions_any is None:
            raise ValueError("Rule must define either 'all' or 'any' condition groups")

        if conditions_all is not None:
            series_list = [self._evaluate_condition(cond, signals) for cond in conditions_all]
            result = self._series_reduce(series_list, operator="and")
        else:
            series_list = [self._evaluate_condition(cond, signals) for cond in conditions_any]
            result = self._series_reduce(series_list, operator="or")

        return result.fillna(False)

    def _evaluate_condition(self, condition: Mapping[str, Any], signals: Mapping[str, IndicatorOutput]) -> pd.Series:
        if not isinstance(condition, Mapping):
            raise ValueError("Condition must be a mapping")
        signal_name = condition.get("signal")
        if signal_name is None:
            raise ValueError("Condition missing 'signal' reference")
        indicator = signals.get(signal_name)
        if indicator is None:
            raise KeyError(f"Signal '{signal_name}' not computed")
        field = condition.get("field")
        series = indicator.get_series(field)
        operator = condition.get("operator", "eq")
        value = condition.get("value")

        if operator == "eq":
            return series == value
        if operator == "ne":
            return series != value
        if operator == "gt":
            return series.astype(float) > float(value)
        if operator == "gte":
            return series.astype(float) >= float(value)
        if operator == "lt":
            return series.astype(float) < float(value)
        if operator == "lte":
            return series.astype(float) <= float(value)
        if operator == "is_true":
            return series.astype(bool)
        if operator == "is_false":
            return ~series.astype(bool)
        raise ValueError(f"Unsupported operator '{operator}' in condition for signal '{signal_name}'")

    @staticmethod
    def _series_reduce(series_list: Iterable[pd.Series], operator: str) -> pd.Series:
        iterator = iter(series_list)
        try:
            result = next(iterator)
        except StopIteration:
            raise ValueError("Rule conditions cannot be empty") from None
        for series in iterator:
            if operator == "and":
                result = result & series
            else:
                result = result | series
        return result

    def _collect_signal_payload(self, signals: Mapping[str, IndicatorOutput], timestamp: pd.Timestamp) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for name, output in signals.items():
            try:
                payload[name] = output.point_values(timestamp)
            except KeyError:
                continue
        return payload

    def _resolve_rule_value(
        self,
        rule: Mapping[str, Any],
        signals: Mapping[str, IndicatorOutput],
        timestamp: pd.Timestamp,
    ) -> Any:
        value_spec = rule.get("value")
        if value_spec is None:
            return 1.0
        if isinstance(value_spec, (int, float, str)):
            return value_spec
        if isinstance(value_spec, Mapping):
            signal_name = value_spec.get("signal")
            field = value_spec.get("field")
            if signal_name is None:
                raise ValueError("Value spec mapping must include 'signal'")
            indicator = signals.get(signal_name)
            if indicator is None:
                raise KeyError(f"Signal '{signal_name}' referenced in value spec not computed")
            series = indicator.get_series(field)
            try:
                return series.loc[timestamp]
            except KeyError:
                return None
        raise ValueError("Unsupported value specification in rule")


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a configuration dictionary from YAML or JSON."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(text)
    return json.loads(text)


__all__ = ["RulesEngine", "load_config"]
