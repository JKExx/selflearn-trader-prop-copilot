from dataclasses import dataclass


@dataclass
class ModelConfig:
    threshold: float = 0.55
    epsilon: float = 0.1
    min_samples_before_trade: int = 200


@dataclass
class RiskConfig:
    starting_cash: float = 10000.0
    risk_per_trade: float = 0.005
    kill_switch_dd: float = 0.10


@dataclass
class GeneralConfig:
    symbol: str = "EURUSD=X"
    interval: str = "1h"
    start: str = "2023-01-01"
    end: str = ""


@dataclass
class AppConfig:
    general: GeneralConfig = GeneralConfig()
    model: ModelConfig = ModelConfig()
    risk: RiskConfig = RiskConfig()
