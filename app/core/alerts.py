"""Sistema de alertas y notificaciones multicanal.

Canales soportados:
    - console: imprime en stdout (útil para desarrollo y logs).
    - file:    escribe en un archivo de log local.
    - email:   envía via SMTP (Gmail, Outlook, etc.).
    - telegram: envía via bot de Telegram.
    - slack:   envía via webhook de Slack.

Las credenciales se leen de variables de entorno o de
`.streamlit/secrets.toml` (nunca hardcodeadas).

Reglas de alerta:
    - Un conjunto de umbrales sobre métricas del backtest.
    - Si se cumplen, se dispara el envío por todos los canales activos.

Uso:
    from app.core.alerts import (
        AlertChannel, AlertRule, AlertSeverity,
        AlertDispatcher, check_backtest_rules,
    )

    dispatcher = AlertDispatcher.from_env()
    rules = [AlertRule("sharpe", ">=", 1.5, AlertSeverity.INFO, "Sharpe alto")]
    alerts = check_backtest_rules(metrics, rules)
    for alert in alerts:
        dispatcher.dispatch(alert)
"""
from __future__ import annotations

import json
import logging
import os
import smtplib
import ssl
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from pathlib import Path
from typing import Any, Literal, cast, get_args
from urllib import request as urlrequest

logger = logging.getLogger(__name__)


# ============================================================
#  Tipos
# ============================================================
class AlertSeverity(str, Enum):
    """Nivel de severidad de una alerta."""
    INFO = "info"
    WARNING = "warning"
    DANGER = "danger"
    SUCCESS = "success"


ChannelName = Literal["console", "file", "email", "telegram", "slack"]
Comparator = Literal[">", ">=", "<", "<=", "==", "!="]


# ============================================================
#  Modelos
# ============================================================
@dataclass
class Alert:
    """Una alerta lista para enviar."""
    title: str
    message: str
    severity: AlertSeverity = AlertSeverity.INFO
    metric: str | None = None
    value: float | None = None
    threshold: float | None = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "message": self.message,
            "severity": self.severity.value,
            "metric": self.metric,
            "value": self.value,
            "threshold": self.threshold,
            "timestamp": self.timestamp,
            "extra": self.extra,
        }

    def format_text(self) -> str:
        """Formato texto plano para consola, email o Telegram."""
        emoji = {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.DANGER: "🚨",
            AlertSeverity.SUCCESS: "✅",
        }.get(self.severity, "•")
        parts = [f"{emoji} [{self.severity.value.upper()}] {self.title}"]
        parts.append(self.message)
        if self.metric is not None:
            parts.append(f"Métrica: {self.metric} = {self.value}")
            if self.threshold is not None:
                parts.append(f"Umbral: {self.threshold}")
        parts.append(f"Timestamp: {self.timestamp}")
        return "\n".join(parts)


@dataclass
class AlertRule:
    """Regla de disparo de alerta.

    Si `metric <comparator> threshold` se cumple, se genera una alerta.
    """
    metric: str
    comparator: Comparator
    threshold: float
    severity: AlertSeverity = AlertSeverity.INFO
    title: str = ""
    message: str = ""

    def matches(self, metrics: dict[str, Any]) -> bool:
        """Comprueba si la regla se cumple con las métricas dadas."""
        if self.metric not in metrics:
            return False
        try:
            value = float(metrics[self.metric])
        except (TypeError, ValueError):
            return False

        ops: dict[Comparator, Callable[[float, float], bool]] = {
            ">": lambda a, b: a > b,
            ">=": lambda a, b: a >= b,
            "<": lambda a, b: a < b,
            "<=": lambda a, b: a <= b,
            "==": lambda a, b: a == b,
            "!=": lambda a, b: a != b,
        }
        return ops[self.comparator](value, self.threshold)

    def to_alert(self, metrics: dict[str, Any]) -> Alert:
        """Construye la alerta a partir de las métricas."""
        value = float(metrics[self.metric]) if self.metric in metrics else None
        title = self.title or f"{self.metric} {self.comparator} {self.threshold}"
        msg = self.message or (
            f"La métrica '{self.metric}' = {value} cumple la condición "
            f"{self.comparator} {self.threshold}."
        )
        return Alert(
            title=title,
            message=msg,
            severity=self.severity,
            metric=self.metric,
            value=value,
            threshold=self.threshold,
        )


# ============================================================
#  Configuración de canales
# ============================================================
@dataclass
class EmailConfig:
    smtp_host: str
    smtp_port: int
    username: str
    password: str
    from_addr: str
    to_addrs: list[str]
    use_tls: bool = True


@dataclass
class TelegramConfig:
    bot_token: str
    chat_id: str


@dataclass
class SlackConfig:
    webhook_url: str


@dataclass
class FileConfig:
    path: Path = Path("data/alerts.log")


# ============================================================
#  Dispatcher
# ============================================================
class AlertDispatcher:
    """Envía alertas a todos los canales activos.

    Los canales se configuran en el constructor. Las credenciales se
    leen de variables de entorno o de un dict (por ejemplo, de
    `st.secrets`).
    """

    def __init__(
        self,
        channels: list[ChannelName] | None = None,
        email_config: EmailConfig | None = None,
        telegram_config: TelegramConfig | None = None,
        slack_config: SlackConfig | None = None,
        file_config: FileConfig | None = None,
    ) -> None:
        self.channels = channels or ["console"]
        self.email_config = email_config
        self.telegram_config = telegram_config
        self.slack_config = slack_config
        self.file_config = file_config or FileConfig()
        self._history: list[Alert] = []

    # --------------------------------------------------------
    #  Construcción desde entorno
    # --------------------------------------------------------
    @classmethod
    def from_env(cls, source: dict[str, Any] | None = None) -> AlertDispatcher:
        """Construye un dispatcher leyendo config de variables de entorno.

        Args:
            source: Dict opcional con la config (por ejemplo, `st.secrets`).
                Si None, se leen variables de entorno del sistema.
        """
        def get(key: str, default: str | None = None) -> str | None:
            if source is not None and key in source:
                return str(source[key])
            return os.getenv(key, default)

        def get_bool(key: str, default: bool = False) -> bool:
            val = get(key)
            if val is None:
                return default
            return val.lower() in ("1", "true", "yes", "on")

        channels_raw = get("FQL_ALERT_CHANNELS", "console") or "console"
        raw_list = [c.strip() for c in channels_raw.split(",") if c.strip()]
        valid_names = set(get_args(ChannelName))
        channels: list[ChannelName] = [
            cast(ChannelName, c) for c in raw_list if c in valid_names
        ]
        invalid = [c for c in raw_list if c not in valid_names]
        if invalid:
            logger.warning(
                "Canales de alerta desconocidos ignorados: %s (válidos: %s)",
                invalid, sorted(valid_names),
            )

        email_config: EmailConfig | None = None
        if "email" in channels:
            to_addrs = (get("FQL_EMAIL_TO", "") or "").split(",")
            to_addrs = [a.strip() for a in to_addrs if a.strip()]
            if get("FQL_SMTP_HOST") and to_addrs:
                email_config = EmailConfig(
                    smtp_host=get("FQL_SMTP_HOST", "smtp.gmail.com") or "smtp.gmail.com",
                    smtp_port=int(get("FQL_SMTP_PORT", "587") or "587"),
                    username=get("FQL_SMTP_USER", "") or "",
                    password=get("FQL_SMTP_PASS", "") or "",
                    from_addr=get("FQL_EMAIL_FROM", "") or "",
                    to_addrs=to_addrs,
                    use_tls=get_bool("FQL_SMTP_TLS", True),
                )

        telegram_config: TelegramConfig | None = None
        if "telegram" in channels and get("FQL_TELEGRAM_TOKEN") and get("FQL_TELEGRAM_CHAT_ID"):
            telegram_config = TelegramConfig(
                bot_token=get("FQL_TELEGRAM_TOKEN", "") or "",
                chat_id=get("FQL_TELEGRAM_CHAT_ID", "") or "",
            )

        slack_config: SlackConfig | None = None
        if "slack" in channels and get("FQL_SLACK_WEBHOOK"):
            slack_config = SlackConfig(webhook_url=get("FQL_SLACK_WEBHOOK", "") or "")

        file_config = FileConfig(
            path=Path(get("FQL_ALERT_LOG", "data/alerts.log") or "data/alerts.log")
        )

        return cls(
            channels=channels,
            email_config=email_config,
            telegram_config=telegram_config,
            slack_config=slack_config,
            file_config=file_config,
        )

    # --------------------------------------------------------
    #  Dispatch
    # --------------------------------------------------------
    def dispatch(self, alert: Alert) -> dict[str, bool]:
        """Envía una alerta a todos los canales activos.

        Returns:
            Dict {canal: éxito}. Nunca lanza; los errores se registran
            como False.
        """
        results: dict[str, bool] = {}
        self._history.append(alert)

        for channel in self.channels:
            try:
                if channel == "console":
                    results["console"] = self._send_console(alert)
                elif channel == "file":
                    results["file"] = self._send_file(alert)
                elif channel == "email":
                    results["email"] = self._send_email(alert)
                elif channel == "telegram":
                    results["telegram"] = self._send_telegram(alert)
                elif channel == "slack":
                    results["slack"] = self._send_slack(alert)
                else:
                    results[channel] = False
            except Exception:
                results[channel] = False

        return results

    def dispatch_many(self, alerts: list[Alert]) -> dict[str, int]:
        """Envía varias alertas. Devuelve {canal: nº éxitos}."""
        counts: dict[str, int] = {}
        for alert in alerts:
            for channel, ok in self.dispatch(alert).items():
                counts.setdefault(channel, 0)
                if ok:
                    counts[channel] += 1
        return counts

    def history(self) -> list[Alert]:
        """Historial en memoria de las alertas enviadas."""
        return list(self._history)

    def clear_history(self) -> None:
        self._history.clear()

    # --------------------------------------------------------
    #  Canales internos
    # --------------------------------------------------------
    def _send_console(self, alert: Alert) -> bool:
        print(alert.format_text())
        return True

    def _send_file(self, alert: Alert) -> bool:
        path = self.file_config.path
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(alert.to_dict(), ensure_ascii=False)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        return True

    def _send_email(self, alert: Alert) -> bool:
        cfg = self.email_config
        if cfg is None:
            return False

        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[Fintech Quant Lab] {alert.title}"
        msg["From"] = cfg.from_addr
        msg["To"] = ", ".join(cfg.to_addrs)
        msg.attach(MIMEText(alert.format_text(), "plain", "utf-8"))

        # HTML simple
        html = (
            f"<h3>{alert.severity.value.upper()}: {alert.title}</h3>"
            f"<p>{alert.message}</p>"
            f"<pre>{alert.format_text()}</pre>"
        )
        msg.attach(MIMEText(html, "html", "utf-8"))

        if cfg.use_tls:
            context = ssl.create_default_context()
            with smtplib.SMTP(cfg.smtp_host, cfg.smtp_port) as server:
                server.starttls(context=context)
                if cfg.username:
                    server.login(cfg.username, cfg.password)
                server.sendmail(cfg.from_addr, cfg.to_addrs, msg.as_string())
        else:
            with smtplib.SMTP(cfg.smtp_host, cfg.smtp_port) as server:
                if cfg.username:
                    server.login(cfg.username, cfg.password)
                server.sendmail(cfg.from_addr, cfg.to_addrs, msg.as_string())
        return True

    def _send_telegram(self, alert: Alert) -> bool:
        cfg = self.telegram_config
        if cfg is None:
            return False

        url = f"https://api.telegram.org/bot{cfg.bot_token}/sendMessage"
        payload = {
            "chat_id": cfg.chat_id,
            "text": alert.format_text(),
            "parse_mode": "HTML",
        }
        data = json.dumps(payload).encode("utf-8")
        req = urlrequest.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlrequest.urlopen(req, timeout=10) as resp:
            return bool(200 <= resp.status < 300)

    def _send_slack(self, alert: Alert) -> bool:
        cfg = self.slack_config
        if cfg is None:
            return False

        emoji = {
            AlertSeverity.INFO: ":information_source:",
            AlertSeverity.WARNING: ":warning:",
            AlertSeverity.DANGER: ":rotating_light:",
            AlertSeverity.SUCCESS: ":white_check_mark:",
        }.get(alert.severity, ":bell:")

        payload = {
            "text": f"{emoji} *{alert.title}*",
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"{emoji} *{alert.title}*\n{alert.message}",
                    },
                },
                {
                    "type": "context",
                    "elements": [
                        {"type": "mrkdwn", "text": f"_{alert.timestamp}_"}
                    ],
                },
            ],
        }
        data = json.dumps(payload).encode("utf-8")
        req = urlrequest.Request(
            cfg.webhook_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlrequest.urlopen(req, timeout=10) as resp:
            return bool(200 <= resp.status < 300)


# ============================================================
#  Reglas predefinidas
# ============================================================
def default_backtest_rules() -> list[AlertRule]:
    """Reglas habituales para backtests."""
    return [
        AlertRule(
            metric="sharpe", comparator=">=", threshold=2.0,
            severity=AlertSeverity.SUCCESS,
            title="Sharpe excelente",
            message="El Sharpe del backtest supera 2.0. Revisar robustez.",
        ),
        AlertRule(
            metric="sharpe", comparator="<", threshold=0.0,
            severity=AlertSeverity.WARNING,
            title="Sharpe negativo",
            message="El Sharpe del backtest es negativo. Revisar estrategia.",
        ),
        AlertRule(
            metric="max_drawdown", comparator="<=", threshold=-0.30,
            severity=AlertSeverity.DANGER,
            title="Drawdown severo",
            message="El Max Drawdown supera el 30%. Riesgo elevado.",
        ),
        AlertRule(
            metric="n_trades", comparator="<", threshold=10,
            severity=AlertSeverity.WARNING,
            title="Pocas operaciones",
            message="Menos de 10 operaciones. Resultado poco fiable.",
        ),
        AlertRule(
            metric="win_rate", comparator=">=", threshold=0.60,
            severity=AlertSeverity.INFO,
            title="Win rate alto",
            message="Win rate por encima del 60%.",
        ),
    ]


def default_walkforward_rules() -> list[AlertRule]:
    """Reglas habituales para walk-forward."""
    return [
        AlertRule(
            metric="sharpe", comparator="<", threshold=0.5,
            severity=AlertSeverity.WARNING,
            title="Sharpe OOS bajo",
            message="El Sharpe out-of-sample es inferior a 0.5.",
        ),
    ]


def check_backtest_rules(
    metrics: dict[str, Any],
    rules: list[AlertRule] | None = None,
) -> list[Alert]:
    """Evalúa las reglas sobre las métricas de un backtest.

    Args:
        metrics: Dict de métricas (salida de `BacktestResult.metrics`).
        rules: Reglas a evaluar. Si None, usa `default_backtest_rules()`.

    Returns:
        Lista de alertas que han disparado.
    """
    if rules is None:
        rules = default_backtest_rules()
    return [r.to_alert(metrics) for r in rules if r.matches(metrics)]


def check_custom_rule(
    metrics: dict[str, Any],
    metric: str,
    comparator: Comparator,
    threshold: float,
    severity: AlertSeverity = AlertSeverity.INFO,
    title: str = "",
    message: str = "",
) -> Alert | None:
    """Evalúa una única regla ad-hoc."""
    rule = AlertRule(
        metric=metric,
        comparator=comparator,
        threshold=threshold,
        severity=severity,
        title=title,
        message=message,
    )
    return rule.to_alert(metrics) if rule.matches(metrics) else None


# ============================================================
#  Utilidades
# ============================================================
def list_channels(dispatcher: AlertDispatcher) -> list[str]:
    """Lista los canales activos en un dispatcher."""
    return list(dispatcher.channels)


def test_dispatcher(dispatcher: AlertDispatcher) -> dict[str, bool]:
    """Envía una alerta de prueba por todos los canales activos."""
    alert = Alert(
        title="Prueba de conexión",
        message="Esta es una alerta de prueba desde Fintech Quant Lab.",
        severity=AlertSeverity.INFO,
    )
    return dispatcher.dispatch(alert)
