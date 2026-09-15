"""Página de configuración y prueba de alertas."""
import os

import pandas as pd
import streamlit as st

from app.core.alerts import (
    Alert,
    AlertDispatcher,
    AlertSeverity,
    check_custom_rule,
    default_backtest_rules,
    list_channels,
    test_dispatcher,
)
from app.styles import callout, footer, hero, page_setup, section

page_setup("Alertas", "🔔")

hero(
    title="Alertas y Notificaciones",
    subtitle=(
        "Configura canales de notificación (consola, archivo, email, "
        "Telegram, Slack) y reglas que disparen avisos cuando las métricas "
        "de un backtest crucen umbrales."
    ),
    icon="🔔",
)

# ============================================================
#  Cargar configuración de secrets o env
# ============================================================
def _load_secrets() -> dict:
    """Carga st.secrets como dict si existe."""
    try:
        return dict(st.secrets)
    except Exception:
        return {}


secrets = _load_secrets()
dispatcher = AlertDispatcher.from_env(source=secrets if secrets else None)
active_channels = list_channels(dispatcher)


# ============================================================
#  Estado de configuración
# ============================================================
section("🔧 Canales configurados")

if not active_channels:
    callout("No hay canales activos. Configura al menos uno.", variant="warning")
else:
    channel_status = []
    for ch in active_channels:
        if ch == "email":
            ok = dispatcher.email_config is not None
        elif ch == "telegram":
            ok = dispatcher.telegram_config is not None
        elif ch == "slack":
            ok = dispatcher.slack_config is not None
        elif ch == "file":
            ok = True  # siempre disponible
        else:
            ok = True
        channel_status.append({
            "Canal": ch,
            "Configurado": "✅" if ok else "❌ (faltan credenciales)",
        })
    st.dataframe(
        pd.DataFrame(channel_status),
        use_container_width=True,
        hide_index=True,
    )

# ============================================================
#  Probar canales
# ============================================================
section("🧪 Probar canales")

callout(
    "Envía una alerta de prueba por todos los canales activos. Los errores "
    "se muestran como ❌ — revisa credenciales o conexión.",
    variant="info",
)

if st.button("📨 Enviar alerta de prueba", type="primary"):
    results = test_dispatcher(dispatcher)
    if not results:
        callout("No hay canales configurados.", variant="warning")
    else:
        rows = [{"Canal": k, "Resultado": "✅ OK" if v else "❌ Fallo"}
                for k, v in results.items()]
        st.dataframe(
            pd.DataFrame(rows), use_container_width=True, hide_index=True,
        )

# ============================================================
#  Reglas configurables
# ============================================================
section("📐 Reglas de alerta")

st.caption(
    "Las reglas se evalúan contra las métricas de un backtest. "
    "Puedes ver las reglas por defecto y probar una regla ad-hoc."
)

with st.expander("📋 Reglas por defecto"):
    rules = default_backtest_rules()
    rules_df = pd.DataFrame([
        {
            "Métrica": r.metric,
            "Condición": f"{r.comparator} {r.threshold}",
            "Severidad": r.severity.value,
            "Título": r.title,
        }
        for r in rules
    ])
    st.dataframe(rules_df, use_container_width=True, hide_index=True)

# ============================================================
#  Probar regla ad-hoc
# ============================================================
section("🧪 Probar regla ad-hoc")

st.caption(
    "Introduce métricas ficticias y una condición. Si se cumple, se "
    "enviará una alerta por los canales activos."
)

col1, col2 = st.columns(2)
with col1:
    test_metrics = {
        "sharpe": st.number_input("Sharpe", value=1.5, step=0.1, format="%.2f"),
        "max_drawdown": st.number_input(
            "Max Drawdown (negativo)", value=-0.15, step=0.01, format="%.2f",
        ),
        "total_return": st.number_input(
            "Total Return", value=0.25, step=0.01, format="%.2f",
        ),
        "n_trades": st.number_input("Nº trades", value=50, step=1),
    }

with col2:
    rule_metric = st.selectbox(
        "Métrica a evaluar",
        list(test_metrics.keys()),
    )
    rule_comparator = st.selectbox(
        "Comparador",
        [">", ">=", "<", "<=", "==", "!="],
    )
    rule_threshold = st.number_input(
        "Umbral", value=1.0, step=0.1, format="%.2f",
    )
    rule_severity = st.selectbox(
        "Severidad",
        [s.value for s in AlertSeverity],
    )

if st.button("🔍 Evaluar y enviar si se cumple"):
    alert = check_custom_rule(
        metrics=test_metrics,
        metric=rule_metric,
        comparator=rule_comparator,
        threshold=rule_threshold,
        severity=AlertSeverity(rule_severity),
    )
    if alert is None:
        callout(
            f"La condición <code>{rule_metric} {rule_comparator} {rule_threshold}</code> "
            "NO se cumple. No se envía alerta.",
            variant="info",
        )
    else:
        results = dispatcher.dispatch(alert)
        rows = [{"Canal": k, "Resultado": "✅" if v else "❌"}
                for k, v in results.items()]
        callout(
            f"Alerta enviada: <strong>{alert.title}</strong>",
            variant="success",
        )
        st.dataframe(
            pd.DataFrame(rows), use_container_width=True, hide_index=True,
        )

# ============================================================
#  Historial en memoria
# ============================================================
section("📜 Historial de esta sesión")

history = dispatcher.history()
if not history:
    callout("No se han enviado alertas en esta sesión.", variant="info")
else:
    rows = [
        {
            "Timestamp": a.timestamp,
            "Severidad": a.severity.value,
            "Título": a.title,
            "Mensaje": a.message,
            "Métrica": a.metric or "—",
            "Valor": a.value if a.value is not None else "—",
        }
        for a in history
    ]
    st.dataframe(
        pd.DataFrame(rows), use_container_width=True, hide_index=True,
    )

    if st.button("🗑️ Limpiar historial"):
        dispatcher.clear_history()
        st.rerun()

# ============================================================
#  Guía de configuración
# ============================================================
section("📖 Cómo configurar los canales")

with st.expander("🔐 Variables de entorno"):
    st.markdown(
        """
        Crea un archivo `.env` en la raíz del proyecto (excluido de git) o
        exporta las variables en tu shell:

        ```bash
        # Canales activos (separados por coma)
        FQL_ALERT_CHANNELS=console,file,telegram

        # Archivo de log
        FQL_ALERT_LOG=data/alerts.log

        # Email (SMTP)
        FQL_SMTP_HOST=smtp.gmail.com
        FQL_SMTP_PORT=587
        FQL_SMTP_USER=tu@email.com
        FQL_SMTP_PASS=tu_app_password
        FQL_SMTP_TLS=true
        FQL_EMAIL_FROM=tu@email.com
        FQL_EMAIL_TO=destino1@email.com,destino2@email.com

        # Telegram
        FQL_TELEGRAM_TOKEN=123456:ABC-DEF...
        FQL_TELEGRAM_CHAT_ID=123456789

        # Slack
        FQL_SLACK_WEBHOOK=https://hooks.slack.com/services/...
