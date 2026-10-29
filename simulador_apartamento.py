"""
Proyecto de simulación: ¿es rentable invertir en un apartamento para arrendar?

Ejecutar con:
    streamlit run simulador_apartamento.py

Estructura del archivo (sigue las partes de la actividad):
    Parte A. Generadores de variables aleatorias (Bernoulli y uniforme).
    Parte B. Simulador mes a mes de una trayectoria del inmueble.
    Parte C. Réplicas independientes y análisis estadístico (media, desviación, IC t-Student).
    Parte D. Aplicación interactiva en Streamlit.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
import streamlit as st

# Valores iniciales de la tabla "Condiciones iniciales" de la actividad
PARAMETROS_INICIALES = {
    "valor_compra": 250_000_000,
    "canon": 1_500_000,
    "administracion": 250_000,
    "predial": 1_200_000,
    "p_desocupacion": 0.10,
    "p_mantenimiento": 0.20,
    "costo_min": 100_000,
    "costo_max": 500_000,
    "horizonte": 12,
    "replicas": 30,
}


# ==========================================================
# PARTE A. Generación de variables aleatorias
# ==========================================================

def generar_uniforme_01(rng):
    """U ~ Uniforme(0, 1). Es la base de los otros tres generadores."""
    return rng.random()


def generar_bernoulli(rng, p):
    """Devuelve 1 con probabilidad p y 0 con probabilidad 1 - p (método de la transformada inversa)."""
    return 1 if generar_uniforme_01(rng) < p else 0


def generar_desocupacion(rng, p_desocupacion):
    """(a) ¿El inquilino se va al finalizar un mes ocupado? Bernoulli(p_desocupacion)."""
    return generar_bernoulli(rng, p_desocupacion)


def generar_mantenimiento(rng, p_mantenimiento):
    """(b) ¿Hay mantenimiento este mes? Bernoulli(p_mantenimiento)."""
    return generar_bernoulli(rng, p_mantenimiento)


def generar_costo_mantenimiento(rng, costo_min, costo_max):
    """(c) Costo = Cmín + (Cmáx - Cmín) * U, con U ~ Uniforme(0, 1)."""
    u = generar_uniforme_01(rng)
    return costo_min + (costo_max - costo_min) * u


# ==========================================================
# PARTE B. Simulador mes a mes
# ==========================================================

def simular_trayectoria(params, rng):
    """
    Simula una trayectoria del apartamento durante el horizonte.

    Reglas:
    - El mes 1 empieza arrendado.
    - Al terminar un mes ocupado se sortea la desocupación. Si ocurre, el mes siguiente
      queda vacío y el apartamento se arrienda otra vez al comenzar el mes posterior.
    - Cada mes se sortea si hay mantenimiento y, si lo hay, su costo uniforme.
    - La administración se paga todos los meses y el predial en el mes 12, 24, 36...
    """
    filas = []
    ocupado = True
    flujo_acumulado = 0.0

    for mes in range(1, params["horizonte"] + 1):
        arriendo = params["canon"] if ocupado else 0

        hay_mantenimiento = generar_mantenimiento(rng, params["p_mantenimiento"])
        mantenimiento = (
            generar_costo_mantenimiento(rng, params["costo_min"], params["costo_max"])
            if hay_mantenimiento
            else 0.0
        )

        administracion = params["administracion"]
        predial = params["predial"] if mes % 12 == 0 else 0
        otros_gastos = administracion + predial

        flujo_neto = arriendo - mantenimiento - administracion - predial
        flujo_acumulado += flujo_neto

        # Evento al final del mes: solo un mes ocupado puede terminar en desocupación
        if ocupado:
            se_desocupa = generar_desocupacion(rng, params["p_desocupacion"])
            desocupa_txt = "Sí" if se_desocupa else "No"
        else:
            se_desocupa = 0
            desocupa_txt = "No aplica"

        filas.append(
            {
                "Mes": mes,
                "Estado": "Ocupado" if ocupado else "Desocupado",
                "Arriendo": arriendo,
                "Mantenimiento": mantenimiento,
                "Administración": administracion,
                "Predial": predial,
                "Otros gastos": otros_gastos,
                "Flujo neto": flujo_neto,
                "Flujo acumulado": flujo_acumulado,
                "¿Se desocupa al final?": desocupa_txt,
            }
        )

        # Estado del mes siguiente
        if ocupado:
            ocupado = not se_desocupa
        else:
            ocupado = True  # tras un mes vacío se vuelve a arrendar

    return pd.DataFrame(filas)


def resumir_trayectoria(trayectoria):
    """Totales de una réplica: meses arrendados, ingresos, gastos y flujo neto acumulado."""
    ingresos = trayectoria["Arriendo"].sum()
    gastos = trayectoria["Mantenimiento"].sum() + trayectoria["Otros gastos"].sum()
    return {
        "Meses arrendados": int((trayectoria["Estado"] == "Ocupado").sum()),
        "Meses con mantenimiento": int((trayectoria["Mantenimiento"] > 0).sum()),
        "Ingresos totales": ingresos,
        "Gastos totales": gastos,
        "Flujo neto acumulado": ingresos - gastos,
    }


# ==========================================================
# PARTE C. Réplicas y análisis estadístico
# ==========================================================

def ejecutar_replicas(params, n_replicas, semilla=None):
    """Corre n réplicas independientes con un mismo generador (semilla fija = resultados repetibles)."""
    rng = np.random.default_rng(semilla)
    resumen = []
    trayectorias = []
    for r in range(1, n_replicas + 1):
        tray = simular_trayectoria(params, rng)
        trayectorias.append(tray)
        resumen.append({"Réplica": r, **resumir_trayectoria(tray)})
    return pd.DataFrame(resumen), trayectorias


def analizar_resultados(resumen, params, confianza=0.95):
    """Media, desviación, IC con t de Student y medidas de dispersión del flujo neto acumulado."""
    flujos = resumen["Flujo neto acumulado"].to_numpy()
    n = len(flujos)
    media = flujos.mean()
    desv = flujos.std(ddof=1)  # desviación muestral
    error_estandar = desv / np.sqrt(n)
    t_critico = stats.t.ppf(1 - (1 - confianza) / 2, df=n - 1)
    margen = t_critico * error_estandar

    anios = params["horizonte"] / 12
    valor = params["valor_compra"]

    return {
        "n": n,
        "media": media,
        "desv": desv,
        "cv": desv / abs(media) if media != 0 else np.nan,
        "minimo": flujos.min(),
        "maximo": flujos.max(),
        "p5": np.percentile(flujos, 5),
        "p95": np.percentile(flujos, 95),
        "t_critico": t_critico,
        "margen": margen,
        "ic_inf": media - margen,
        "ic_sup": media + margen,
        "meses_promedio": resumen["Meses arrendados"].mean(),
        "ingresos_promedio": resumen["Ingresos totales"].mean(),
        "gastos_promedio": resumen["Gastos totales"].mean(),
        "rentabilidad_anual": (media / anios) / valor if valor > 0 else np.nan,
        "rentabilidad_ic_inf": ((media - margen) / anios) / valor if valor > 0 else np.nan,
        "rentabilidad_ic_sup": ((media + margen) / anios) / valor if valor > 0 else np.nan,
        "replicas_negativas": int((flujos < 0).sum()),
    }


def flujo_maximo_posible(params):
    """Flujo si nunca se desocupa ni hay mantenimiento: sirve de referencia en las conclusiones."""
    h = params["horizonte"]
    return h * (params["canon"] - params["administracion"]) - (h // 12) * params["predial"]


# ==========================================================
# PARTE D. Aplicación Streamlit
# ==========================================================

def pesos(x):
    """Formato colombiano: $1.234.567"""
    signo = "-" if x < 0 else ""
    return f"{signo}${abs(x):,.0f}".replace(",", ".")


def pct(x, dec=2):
    return f"{x * 100:.{dec}f} %".replace(".", ",")


def tabla_en_pesos(df, columnas):
    out = df.copy()
    for c in columnas:
        out[c] = out[c].apply(pesos)
    return out


def leer_parametros():
    """Barra lateral: todo lo que la actividad pide modificar sin editar el código."""
    p0 = PARAMETROS_INICIALES
    sb = st.sidebar
    sb.header("Parámetros")

    with sb.form("parametros"):
        st.subheader("Valores del inmueble")
        valor_compra = st.number_input("Valor de compra ($)", min_value=1, value=p0["valor_compra"], step=5_000_000)
        canon = st.number_input("Canon mensual ($)", min_value=0, value=p0["canon"], step=50_000)
        administracion = st.number_input("Administración mensual ($)", min_value=0, value=p0["administracion"], step=10_000)
        predial = st.number_input("Predial anual ($)", min_value=0, value=p0["predial"], step=50_000)

        st.subheader("Probabilidades mensuales")
        p_desocupacion = st.slider(
            "Desocupación (%)", 0.0, 100.0, p0["p_desocupacion"] * 100, 0.5,
            help="Probabilidad de que el inquilino se vaya al terminar un mes ocupado.",
        ) / 100
        p_mantenimiento = st.slider(
            "Mantenimiento (%)", 0.0, 100.0, p0["p_mantenimiento"] * 100, 0.5,
            help="Probabilidad de que en un mes haya que hacer un arreglo.",
        ) / 100

        st.subheader("Costo de mantenimiento")
        costo_min = st.number_input("Costo mínimo ($)", min_value=0, value=p0["costo_min"], step=10_000)
        costo_max = st.number_input("Costo máximo ($)", min_value=0, value=p0["costo_max"], step=10_000)

        st.subheader("Simulación")
        horizonte = st.number_input("Horizonte (meses)", min_value=1, max_value=600, value=p0["horizonte"], step=1)
        replicas = st.number_input("Número de réplicas", min_value=2, max_value=20_000, value=p0["replicas"], step=10)
        semilla_fija = st.checkbox("Usar semilla fija (resultados repetibles)", value=True)
        semilla = st.number_input("Semilla", min_value=0, value=2026, step=1, disabled=False)

        st.subheader("Comparación")
        tasa_alternativa = st.number_input(
            "Tasa anual de una inversión alternativa, ej. CDT (%)",
            min_value=0.0, max_value=100.0, value=9.0, step=0.25,
            help="Ajústala a la tasa vigente que te ofrezca un banco. Sirve para juzgar si el arriendo compensa.",
        ) / 100

        st.form_submit_button("Simular", type="primary", width="stretch")

    params = {
        "valor_compra": int(valor_compra),
        "canon": int(canon),
        "administracion": int(administracion),
        "predial": int(predial),
        "p_desocupacion": p_desocupacion,
        "p_mantenimiento": p_mantenimiento,
        "costo_min": int(costo_min),
        "costo_max": int(costo_max),
        "horizonte": int(horizonte),
    }
    return params, int(replicas), (int(semilla) if semilla_fija else None), tasa_alternativa


def mostrar_resultados(resumen, a, params, tasa_alt):
    st.header("Resultados de las réplicas")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Flujo neto promedio", pesos(a["media"]))
    c2.metric("Desviación estándar", pesos(a["desv"]))
    c3.metric("Meses arrendados promedio", f"{a['meses_promedio']:.2f} de {params['horizonte']}".replace(".", ",", 1))
    c4.metric("Rentabilidad anual promedio", pct(a["rentabilidad_anual"]))

    st.info(
        f"**Intervalo de confianza del 95 % para el flujo neto promedio** (t de Student, "
        f"{a['n'] - 1} grados de libertad, t = {a['t_critico']:.4f}):  \n"
        f"[ {pesos(a['ic_inf'])} ; {pesos(a['ic_sup'])} ]   (margen de error ± {pesos(a['margen'])})"
    )

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Ingresos promedio", pesos(a["ingresos_promedio"]))
    c6.metric("Gastos promedio", pesos(a["gastos_promedio"]))
    c7.metric("Coeficiente de variación", pct(a["cv"], 1))
    c8.metric("Réplicas con flujo negativo", f"{a['replicas_negativas']} de {a['n']}")

    # Histograma con la media y el intervalo de confianza
    fig = px.histogram(
        resumen, x="Flujo neto acumulado", nbins=min(30, max(8, a["n"] // 3)),
        labels={"Flujo neto acumulado": "Flujo neto acumulado ($)", "count": "Réplicas"},
        color_discrete_sequence=["#4C78A8"],
    )
    fig.add_vrect(x0=a["ic_inf"], x1=a["ic_sup"], fillcolor="orange", opacity=0.25, line_width=0,
                  annotation_text="IC 95 %", annotation_position="top left")
    fig.add_vline(x=a["media"], line_dash="dash", line_color="red", annotation_text="Media")
    fig.update_layout(title="Histograma del flujo neto acumulado", yaxis_title="Número de réplicas", bargap=0.05)
    st.plotly_chart(fig, width="stretch")

    col_a, col_b = st.columns(2)
    with col_a:
        # Qué explica la variabilidad: los meses vacíos
        fig2 = px.box(
            resumen, x="Meses arrendados", y="Flujo neto acumulado", points="all",
            labels={"Flujo neto acumulado": "Flujo neto acumulado ($)"},
            title="Flujo neto según meses arrendados",
        )
        st.plotly_chart(fig2, width="stretch")
    with col_b:
        # Estabilidad de la media al aumentar réplicas
        media_corrida = resumen["Flujo neto acumulado"].expanding().mean()
        fig3 = go.Figure(go.Scatter(x=resumen["Réplica"], y=media_corrida, mode="lines"))
        fig3.add_hline(y=a["media"], line_dash="dot", line_color="red")
        fig3.update_layout(title="Media acumulada del flujo según el número de réplicas",
                           xaxis_title="Réplica", yaxis_title="Media del flujo ($)")
        st.plotly_chart(fig3, width="stretch")

    st.subheader("Tabla de réplicas")
    st.dataframe(
        tabla_en_pesos(resumen, ["Ingresos totales", "Gastos totales", "Flujo neto acumulado"]),
        width="stretch", hide_index=True, height=360,
    )

    st.subheader("Indicadores estadísticos")
    indicadores = pd.DataFrame(
        [
            ("Número de réplicas (n)", f"{a['n']}"),
            ("Media del flujo neto acumulado", pesos(a["media"])),
            ("Desviación estándar muestral", pesos(a["desv"])),
            ("Error estándar (s / √n)", pesos(a["desv"] / np.sqrt(a["n"]))),
            ("Valor t (95 %)", f"{a['t_critico']:.4f}"),
            ("IC 95 % para la media", f"[{pesos(a['ic_inf'])} ; {pesos(a['ic_sup'])}]"),
            ("Mínimo / Máximo", f"{pesos(a['minimo'])} / {pesos(a['maximo'])}"),
            ("Percentil 5 / Percentil 95", f"{pesos(a['p5'])} / {pesos(a['p95'])}"),
            ("Meses arrendados promedio", f"{a['meses_promedio']:.2f}"),
            ("Rentabilidad anual (media e IC)",
             f"{pct(a['rentabilidad_anual'])}  [{pct(a['rentabilidad_ic_inf'])} ; {pct(a['rentabilidad_ic_sup'])}]"),
            ("Tasa de la alternativa usada para comparar", pct(tasa_alt)),
        ],
        columns=["Indicador", "Valor"],
    )
    st.table(indicadores.set_index("Indicador"))


def mostrar_trayectoria(trayectorias):
    st.header("Detalle mes a mes de una réplica")
    n = len(trayectorias)
    r = st.number_input("Réplica a mostrar", min_value=1, max_value=n, value=1, step=1)
    tray = trayectorias[int(r) - 1]
    st.dataframe(
        tabla_en_pesos(tray, ["Arriendo", "Mantenimiento", "Administración", "Predial",
                              "Otros gastos", "Flujo neto", "Flujo acumulado"]),
        width="stretch", hide_index=True,
    )
    tot = resumir_trayectoria(tray)
    st.write(
        f"Meses arrendados: **{tot['Meses arrendados']}** · Ingresos: **{pesos(tot['Ingresos totales'])}** · "
        f"Gastos: **{pesos(tot['Gastos totales'])}** · Flujo neto acumulado: **{pesos(tot['Flujo neto acumulado'])}**"
    )
    st.caption("Otros gastos = administración + predial. Flujo neto = arriendo − mantenimiento − administración − predial.")


def mostrar_conclusiones(resumen, a, params, tasa_alt):
    st.header("Conclusiones")
    h = params["horizonte"]
    anios = h / 12
    techo = flujo_maximo_posible(params)
    meses_perdidos = h - a["meses_promedio"]
    alternativa = params["valor_compra"] * tasa_alt * anios

    st.markdown("#### Variabilidad entre réplicas")
    corr = resumen["Meses arrendados"].corr(resumen["Flujo neto acumulado"])
    st.write(
        f"En las {a['n']} réplicas el flujo neto acumulado fue de {pesos(a['media'])} en promedio, "
        f"con una desviación estándar de {pesos(a['desv'])} (coeficiente de variación de {pct(a['cv'], 1)}). "
        f"El peor caso dejó {pesos(a['minimo'])} y el mejor {pesos(a['maximo'])}. "
        f"Si el apartamento nunca se desocupara ni necesitara arreglos, el flujo sería {pesos(techo)}; "
        f"la diferencia con la media sale de los meses vacíos (en promedio {meses_perdidos:.2f} por réplica) "
        f"y de los mantenimientos."
    )
    if not np.isnan(corr):
        st.write(
            f"La correlación entre meses arrendados y flujo neto es {corr:.2f}. Cada mes vacío cuesta el canon "
            f"completo ({pesos(params['canon'])}) mientras la administración se sigue pagando, así que la "
            f"desocupación pesa mucho más que el mantenimiento, que en promedio cuesta "
            f"{pesos((params['costo_min'] + params['costo_max']) / 2)} y aparece en el "
            f"{pct(params['p_mantenimiento'], 0)} de los meses."
        )
    st.write(
        f"El intervalo de confianza del 95 % para la media va de {pesos(a['ic_inf'])} a {pesos(a['ic_sup'])}. "
        f"Ese intervalo describe dónde está el promedio, no lo que le puede pasar al inversionista en un año "
        f"concreto: el 90 % central de las réplicas quedó entre {pesos(a['p5'])} y {pesos(a['p95'])}. "
        f"Con más réplicas el intervalo se estrecha porque el error estándar baja con √n."
    )

    st.markdown("#### ¿Conviene la inversión?")
    st.write(
        f"Sobre un valor de compra de {pesos(params['valor_compra'])}, el flujo promedio equivale a una "
        f"rentabilidad anual de {pct(a['rentabilidad_anual'])} (IC 95 %: {pct(a['rentabilidad_ic_inf'])} a "
        f"{pct(a['rentabilidad_ic_sup'])}). Ese mismo capital a una tasa de {pct(tasa_alt)} anual "
        f"produciría unos {pesos(alternativa)} en {h} meses."
    )
    if a["ic_sup"] < alternativa:
        st.error(
            "Con estos supuestos el arriendo rinde menos que la alternativa, y la diferencia es significativa: "
            "todo el intervalo de confianza queda por debajo. Solo por flujo de caja, no conviene."
        )
    elif a["ic_inf"] > alternativa:
        st.success(
            "El arriendo supera a la alternativa incluso en el extremo inferior del intervalo de confianza. "
            "Solo por flujo de caja, la inversión conviene."
        )
    else:
        st.warning(
            "El intervalo de confianza contiene el rendimiento de la alternativa, así que con estas réplicas "
            "no se puede afirmar que una sea mejor que la otra. Conviene aumentar las réplicas o revisar los supuestos."
        )
    if a["replicas_negativas"] == 0:
        st.write("Ninguna réplica terminó en pérdida: el riesgo aquí no es perder plata, es ganar poco.")

    st.markdown("#### Supuestos que limitan la conclusión")
    st.write(
        "- No se incluye la valorización del inmueble. En finca raíz suele ser una parte grande del retorno, "
        "así que la rentabilidad calculada solo mide el arriendo.\n"
        "- No hay crédito hipotecario: se supone compra de contado. Con crédito, los intereses reducirían el flujo.\n"
        "- No se cuentan impuesto de renta, gastos notariales, registro ni comisión inmobiliaria.\n"
        "- Canon, administración y predial se mantienen fijos; a 12 meses es razonable, a varios años no "
        "(el canon y la administración se reajustan cada año).\n"
        "- La desocupación dura exactamente un mes. En la práctica conseguir inquilino puede tardar más, "
        "y eso bajaría el flujo."
    )


def main():
    st.set_page_config(page_title="Simulador apartamento para arrendar", layout="wide")
    st.title("¿Es rentable invertir en un apartamento para arrendar?")
    st.caption(
        "Simulación Monte Carlo de ingresos, gastos y flujo neto. Cambia los parámetros en la barra "
        "lateral y presiona Simular."
    )

    params, n_replicas, semilla, tasa_alt = leer_parametros()

    if params["costo_min"] > params["costo_max"]:
        st.error("El costo mínimo de mantenimiento no puede ser mayor que el máximo. Corrígelo para simular.")
        st.stop()
    if n_replicas < 30:
        st.warning("La actividad pide al menos 30 réplicas. Con menos, el intervalo de confianza es poco preciso.")

    resumen, trayectorias = ejecutar_replicas(params, n_replicas, semilla)
    a = analizar_resultados(resumen, params)

    if semilla is None:
        st.caption("Semilla aleatoria: cada vez que presiones Simular saldrán resultados distintos.")
    else:
        st.caption(f"Semilla {semilla}: los resultados se repiten mientras no cambies los parámetros.")

    tab1, tab2, tab3 = st.tabs(["Resultados y estadística", "Trayectoria mes a mes", "Conclusiones"])
    with tab1:
        mostrar_resultados(resumen, a, params, tasa_alt)
    with tab2:
        mostrar_trayectoria(trayectorias)
    with tab3:
        mostrar_conclusiones(resumen, a, params, tasa_alt)


if __name__ == "__main__":
    main()
