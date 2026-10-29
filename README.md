# Simulador: ¿es rentable un apartamento para arrendar?

Simulación Monte Carlo de los ingresos, gastos y flujo neto de un apartamento durante 12 meses, con una app en Streamlit para cambiar los parámetros y ver los resultados.

## Cómo correrlo en local

```
pip install -r requirements.txt
streamlit run simulador_apartamento.py
```

## Qué hace cada parte del código

- **Parte A:** generadores Bernoulli para la desocupación y el mantenimiento, y costo de mantenimiento con `Cmín + (Cmáx - Cmín) * U`.
- **Parte B:** `simular_trayectoria()` recorre el horizonte mes a mes y registra estado, arriendo, mantenimiento, administración, predial, flujo neto y flujo acumulado.
- **Parte C:** `ejecutar_replicas()` y `analizar_resultados()` calculan la media, la desviación estándar, los meses arrendados promedio y el intervalo de confianza del 95 % con t de Student.
- **Parte D:** la app tiene tres pestañas: resultados y estadística, trayectoria mes a mes y conclusiones.

Con la semilla fija (2026 por defecto) los resultados se pueden repetir.
