# Interconnect — Predicción de fuga de clientes

![Python](https://img.shields.io/badge/Python-3.11-2E3C2C?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-4A5E48?style=flat-square)
![LightGBM](https://img.shields.io/badge/LightGBM-4.7-8A9E80?style=flat-square)
![AUC-ROC](https://img.shields.io/badge/AUC--ROC%20en%20prueba-0.8421-960018?style=flat-square)

Modelo que ordena a los clientes de un operador de telecomunicaciones por riesgo de cancelación y calcula a cuántos conviene contactar para que la campaña de retención gane dinero en lugar de perderlo.

**→ [Lee el caso de estudio](https://he1109-ci3l0.github.io/interconnect-fuga-clientes/)**

---

## Resultado

Medido **una sola vez** sobre 1,409 clientes que el modelo nunca había visto:

| Métrica | Resultado | Lectura |
|---|---|---|
| AUC-ROC | **0.8421** | Entre un cliente que se fue y uno que se quedó, el modelo acierta cuál es cuál en 84 de cada 100 casos |
| Recall | **0.7968** | De cada 100 cancelaciones, identifica 80 a tiempo |
| Precisión | 0.5256 | De cada 100 clientes señalados, 53 se iban de verdad |

El mínimo exigido para este problema era 0.75 de AUC-ROC. El modelo constante, que predice que nadie se va, obtiene 0.5000 de AUC-ROC y **0 de Recall**: acierta el 73.5 % de las veces y no detecta a un solo cliente en riesgo. Esa comparación es la que justifica no usar exactitud como métrica.

**Modelo elegido:** regresión logística con ajuste de pesos (`C=1.6307`, `solver=liblinear`, `class_weight=balanced`).

---

## Valor de negocio

**El problema en dinero.** De los 7,043 clientes del registro, 1,869 cancelaron: el 26.5%. Facturaban **139,130.85 al mes**, el 30.5% de los 456,116.60 que suman los cargos mensuales de todo el registro.

**A cuántos contactar.** Ordenando por riesgo, atender al **30 %** de la cartera alcanza al **66 %** de quienes se van. Sin criterio, ese mismo esfuerzo encontraría el 30 %.

**Cuánto vale decidir con el modelo.** El umbral de contacto se elige maximizando el valor neto esperado sobre el conjunto de validación. En el escenario intermedio —20 % de aceptación, promoción de un mes de servicio— el umbral óptimo es **0.73**:

| Decisión | Clientes contactados | Valor neto |
|---|---|---|
| Contactar según el modelo | 310 de 1,409 | **15,216** |
| Ofrecer la promoción a todos | 1,409 | -37,716 |
| No hacer nada | 0 | 0 |

**La diferencia entre decidir con el modelo y decidir sin él es de 52,932** en ese escenario. Y ofrecer promociones a toda la cartera **produce pérdida en 7 de los 9 escenarios** evaluados, con cifras que llegan a −177,136: no saber a quién dirigir la oferta cuesta dinero.

En el escenario más adverso —10 % de aceptación, promoción de dos meses— ningún umbral produce valor positivo y la recomendación es **no lanzar la campaña**.

**Segmentos y acción:**

| Segmento | Tasa de baja | Acción propuesta |
|---|---|---|
| Contrato mes a mes | 42.7% | Incentivo de migración a contrato anual con meses bonificados |
| Menos de tres meses de antigüedad | 58.4% | Programa de bienvenida con contacto en el primer y segundo mes |
| Pago con cheque electrónico | 45.3% | Migración a domiciliación con un beneficio asociado |
| Fibra óptica | 41.9% | Empaquetar soporte técnico o antivirus antes de descontar |

**Siguiente paso:** un piloto de un trimestre con grupo de control. Dividir la lista de riesgo en dos mitades al azar, ofrecer la promoción solo a una y comparar la permanencia. Es lo único que convierte los supuestos de este análisis en cifras propias de la empresa.

---

## Comparación de modelos

Baseline, sin ninguna técnica de balanceo (AUC-ROC y Recall en validación):

| Modelo | AUC-ROC | Recall |
|---|---|---|
| Modelo constante | 0.5 | 0.0 |
| Regresión logística | 0.8356 | 0.5749 |
| Bosque aleatorio | 0.8157 | 0.5107 |
| LightGBM | 0.8228 | 0.5455 |

Tras optimizar hiperparámetros con `RandomizedSearchCV` y validación cruzada estratificada:

| Candidato | AUC-ROC | Recall | Precisión |
|---|---|---|---|
| Regresión logística · baseline | 0.8356 | 0.5749 | 0.6305 |
| Regresión logística · optimizada | 0.8356 | 0.5749 | 0.6324 |
| Regresión logística · optimizada con ajuste de pesos | 0.8354 | 0.8075 | 0.4855 |
| Bosque aleatorio · optimizado | 0.8365 | 0.7968 | 0.4958 |
| LightGBM · optimizado | 0.8374 | 0.8021 | 0.491 |

La diferencia entre el mejor modelo de árboles y la regresión logística es de 0.0020, mientras que la variabilidad entre particiones de la validación cruzada es de 0.0163: ocho veces mayor. Es un empate técnico, y en el empate pesan el Recall más alto, la interpretabilidad de los coeficientes y el costo de operación.

---

## Hallazgos

1. **El contrato mes a mes concentra el 88.5% de todas las cancelaciones**, con una tasa de 42.7% frente al 2.8% del contrato a dos años.
2. **El riesgo se decide en los primeros meses**: 58.4% de baja entre quienes llevan menos de tres meses, contra 9.6% de quienes llevan cuatro años o más.
3. **El cheque electrónico y la fibra óptica son focos de fuga**: 45.3% y 41.9%. No son causas: la fibra es el servicio más caro y atrae a clientes sensibles al precio.
4. **La antigüedad domina el modelo.** Desordenarla cuesta 0.1697 de AUC-ROC; la segunda variable más influyente cuesta 0.0417.
5. **Hay variables que no informan.** El género separa 0.8 puntos porcentuales y tener línea telefónica, 1.8. Están en los datos y no aportan.
6. **Equilibrar las clases no mejora el modelo, mueve el umbral.** El AUC-ROC apenas cambia entre tratamientos (0.8356 / 0.8354 / 0.8344) mientras el Recall salta de 0.5749 a 0.8075. De ahí se sigue que el umbral es una decisión de negocio y no una preferencia técnica.

---

## Estructura

```
interconnect-fuga-clientes/
├── notebooks/
│   └── interconnect_fuga_clientes.ipynb   análisis completo: plan, código, informe y capa de negocio
├── figures/                               las nueve figuras del notebook en PNG a 150 ppp
├── resultados/
│   └── metricas.json                      fuente única de todos los números del proyecto
├── assets/fuentes/                        tipografías .ttf registradas en matplotlib sin instalarlas
├── docs/
│   └── index.html                         caso de estudio publicado en GitHub Pages
├── environment.yml
└── README.md
```

Los datos de origen **no se versionan**. Se descargan del [archivo comprimido original](https://practicum-content.s3.us-west-1.amazonaws.com/new-markets/Spreadsheets_for_DA/datasets/final_provider.zip) y se colocan en una carpeta `data/` en la raíz.

---

## Metodología

1. **Unión** de los cuatro archivos por `customerID`, con verificación de que la llave es única y de que el número de clientes no cambia.
2. **Variable objetivo** construida desde la fecha de término del contrato: una fecha es baja, el texto `No` es permanencia.
3. **Antigüedad en meses**, medida hasta la cancelación para quien se fue y hasta la fecha de corte (2020-02-01) para quien sigue activo. La fecha de término se elimina del conjunto de entrada, con comprobación explícita.
4. **Ausencia de servicio** codificada como categoría propia, no como dato faltante.
5. **Codificación doble**: One-Hot para el modelo lineal, ordinal para los de árboles, dentro de un `Pipeline` para que el ajuste ocurra solo con datos de entrenamiento.
6. **División estratificada** 60 / 20 / 20 con semilla fija; la proporción de bajas se verifica en las tres partes.
7. **Baseline sin balanceo**, después ajuste de pesos y sobremuestreo, y búsqueda de hiperparámetros con `scoring="roc_auc"`.
8. **Prueba evaluada una sola vez**, con el modelo ya elegido. El umbral de operación se elige en validación, nunca en prueba.

---

## Reproducción

```bash
conda env create -f environment.yml
conda activate interconnect-fuga-clientes
python -m ipykernel install --user --name interconnect-fuga-clientes --display-name "Python (interconnect-fuga-clientes)"
jupyter lab notebooks/interconnect_fuga_clientes.ipynb
```

El notebook busca los datos primero en `/datasets/final_provider/` y, si no existe, en `../data/`. Se ejecuta completo de principio a fin en cualquiera de las dos rutas y tarda unos cinco minutos.

---

## Salvedades

- `environment.yml` se exportó del entorno real del proyecto pero **no se ha probado creando un entorno limpio desde cero**.
- El requisito mínimo es scikit-learn 1.2, por el parámetro `sparse_output` del codificador One-Hot.
- El modelo entregado **no está calibrado**: el ajuste de pesos desplaza sus probabilidades hacia arriba, de modo que sirven para ordenar clientes por riesgo, no para leerse como probabilidad absoluta.
- Las cifras de dinero dependen de tres supuestos declarados —tasa de aceptación, costo de la promoción y meses adicionales de permanencia— que la empresa no proporciona.
- El repositorio todavía no incluye módulos en `src/` ni el modelo serializado en `models/`.

---

## Créditos

**Belem Cisneros Díaz** · [github.com/he1109-ci3l0](https://github.com/he1109-ci3l0)

Datos proporcionados por TripleTen para el proyecto final del bootcamp de Ciencia de Datos.
