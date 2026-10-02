# Defiende el servidor

## Guía conceptual · Datos, experimentos, modelos y pipelines

**Curso:** MLOps  
**Fecha:** 2 de octubre de 2026  
**Uso:** Académico  
**Institución:** Universidad Externado de Colombia  
**Unidad académica:** Departamento de Matemáticas  
**Docente:** Julián Zuluaga · Docente

---

## El reto: detectar ataques y defender tus decisiones

Imagina que un equipo de seguridad recibe miles de conexiones de red y necesita identificar cuáles merecen una alerta. Tu equipo debe construir una solución de machine learning y demostrar cómo llegó a ella.

El resultado debe permitir entender qué datos se usaron, qué alternativas se probaron, por qué se eligió un modelo y cómo repetir el proceso.

## Punto de partida

Trabaja con **UNSW-NB15**, un dataset de conexiones normales y ataques de laboratorio. Investiga su origen, sus variables y sus condiciones de uso en la fuente de UNSW. Obtén los datos por tu cuenta y documenta la versión utilizada.

El objetivo principal es distinguir tráfico normal de ataque. Como ampliación, puedes intentar reconocer la familia del ataque. Construye tu propio repositorio e implementación; investiga la documentación de las herramientas cuando lo necesites.

**Pregunta central:** ¿puede otra persona comprender y reproducir tu solución con la evidencia que entregas?

---

# Las decisiones que debes tomar

## 1. ¿Qué puedes aprender de estos datos?

Explora las clases, los valores ausentes y las repeticiones. Distingue las variables disponibles al predecir de las respuestas que intentas aprender. Investiga si el identificador tiene significado para el problema o solo refleja cómo se organizó el archivo.

Separa datos para entrenar, comparar alternativas y evaluar al final. Explica cómo evitarás que ejemplos repetidos o información de la respuesta hagan parecer mejor a un modelo. Documenta cualquier limpieza: también cambia la población que estás estudiando.

## 2. ¿Qué merece convertirse en un experimento?

Empieza con una referencia sencilla. Prueba al menos cuatro configuraciones, incluyendo dos tipos de modelo. Cada experimento debe responder una pregunta: por ejemplo, si aumentar complejidad mejora el resultado, si equilibrar clases cambia los errores o si usar más datos compensa su costo.

Escribe una hipótesis antes de entrenar. Para comparar, conserva las mismas particiones y cambia una decisión a la vez cuando sea posible. Un resultado que contradice tu hipótesis también aporta aprendizaje.

## 3. ¿Qué significa que un modelo sea útil?

Un ataque ignorado y una falsa alarma tienen consecuencias distintas. Examina cuántos ataques detectas (**recall**), cuántas alertas son correctas (**precision**) y cuántas conexiones normales alarmas (**tasa de falsos positivos**). Usa también una medida de equilibrio, como **macro-F1**; una accuracy alta puede ocultar errores.

Elige modelo y umbral de alerta con validación. Justifica el equilibrio entre detectar ataques y generar falsas alarmas. Después congela esa elección y evalúa una sola vez con el test; conserva el resultado aunque empeore.

## 4. ¿Qué necesitas guardar y automatizar?

El modelo incluye las transformaciones de entrada que aprendiste durante el entrenamiento. Guarda esa solución completa, identifica su versión y comprueba que puedes cargarla para predecir. Construye un pipeline que conecte preparación, entrenamiento, comparación y registro; debe mostrar si una etapa falla y por qué.

> **Antes de seguir:** ¿estás eligiendo por evidencia o porque un algoritmo es más conocido, más complejo o el último que ejecutaste?

---

# Herramientas y evidencia de aprendizaje

Las herramientas tienen responsabilidades distintas. Úsalas para responder estas preguntas sobre tu propio proyecto:

| Herramienta | Para qué la usarás | Pregunta que debes poder responder |
|---|---|---|
| **Git** | Conservar código y decisiones | ¿Qué cambio produjo esta versión? |
| **DVC** | Versionar datos y archivos del modelo | ¿Puedo recuperar los mismos datos y modelo? |
| **MLflow** | Registrar experimentos y versiones de modelos | ¿Qué ejecución produjo el modelo elegido? |
| **Prefect** | Coordinar etapas y observar sus estados | ¿Qué ocurrió cuando ejecuté el pipeline? |

### Experimento, modelo y pipeline

El **experimento** es una prueba con datos y configuración identificados; el **modelo** es el resultado entrenado; el **pipeline** es el proceso que conecta las etapas. Versionarlos permite relacionar el proceso con su resultado.

---

# Qué entregar

### Repositorio propio

Código, instrucciones de ejecución y referencias a las versiones de datos y del modelo. Otra persona debe poder reconstruir la solución.

### Comparación de experimentos

Hipótesis, cambio realizado, resultado y conclusión para cada configuración. Conserva los registros en MLflow.

### Decisión final breve

Modelo y umbral elegidos, errores relevantes, resultado del test y limitaciones del dataset. No basta con decir que obtuvo la mejor puntuación.

### Demostración

Ejecutar el pipeline en Prefect, cargar una versión guardada y producir predicciones.

Mostrar que puedes recuperar datos y modelo mediante DVC.

---

# Preguntas para tu defensa

- ¿Qué experimento cambió tu opinión?
- ¿Qué errores aceptaste y por qué?
- ¿Cómo sabes que tu comparación fue justa?
- Si perdieras tu entorno local, ¿qué necesitarías para reconstruirlo?

**Se valorará:** claridad de las decisiones, comparación coherente, trazabilidad y capacidad de reproducción.

Una métrica alta por sí sola no demuestra que resolviste el problema. Este benchmark histórico permite practicar el ciclo de ML; no prueba eficacia sobre una red actual.

---

# Dónde investigar

**Dataset y atribución:** UNSW · Git · DVC · MLflow · Prefect

Consulta las fuentes, elige cómo implementar tu solución y deja por escrito lo que aprendiste. El documento orienta las decisiones; el diseño y la construcción del proyecto son responsabilidad de tu equipo.
