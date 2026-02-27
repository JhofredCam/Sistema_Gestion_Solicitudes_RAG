# Sistema de Gestion de Solicitudes RAG (UNAL)

RAG para normativa de pregrado de la Universidad Nacional de Colombia (Sede Medellin), implementado con LangChain + LangGraph.

## Uso Diferenciado de LLMs (Seccion 4)

El proyecto usa dos proveedores con responsabilidades separadas:

- Groq (velocidad / baja latencia)
  - `intent_router` (clasificacion de intencion)
  - `k_selector` (seleccion dinamica de `k`)
  - `direct_llm` (respuestas generales sin retrieval)
- Gemini (razonamiento sobre contexto)
  - `rag_generator` (sintesis basada en evidencia recuperada)
  - `evaluator` (verificacion de grounding y coherencia)

Justificacion tecnica:

- Groq se usa donde importa respuesta rapida y costo de inferencia bajo por nodo.
- Gemini se usa en tareas con mayor carga semantica: construir respuesta con contexto y criticar soporte factual.
- Esta separacion reduce latencia promedio del flujo sin sacrificar control de alucinaciones en la rama RAG.

## Configuracion central

La asignacion de modelos por rol esta centralizada en:

- `src/llm_config.py`

Esto evita configuraciones dispersas y facilita justificar/ajustar decisiones por actividad.

## Variables de entorno

- `GROQ_API_KEY`
- `GOOGLE_API_KEY`
