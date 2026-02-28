# Sistema de Gestion de Solicitudes RAG (UNAL)

RAG para normativa de pregrado de la Universidad Nacional de Colombia (Sede Medellin), implementado con LangChain + LangGraph.

## Objetivo de la practica / Practice goal

ES:
Sistema RAG para el Sistema de Gestion Estudiantil de la Facultad de Minas. Enfoque principal en pregrado, con soporte para consultas de posgrado cuando la normativa aplica.

EN:
RAG system for the Faculty of Mines Student Management System. Primary focus on undergraduate queries, with partial coverage for graduate queries when the regulations apply.

## Arquitectura RAG (alto nivel) / RAG architecture (high level)

ES:
Carga de documentos HTML -> chunking -> embeddings -> base vectorial -> retrieval con k dinamico -> generacion con grounding -> verificacion y retry controlado.

EN:
HTML ingestion -> chunking -> embeddings -> vector store -> dynamic top-k retrieval -> grounded generation -> verification with controlled retries.

## Uso Diferenciado de LLMs

El proyecto usa una arquitectura de modelos especializados para optimizar latencia y precision.

ES:
- Groq (llama-3.1-8b-instant) para clasificacion de intencion, seleccion de k y respuestas directas.
- Gemini (gemini-2.5-flash) para generacion RAG y verificacion de grounding.
- Justificacion: Groq minimiza latencia en tareas ligeras y Gemini aporta razonamiento sobre contexto normativo.

EN:
- Groq (llama-3.1-8b-instant) for intent routing, k selection, and direct answers.
- Gemini (gemini-2.5-flash) for grounded RAG generation and evaluation.
- Rationale: Groq reduces latency for light tasks; Gemini improves contextual reasoning and verification.

## Configuracion central

La asignacion de modelos por rol esta centralizada en:

- `src/llm_config.py`

Esto evita configuraciones dispersas y facilita justificar/ajustar decisiones por actividad.

## Variables de entorno

- `GROQ_API_KEY`
- `GOOGLE_API_KEY`
- `OPENAI_API_KEY`

Opcionales:

- `UNAL_RAG_DOCS_PATH` (default: `docs`)
- `UNAL_RAG_VECTORSTORE_PATH` (default: `db/chroma_db`)
- `UNAL_RAG_MIN_DOCS` (default: `50`)

Se recomienda crear un `.env` usando `.env.example`.

## Instalacion local

Pasos sugeridos (PowerShell):

```
python -m venv .venv
pip install -e .
Copy-Item .env.example .env
```

Luego edita `.env` y completa las llaves requeridas.

Nota Windows:
Si `Activate.ps1` esta bloqueado por politica de ejecucion, puedes usar el
Python del entorno virtual sin activarlo:

```powershell
.\.venv\Scripts\python -m unal_rag doctor
```

Opcional (habilitar activacion):

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv\Scripts\Activate.ps1
```

## Ejecucion

Con el CLI instalado:

```powershell
.\.venv\Scripts\unal-rag.exe doctor
```

Sin instalar el paquete (modo directo):

```powershell
python -c "import sys; sys.path.insert(0, 'src'); from unal_rag.app.cli import main; main(['doctor'])"
```

## CLI

Comandos disponibles:

- `unal-rag doctor`
- `unal-rag ingest` (stub)
- `unal-rag ask "pregunta..."` (stub)

Si prefieres usar el modulo directamente:

```bash
python -m unal_rag doctor
```

## Pruebas

Validar el CLI y el doctor:

```bash
python -m unal_rag doctor
```

```bash
python -m unal_rag doctor --strict
```

Notas:
- `doctor --strict` retorna salida no-cero si hay menos de 50 documentos soportados.

## Mapa de requisitos (Practica 1) / Requirements map (Practice 1)

ES:
- Pipeline extremo a extremo: `src/ingestion_pipeline.py`, `src/main.py`, `src/nodes/*`.
- Grafo con LangGraph: `src/main.py` y `workflow.mmd`.
- Control de alucinaciones: `src/nodes/generator.py`, `src/nodes/evaluator.py`.
- Uso diferenciado de LLMs: `src/llm_config.py`.
- Trazabilidad: `src/nodes/retriever.py`, `src/nodes/generator.py`, `src/state.py`.
- CLI de soporte: `src/unal_rag/app/cli.py`, `src/unal_rag/app/doctor.py`.

EN:
- End-to-end pipeline: `src/ingestion_pipeline.py`, `src/main.py`, `src/nodes/*`.
- LangGraph flow: `src/main.py` and `workflow.mmd`.
- Hallucination control: `src/nodes/generator.py`, `src/nodes/evaluator.py`.
- Differentiated LLM usage: `src/llm_config.py`.
- Traceability: `src/nodes/retriever.py`, `src/nodes/generator.py`, `src/state.py`.
- Support CLI: `src/unal_rag/app/cli.py`, `src/unal_rag/app/doctor.py`.

## Datos y corpus / Data and corpus

ES:
El corpus contiene 56 documentos HTML en `docs/` (acuerdos, resoluciones, circulares y conceptos), base normativa para pregrado.

EN:
The corpus includes 56 HTML documents in `docs/` (agreements, resolutions, circulars, and concepts), serving as the undergraduate normative base.

## Ingesta y limpieza / Ingestion and cleaning

ES:
Carga por lote con `DirectoryLoader` y `BSHTMLLoader` sobre `docs/*.html`. Metadatos clave: `source`, `doc_id`, `chunk_id`.

EN:
Batch loading with `DirectoryLoader` and `BSHTMLLoader` over `docs/*.html`. Key metadata: `source`, `doc_id`, `chunk_id`.

## Chunking / Segmenting

ES:
`RecursiveCharacterTextSplitter` con `chunk_size=256`, `chunk_overlap=48`, tokenizer `intfloat/multilingual-e5-small`.

EN:
`RecursiveCharacterTextSplitter` with `chunk_size=256`, `chunk_overlap=48`, tokenizer `intfloat/multilingual-e5-small`.

## Embeddings y Vector DB / Embeddings and Vector DB

ES:
Embeddings `intfloat/multilingual-e5-small` y base `Chroma` persistida en `db/chroma_db`, metrica cosine.

EN:
Embeddings `intfloat/multilingual-e5-small` and a `Chroma` store persisted at `db/chroma_db`, cosine similarity.

## Recuperacion / Retrieval

ES:
Top-k dinamico segun intencion. Valores por defecto en `src/nodes/retriever.py` con rango `2..8`.

EN:
Dynamic top-k by intent. Defaults in `src/nodes/retriever.py` with range `2..8`.

## Trazabilidad y verificacion / Traceability and verification

ES:
La respuesta incluye trazas por documento con `doc_id`, `chunk_id`, `page` y `source`, y el evaluador valida grounding con retry controlado.

EN:
Answers include per-document traces with `doc_id`, `chunk_id`, `page`, and `source`, and the evaluator enforces grounding with controlled retries.

## Tools

ES:
Tools: `src/tools/*` y nodos de orquestacion en `src/nodes/tools_pre.py` y `src/nodes/tools_post.py`.
Funciones disponibles:
- `verificar_perdida_calidad_estudiante`: evalua perdida de calidad si PAPA/promedio < 3.0.
- `calcular_promedio`: promedio simple de una lista de notas.
- `calcular_creditos_faltantes`: calcula creditos pendientes (requeridos vs aprobados).
- `contar_menciones_norma`: cuenta ocurrencias de un termino en el contexto.
- `calcular_plazo`: fecha limite a partir de fecha inicio + dias.
- `verificar_requisitos`: verifica requisitos vs perfil (creditos, promedio, semestres).
- `clarificar_plan`: detecta codigos de plan de estudios y necesidad de aclaracion.
- `resumir_norma`: genera un resumen del contexto recuperado.
Uso:
- `tools_pre.py` ejecuta herramientas antes del retrieval para calculos directos.
- `tools_post.py` ejecuta herramientas sobre el contexto recuperado (resumen, menciones, requisitos).

EN:
Tools: `src/tools/*` and orchestration nodes in `src/nodes/tools_pre.py` and `src/nodes/tools_post.py`.
Available functions:
- `verificar_perdida_calidad_estudiante`: checks loss of student status if GPA/PAPA < 3.0.
- `calcular_promedio`: simple average of grades.
- `calcular_creditos_faltantes`: computes remaining credits (required vs approved).
- `contar_menciones_norma`: counts term mentions in context.
- `calcular_plazo`: computes deadline from start date + days.
- `verificar_requisitos`: validates requirements against profile (credits, GPA, semesters).
- `clarificar_plan`: detects study plan codes and whether clarification is needed.
- `resumir_norma`: summarizes the retrieved context.
Usage:
- `tools_pre.py` runs tools before retrieval for direct calculations.
- `tools_post.py` runs tools over retrieved context (summary, mentions, requirements).

## Memoria / Memory

ES:
Memoria: `src/nodes/memory.py` y `src/unal_rag/memory.py`.
Persistencia en `db/memory.json` via `MemoryStore`.
Extrae perfil desde la pregunta: `promedio`, `creditos_aprobados`, `semestres`, `programa`.
Incluye glosario inicial (PAPA) y guarda `plan_code` si se detecta.
Solo marca `memory_updated` si el usuario solicita guardar (recuerda/guarda).

EN:
Memory: `src/nodes/memory.py` and `src/unal_rag/memory.py`.
Persists to `db/memory.json` via `MemoryStore`.
Extracts profile from the question: `promedio`, `creditos_aprobados`, `semestres`, `programa`.
Includes a default glossary (PAPA) and stores `plan_code` if detected.
Only sets `memory_updated` when the user explicitly asks to save (remember/save).

## Pruebas / Tests

ES:
`tests/test_doctor.py` valida el CLI y el conteo minimo de documentos.

EN:
`tests/test_doctor.py` validates the CLI and minimum document count.
