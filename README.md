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
