# Proyecto: Agente RAG - Condiciones Generales de Trabajo IMSS Bienestar

## Objetivo
Agente de IA que responde preguntas sobre el PDF de las Condiciones
Generales de Trabajo (CGT) de IMSS Bienestar. Desafío de Alura.
Tres etapas: (1) leer/procesar el PDF, (2) agente de preguntas y
respuestas, (3) deploy en Oracle Cloud (OCI).

## Stack
- Python 3.10+ con entorno virtual (venv)
- LangChain para orquestación RAG
- pypdf para extracción de texto del PDF
- Interfaz: Streamlit (simple, sin enfocarse en estética)
- LLM: Cohere (command-r-plus-08-2024)
- Deploy: OCI Compute (VM Always Free tier)

## Convenciones
- Código y comentarios en español
- Commits descriptivos en español (historial de commits es evaluado)
- El documento fuente vive en documentos/ y NO se sube al repo
  si contiene información sensible (evaluar; agregar a .gitignore si aplica)
- Nunca hardcodear API keys: usar variables de entorno (.env con python-dotenv)
- .env SIEMPRE en .gitignore

## Comandos
- Activar entorno: source venv/bin/activate
- Instalar deps: pip install -r requirements.txt
- Correr app: streamlit run src/app.py

## Entregables (checklist del desafío)
- [ ] Repo público en GitHub con historial de commits
- [ ] README con: arquitectura, ejemplos de preguntas/respuestas,
      instrucciones de ejecución, captura del deploy en OCI
- [ ] Deploy funcionando en OCI (mínimo un servicio de OCI)

## Prioridades (consejos del desafío)
1. Primero que funcione LOCAL, después el deploy
2. No perder tiempo en interfaz bonita
3. Mantener el código simple y organizado

## Sobre el usuario
- No tiene conocimientos de programación
- Aprobar comandos de pip install, git, python y streamlit siempre es seguro
- Explicar brevemente qué hace cada comando antes de ejecutarlo
- Usar lenguaje simple, sin jerga técnica innecesaria
