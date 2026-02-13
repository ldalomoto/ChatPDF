import streamlit as st
import os
import hashlib
import chromadb
import google.generativeai as genai
from typing import Tuple, Optional

# Procesamiento de archivos
from pypdf import PdfReader
import docx
import xml.etree.ElementTree as ET
import pandas as pd
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================
st.set_page_config(page_title="Chat con Documentos + Gemini")

# Carga variables de entorno desde .env
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Modelo de embeddings local
EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# Se Inicializa el Cliente de ChromaDB
client = chromadb.Client()

# Extensiones soportadas
SUPPORTED_EXTENSIONS = {
    '.pdf': 'PDF',
    '.docx': 'Word',
    '.doc': 'Word',
    '.txt': 'Texto',
    '.xml': 'XML',
    '.csv': 'CSV',
    '.xlsx': 'Excel',
    '.xls': 'Excel',
}

# ============================================================
# SESSION STATE
# ============================================================
if "collection" not in st.session_state:
    st.session_state.collection = None

if "file_processed" not in st.session_state:
    st.session_state.file_processed = False

if "file_hash" not in st.session_state:
    st.session_state.file_hash = None

if "file_name" not in st.session_state:
    st.session_state.file_name = None

if "file_type" not in st.session_state:
    st.session_state.file_type = None

# ============================================================
# FUNCIONES DE EXTRACCIÓN DE TEXTO
# ============================================================
def hash_file(file) -> str:
    """Genera hash único para cualquier archivo."""
    return hashlib.sha256(file.getvalue()).hexdigest()

def get_file_type(filename: str) -> Optional[Tuple[str, str]]:
    """Determina el tipo de archivo basado en la extensión."""
    _, ext = os.path.splitext(filename.lower())
    if ext in SUPPORTED_EXTENSIONS:
        return ext, SUPPORTED_EXTENSIONS[ext]
    return None, None

def extract_text_from_pdf(file) -> str:
    """Extrae texto de un PDF."""
    reader = PdfReader(file)
    text = ""

    for i, page in enumerate(reader.pages):
        content = page.extract_text()
        if content:
            text += f"\n[Página {i+1}]\n{content}"

    return text

def extract_text_from_docx(file) -> str:
    """Extrae texto de un archivo Word (.docx)."""
    doc = docx.Document(file)
    text = ""
    
    for i, para in enumerate(doc.paragraphs):
        if para.text.strip():
            text += f"\n[Párrafo {i+1}]\n{para.text}"
    
    return text

def extract_text_from_txt(file) -> str:
    """Extrae texto de un archivo de texto plano."""
    return file.getvalue().decode('utf-8')

def extract_text_from_xml(file) -> str:
    """Extrae texto de un archivo XML."""
    try:
        content = file.getvalue().decode('utf-8')
        root = ET.fromstring(content)
        
        # Función recursiva para extraer texto
        def extract_xml_text(element, depth=0):
            text = ""
            if element.text and element.text.strip():
                text += element.text.strip() + "\n"
            for child in element:
                text += extract_xml_text(child, depth + 1)
            if element.tail and element.tail.strip():
                text += element.tail.strip() + "\n"
            return text
        
        return extract_xml_text(root)
    except ET.ParseError as e:
        return f"Error al parsear XML: {str(e)}\nContenido crudo:\n{file.getvalue().decode('utf-8', errors='ignore')}"

def extract_text_from_csv(file) -> str:
    """Extrae texto de un archivo CSV."""
    try:
        # Intentar diferentes encodings
        content = file.getvalue()
        for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
            try:
                df = pd.read_csv(pd.io.common.BytesIO(content), encoding=encoding)
                text = f"Archivo CSV con {len(df)} filas y {len(df.columns)} columnas\n\n"
                text += "Columnas: " + ", ".join(df.columns) + "\n\n"
                text += "Primeras filas:\n" + df.head().to_string() #df.to_string() para leer todo / .head(#) para solo las primeras # filas
                return text
            except:
                continue
        return "No se pudo leer el archivo CSV con los encodings probados"
    except Exception as e:
        return f"Error al procesar CSV: {str(e)}"

def extract_text_from_excel(file) -> str:
    """Extrae texto de un archivo Excel."""
    try:
        df = pd.read_excel(file, sheet_name=None)  # Leer todas las hojas
        
        text = f"Archivo Excel con {len(df)} hojas\n\n"
        
        for sheet_name, sheet_data in df.items():
            text += f"=== Hoja: {sheet_name} ===\n"
            text += f"Forma: {sheet_data.shape[0]} filas x {sheet_data.shape[1]} columnas\n\n"
            text += "Primeras filas:\n" + sheet_data.head().to_string() + "\n\n"
        
        return text
    except Exception as e:
        return f"Error al procesar Excel: {str(e)}"

def extract_text_from_file(file, file_type: str) -> str:
    """Extrae texto de cualquier tipo de archivo soportado."""
    extraction_functions = {
        '.pdf': extract_text_from_pdf,
        '.docx': extract_text_from_docx,
        '.doc': extract_text_from_docx,
        '.txt': extract_text_from_txt,
        '.xml': extract_text_from_xml,
        '.csv': extract_text_from_csv,
        '.xlsx': extract_text_from_excel,
        '.xls': extract_text_from_excel,
    }
    
    if file_type in extraction_functions:
        return extraction_functions[file_type](file)
    else:
        return f"Tipo de archivo no soportado: {file_type}"

# ============================================================
# PROCESAMIENTO
# ============================================================
def chunk_text(text):
    """
    Divide un texto largo en fragmentos (chunks) con solapamiento.

    chunk_size:
        - Número máximo de caracteres por fragmento
        - Valores típicos: 400–800
        - Más grande = más contexto, pero embeddings más caros

    overlap:
        - Número de caracteres que se repiten entre chunks consecutivos
        - Evita que una idea quede cortada entre fragmentos
        - Regla común: 10–20% del chunk_size

    Devuelve:
        Lista de diccionarios, cada uno representando un chunk con:
        - id           -> identificador único
        - content      -> texto del fragmento
        - start_index  -> posición donde comienza en el texto original
        - size         -> longitud real del chunk
    """
    chunk_size = 500 
    overlap = 100
    chunks = []          # Aquí guardaremos todos los fragmentos
    start = 0            # Puntero que indica desde dónde empezamos a cortar
    chunk_id = 0         # Contador para asignar IDs únicos

    # El while se ejecuta mientras NO hayamos llegado al final del texto
    while start < len(text):

        # 1️⃣ Cortamos el texto desde 'start' hasta 'start + chunk_size'
        #    Python corta automáticamente si se pasa del largo del texto
        chunk_text = text[start:start + chunk_size]

        # 2️⃣ Guardamos el chunk junto con metadata útil
        chunks.append({
            "id": f"chunk_{chunk_id}",   # Identificador único del fragmento
            "content": chunk_text,       # Texto real del fragmento
            "start_index": start,        # Posición en el texto original
            "size": len(chunk_text)      # Tamaño real del fragmento
        })

        # 3️⃣ Incrementamos el ID para el próximo chunk
        chunk_id += 1

        # 4️⃣ Avanzamos el puntero 'start'
        #    No avanzamos chunk_size completo,
        #    sino (chunk_size - overlap) para que haya solapamiento
        #
        #    Ejemplo:
        #    chunk_size = 500
        #    overlap    = 100
        #    start avanza 400 caracteres
        #
        #    Los últimos 100 caracteres del chunk actual
        #    aparecerán también al inicio del siguiente
        start += chunk_size - overlap

    # 5️⃣ Cuando start >= len(text), el while termina
    #    y devolvemos todos los fragmentos creados
    return chunks

def create_chroma_collection(chunks):
    """
    Crea una colección nueva en ChromaDB a partir de los chunks generados.

    Cada chunk se almacena junto con:
    - su embedding (vector numérico)
    - su texto original
    - metadata útil
    """

    # ------------------------------
    # 1️⃣ Borrado defensivo
    # ------------------------------
    # Si ya existe una colección con el mismo nombre ("document_rag"),
    try:
        client.delete_collection("document_rag")
    except:
        # Si la colección no existe, Chroma lanza error.
        # Lo ignoramos porque es un caso esperado.
        pass

    # ------------------------------
    # 2️⃣ Crear colección nueva
    # ------------------------------
    # Aquí Chroma crea:
    # - una tabla de documentos
    # - un índice vectorial
    # - espacio para metadatos

    collection = client.create_collection(name="document_rag")

    # ------------------------------
    # 3️⃣ Separar texto de metadata
    # ------------------------------
    # Extraemos SOLO el contenido textual de cada chunk.
    # Esto es lo que se convertirá en embeddings.
    texts = [c["content"] for c in chunks]

    # ------------------------------
    # 4️⃣ Generar embeddings
    # ------------------------------
    # El modelo de SentenceTransformers convierte cada texto
    # en un vector numérico.
    #
    # Cada vector representa el significado del chunk.
    embeddings = EMBEDDING_MODEL.encode(texts)

    # ------------------------------
    # 5️⃣ Insertar datos en Chroma
    # ------------------------------
    collection.add(
        # Texto original del chunk
        documents=texts,

        # Vectores que permiten búsqueda semántica
        embeddings=embeddings.tolist(),

        # IDs únicos
        # Sirven para identificar cada chunk internamente
        ids=[c["id"] for c in chunks],

        # Metadata asociada a cada chunk
        metadatas=[
            {
                "chunk_index": i,         # Orden del chunk
                "start_index": c["start_index"],  # Posición en el texto original
                "chunk_size": c["size"]   # Tamaño real del fragmento
            }
            for i, c in enumerate(chunks)
        ]
    )

    # ------------------------------
    # 6️⃣ Devolver colección lista
    # ------------------------------
    # La colección ya puede:
    # - recibir queries (preguntas)
    # - devolver chunks relevantes
    return collection

def retrieve_context(collection, query, k=4):
    """
    Recupera los k chunks más similares a la pregunta.
    Devuelve tanto el texto como la metadata asociada.
    """
    query_embedding = EMBEDDING_MODEL.encode([query])

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=k
    )

    return results

def ask_gemini(context, question):
    """
    Llama a Gemini usando el contexto recuperado.
    """
    model = genai.GenerativeModel("models/gemini-2.5-flash-lite")

    prompt = f"""
Eres un asistente que responde SOLO con la información del contexto.
Si la respuesta no está en el contexto, di: "No se encuentra en el documento".

Contexto:
{context}

Pregunta:
{question}
"""

    response = model.generate_content(prompt)
    return response.text

# ============================================================
# INTERFAZ
# ============================================================

st.title("📄 Chat con Documentos + ChromaDB + Gemini")

# Mostrar tipos de archivo soportados
st.markdown("**Formatos soportados:** " + ", ".join(sorted(set(SUPPORTED_EXTENSIONS.values()))))

uploaded_file = st.file_uploader(
    "Sube un documento",
    type=list(SUPPORTED_EXTENSIONS.keys())
)

if uploaded_file:
    # Obtener tipo de archivo
    ext, file_type_name = get_file_type(uploaded_file.name)
    
    if ext is None:
        st.error(f"Tipo de archivo no soportado. Formatos aceptados: {', '.join(SUPPORTED_EXTENSIONS.keys())}")
    else:
        st.info(f"📁 Archivo: {uploaded_file.name} | Tipo: {file_type_name}")
        
        # 🔄 Detectar cambio de archivo y resetear estado
        current_hash = hash_file(uploaded_file)

        if st.session_state.file_hash != current_hash:
            st.session_state.file_hash = current_hash
            st.session_state.file_processed = False
            st.session_state.collection = None
            st.session_state.file_name = uploaded_file.name
            st.session_state.file_type = file_type_name

# ------------------------------
# BOTÓN PROCESAR DOCUMENTO
# ------------------------------
if uploaded_file and ext and not st.session_state.file_processed:
    if st.button(f"📥 Procesar {file_type_name}"):
        with st.spinner(f"Procesando {uploaded_file.name}..."):
            try:
                # Extraer texto del archivo
                text = extract_text_from_file(uploaded_file, ext)
                
                if text.startswith("Error") or text.startswith("Tipo de archivo no soportado"):
                    st.error(text)
                else:
                    # Mostrar vista previa del texto extraído
                    with st.expander("📋 Vista previa del texto extraído"):
                        st.text_area(
                            "Texto extraído (primeros 2000 caracteres)",
                            text[:2000] + ("..." if len(text) > 2000 else ""),
                            height=200
                        )
                    
                    # Procesar texto
                    chunks = chunk_text(text)
                    st.session_state.collection = create_chroma_collection(chunks)
                    st.session_state.file_processed = True
                    
                    st.success(f"✅ Documento procesado ({len(chunks)} fragmentos)")
            except Exception as e:
                st.error(f"Error al procesar el archivo: {str(e)}")

# ------------------------------
# SECCIÓN DE PREGUNTAS
# ------------------------------
if st.session_state.file_processed and st.session_state.collection:
    st.divider()
    st.subheader(f"❓ Pregunta al documento: {st.session_state.file_name}")

    question = st.text_input("Escribe tu pregunta")

    if st.button("🤖 Preguntar") and question:
        with st.spinner("Buscando respuesta..."):
            results = retrieve_context(st.session_state.collection, question)

            # Unimos los documentos para Gemini
            context_text = "\n\n".join(results["documents"][0])

            answer = ask_gemini(context_text, question)

        st.subheader("🤖 Respuesta")
        st.write(answer)

        # ------------------------------
        # DETALLE DEL CONTEXTO USADO
        # ------------------------------
        with st.expander("📚 Contexto usado (detallado)"):
            for i, (doc, meta) in enumerate(
                zip(results["documents"][0], results["metadatas"][0])
            ):
                st.markdown(f"""
**Chunk #{meta['chunk_index']}**
- 📍 Inicio en texto: `{meta['start_index']}`
- 📏 Tamaño: `{meta['chunk_size']}` caracteres

```text
{doc}
""")