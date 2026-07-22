"""
Monitoreo remoto: sincroniza el archivo de métricas con OCI Object Storage.

Cada vez que la app escribe un registro en logs/metricas.jsonl, sube el
archivo COMPLETO al bucket agente-cgt-logs como el objeto metricas.jsonl.
También permite descargarlo (para el reporte de calidad desde la nube).

La fuente de verdad es el archivo local. Si la subida a OCI falla por
cualquier motivo (sin credenciales, sin red, permisos, etc.) se falla en
silencio: la app nunca se rompe por el monitoreo remoto.
"""

# Configuración del bucket de logs en OCI.
PERFIL_OCI = "SANJOSE"          # perfil de ~/.oci/config (región us-sanjose-1)
REGION = "us-sanjose-1"
NAMESPACE = "axj9p0fhoxqv"      # namespace de Object Storage de la tenancy
BUCKET = "agente-cgt-logs"
OBJETO = "metricas.jsonl"       # nombre del archivo en Object Storage
ARCHIVO_LOCAL = "logs/metricas.jsonl"

# El cliente de Object Storage se crea una sola vez y se reutiliza.
# _cliente_intentado evita reintentar la configuración cuando ya falló.
_cliente = None
_cliente_intentado = False


def _en_instancia_oci(timeout=1.0):
    """Comprueba rápido si corremos dentro de una VM de OCI: el endpoint de
    metadatos (169.254.169.254) solo responde ahí. Sin esta guarda, intentar
    'instance principals' fuera de OCI se cuelga esperando ese endpoint."""
    import socket

    try:
        with socket.create_connection(("169.254.169.254", 80), timeout):
            return True
    except OSError:
        return False


def _obtener_cliente():
    """Crea (una única vez) el cliente de Object Storage.

    Primero intenta el perfil SANJOSE de ~/.oci/config (entorno local). Si no
    hay config local —como en la VM de OCI, que no guarda credenciales— cae a
    'instance principals', que autentica por la identidad de la propia VM.
    Devuelve None si ninguna vía funciona."""
    global _cliente, _cliente_intentado
    if _cliente_intentado:
        return _cliente
    _cliente_intentado = True
    import oci

    # 1) Perfil local (~/.oci/config).
    try:
        config = oci.config.from_file(profile_name=PERFIL_OCI)
        _cliente = oci.object_storage.ObjectStorageClient(config)
        return _cliente
    except Exception:
        pass

    # 2) Instance principals (dentro de la VM de OCI, sin credenciales
    # locales). Solo se intenta si de verdad estamos en una VM de OCI, para
    # no colgarnos en desarrollo local sin credenciales.
    if not _en_instancia_oci():
        _cliente = None
        return _cliente
    try:
        signer = oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
        _cliente = oci.object_storage.ObjectStorageClient(
            config={"region": REGION}, signer=signer
        )
    except Exception:
        _cliente = None
    return _cliente


def subir_metricas(ruta_local=ARCHIVO_LOCAL):
    """Sube el archivo local de métricas completo al bucket como
    metricas.jsonl. Falla en silencio ante cualquier error (sin OCI, sin
    credenciales, sin red): el archivo local sigue siendo la fuente de verdad.
    """
    try:
        cliente = _obtener_cliente()
        if cliente is None:
            return
        with open(ruta_local, "rb") as f:
            datos = f.read()
        cliente.put_object(
            namespace_name=NAMESPACE,
            bucket_name=BUCKET,
            object_name=OBJETO,
            put_object_body=datos,
            content_type="application/x-ndjson",
        )
    except Exception:
        # Fallo silencioso: el archivo local ya quedó guardado.
        pass


def descargar_metricas():
    """Devuelve el contenido (texto) del metricas.jsonl del bucket.

    A diferencia de la subida, NO falla en silencio: lanza excepción si no
    hay conexión con OCI o el objeto no existe, para que quien lo llame
    (el reporte con --fuente nube) pueda avisar al usuario."""
    cliente = _obtener_cliente()
    if cliente is None:
        raise RuntimeError(
            "No se pudo conectar con OCI Object Storage "
            "(revisa credenciales o red)."
        )
    respuesta = cliente.get_object(
        namespace_name=NAMESPACE, bucket_name=BUCKET, object_name=OBJETO
    )
    return respuesta.data.content.decode("utf-8")
