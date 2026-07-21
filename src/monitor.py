"""
Monitoreo remoto: sube cada registro de métricas al bucket de OCI Object
Storage (agente-cgt-logs), como complemento del guardado local en
logs/metricas.jsonl.

La fuente de verdad es el archivo local. Si la subida a OCI falla por
cualquier motivo (sin credenciales, sin red, permisos, etc.) se falla en
silencio: la app nunca se rompe por el monitoreo remoto.
"""

import json

# Configuración del bucket de logs en OCI.
PERFIL_OCI = "SANJOSE"          # perfil de ~/.oci/config (región us-sanjose-1)
REGION = "us-sanjose-1"
NAMESPACE = "axj9p0fhoxqv"      # namespace de Object Storage de la tenancy
BUCKET = "agente-cgt-logs"

# El cliente de Object Storage se crea una sola vez y se reutiliza entre
# registros. _cliente_intentado evita reintentar la configuración en cada
# llamada cuando ya falló una vez.
_cliente = None
_cliente_intentado = False


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

    # 2) Instance principals (dentro de la VM de OCI, sin credenciales locales).
    try:
        signer = oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
        _cliente = oci.object_storage.ObjectStorageClient(
            config={"region": REGION}, signer=signer
        )
    except Exception:
        _cliente = None
    return _cliente


def subir_registro(registro):
    """Sube un registro de métrica al bucket como objeto JSON independiente.

    El objeto se nombra metricas/<id>.json para que cada pregunta quede como
    un archivo propio en el bucket. Falla en silencio ante cualquier error.
    """
    try:
        cliente = _obtener_cliente()
        if cliente is None:
            return
        cuerpo = json.dumps(registro, ensure_ascii=False).encode("utf-8")
        cliente.put_object(
            namespace_name=NAMESPACE,
            bucket_name=BUCKET,
            object_name=f"metricas/{registro['id']}.json",
            put_object_body=cuerpo,
            content_type="application/json",
        )
    except Exception:
        # Fallo silencioso: el registro local ya quedó guardado.
        pass
