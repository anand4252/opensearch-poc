"""OpenSearch client factory.

Local dev talks to an unsecured single node over plain HTTP, so there is no
SigV4 / AWS4Auth here (unlike the AWS-managed setup this project is modelled on).
"""

from urllib.parse import urlparse

from opensearchpy import OpenSearch

from app.config import Settings


def build_client(settings: Settings) -> OpenSearch:
    parsed = urlparse(settings.opensearch_url)
    use_ssl = parsed.scheme == "https"
    port = parsed.port or (443 if use_ssl else 9200)

    return OpenSearch(
        hosts=[{"host": parsed.hostname or "localhost", "port": port}],
        use_ssl=use_ssl,
        verify_certs=False,
        ssl_show_warn=False,
    )
