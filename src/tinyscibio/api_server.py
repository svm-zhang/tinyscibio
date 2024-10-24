from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Optional

import requests
from requests import (
    ConnectionError,
    HTTPError,
    RequestException,
    Timeout,
)

_RequestHeaderType = Mapping[str, str]
_RequestParamsType = Mapping[str, str | int | bool | Iterable[str]]


def request_api_server(
    url: str,
    params: Optional[_RequestParamsType] = None,
    headers: Optional[_RequestHeaderType] = None,
    timeout: Optional[float] = None,
    stream: bool = False,
) -> requests.Response:
    """
    Connect to an API server given configuration and return a Response object.

    This is a wrapper around the get function from the requests package. I use
    this to query various remote biology databases with APIs, for instance,
    Varsome, Ensembl, GTEx, cBioPortal, etc.

    Examples:
        Let us query the Varsome API server to annotate a variant with ACMG
        information. Varsome requires API token to use the service. Provide an
        API token inside the HTTP header; otherwise, you can only run one API
        query once per day. The specific variant used here is from
        https://api.varsome.com/.

        >>> from tinyscibio import request_api_server
        >>> api_server_url = "https://api.varsome.com/lookup/"
        >>> qry = "15-73027478-T-C"
        >>> lookup_path = f"{api_server_url}{qry}"
        >>> qry_params = {
                "add_ACMG_annotation": 1
            }
        >>> headers = {"Accept": "application/json"}
        >>> response = request_api_server(
                lookup_path,
                qry_params,
                headers
            )
        >>> assert response.status_code == 200  # if the connection is success


    Parameters:
        url: lookup path (not just the API server URL).
        params: query parameters to defined what you want to request.
        headers: HTTP headers.
        timeout: How many seconds to wait for the server to send data
                 before giving up.
        stream: Whether or not getting response data immediately.

    Returns:
        A requests.Response object.

    Raises:
        HTTPError: When HTTP error occurred.
        Timeout: When request timed out.
        ConnectionError: When connection error occurred.
        RequestException: All other errors not covered by the above ones.
    """
    try:
        response = requests.get(
            url=url,
            params=params,
            headers=headers,
            stream=stream,
            timeout=timeout,
        )
        response.raise_for_status()
        return response
    except HTTPError as exc:
        raise HTTPError(exc.response.reason)
    except Timeout as exc:
        raise Timeout(exc)
    except ConnectionError as exc:
        raise ConnectionError(exc)
    except RequestException as exc:
        raise RequestException(exc)
