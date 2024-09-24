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
