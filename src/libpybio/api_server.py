from __future__ import annotations

from collections.abc import Iterable, Mapping

import httpx
import requests
from requests import (
    ConnectionError,
    HTTPError,
    RequestException,
    Timeout,
    TooManyRedirects,
)

_RequestHeaderType = Mapping[str, str]
_RequestParamsType = Mapping[str, str | int | bool | Iterable[str]]


def run_httpx_get(
    url: httpx.URL | str,
    params: httpx._types.QueryParamTypes,
    headers: httpx._types.HeaderTypes,
) -> httpx.Response:
    try:
        r = httpx.get(url=url, params=params, headers=headers)
        r.raise_for_status()
        return r

    # separately capture two subclass exception of HTTPError to avoid
    # mypy typing error
    # https://github.com/encode/httpx/issues/3277#issuecomment-2313519463
    # HTTPError does not have `response` attribute, rather its child
    # class HTTPStatusError does
    except httpx.RequestError as exc:
        print(
            f"[ERROR] Request to API server {str(exc.request.url)} failed due to `{exc}`"
        )
        raise SystemExit
    except httpx.HTTPStatusError as exc:
        print(
            f"[ERROR] Request to API server return error code {exc.response.status_code} "
            f"for the given URL {str(exc.request.url)}"
        )
        raise SystemExit


def request_api_server(
    lookup_path: str,
    params: _RequestParamsType,
    headers: _RequestHeaderType,
) -> requests.Response:
    try:
        response = requests.get(
            url=lookup_path,
            headers=headers,
            params=params,
        )
        response.raise_for_status()
        return response
    except HTTPError as exc:
        raise HTTPError(exc.response.reason)
    except ConnectionError as exc:
        raise ConnectionError(exc)
    except Timeout as exc:
        raise Timeout(exc.response)
    except TooManyRedirects as exc:
        raise TooManyRedirects(exc)
    except RequestException as exc:
        raise RequestException(exc)
