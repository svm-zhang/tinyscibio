from __future__ import annotations

import httpx


def run_httpx_get(
    url: httpx.URL | str, params: dict[str, str], headers: httpx.Headers
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
