import random
import string

import pytest
from requests.exceptions import (
    ConnectionError,
    HTTPError,
    RequestException,
    Timeout,
)

from tinyscibio import request_api_server


def _ensembl_base_url():
    return "https://rest.ensembl.org"


def _varsome_live_base_url():
    return "https://api.varsome.com"


def _varsome_stable_base_url():
    return "https://stable-api.varsome.com"


def _varsome_staging_base_url():
    return "https://staging-api.varsome.com"


@pytest.fixture(
    autouse=True,
    scope="module",
    params=["ensembl", "varsome", "varsome-stable", "varsome-staging"],
)
def api_server_base_url(request):
    if request.param == "ensembl":
        return _ensembl_base_url()
    if request.param == "varsome":
        return _varsome_live_base_url()
    if request.param == "varsome-stable":
        return _varsome_stable_base_url()
    if request.param == "varsome-staging":
        return _varsome_staging_base_url()


@pytest.mark.parametrize(
    "api_server_base_url",
    ["ensembl", "varsome", "varsome-stable", "varsome-staging"],
    indirect=True,
)
def test_request_api_server(api_server_base_url):
    assert request_api_server(api_server_base_url).status_code == 200


@pytest.mark.parametrize(
    "url, exception",
    [
        ("https://rest.ensembl.org/vep/", HTTPError),
        ("https://api-stable.varsome.com/", ConnectionError),
        # missing url
        ("", RequestException),
        # invalid scheme
        (
            "".join(random.choice(string.ascii_lowercase) for _ in range(15)),
            RequestException,
        ),
    ],
)
def test_request_raising_error(url, exception):
    with pytest.raises(exception):
        request_api_server(url)


def test_request_timeout_error():
    with pytest.raises(Timeout):
        request_api_server(_ensembl_base_url(), timeout=0.01)
