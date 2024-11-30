import pytest

from tinyscibio import parse_region


@pytest.mark.parametrize(
    "region, expect_parse",
    [
        ("chr1", ("chr1", None, None)),
        ("chr1:10", ("chr1", 10, None)),
        ("chr1:10-", ("chr1", 10, None)),
        ("chr1:10-100", ("chr1", 10, 100)),
        ("2:10-100", ("2", 10, 100)),
        ("GCA_000001405.15:10-100", ("GCA_000001405.15", 10, 100)),
    ],
)
def test_parse_region(region, expect_parse):
    assert parse_region(region, one_based=False) == expect_parse


@pytest.mark.parametrize(
    "region, expect_parse",
    [
        ("chr1:10", ("chr1", 9, None)),
        ("chr1:10-100", ("chr1", 9, 100)),
    ],
)
def test_parse_region_one_based(region, expect_parse):
    assert parse_region(region, one_based=True) == expect_parse


@pytest.mark.parametrize(
    "region",
    [
        "chr1:10--100",
        "chr1::10-100",
        "chr1-10:100",
        "chr1:10:100",
        "chr1-10-100",
        "chr1|10|100",
        "chr1:-1000",
        "chr1-1000",
        "100-1000",
        "100-",
        "-1000",
    ],
)
def test_parse_region_with_invalid_input_str(region):
    with pytest.raises(ValueError):
        parse_region(region)


@pytest.mark.parametrize(
    "region",
    [
        "chr1:100-10",
    ],
)
def test_parse_region_with_start_larger_than_end(region):
    with pytest.raises(ValueError):
        parse_region(region)


@pytest.mark.parametrize(
    "region, one_based, expect",
    [
        ("chr1:1-10", True, 0),
        ("chr1:0-10", True, 0),
        ("chr1:1-10", False, 1),
        ("chr1:0-10", False, 0),
    ],
)
def test_parse_region_if_handle_start_pos_correct(region, one_based, expect):
    _, start, _ = parse_region(region, one_based)
    assert start == expect
