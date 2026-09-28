import json
import subprocess
import sys
import threading
import urllib.error
import urllib.request
import itertools
from pathlib import Path

import pytest

from quorum.analyzer import ValidationError, analyze
from quorum.server import build_server

ROOT = Path(__file__).resolve().parents[1]


def replica(identifier, weight=1, dc="A", online=True):
    return {
        "id": identifier,
        "weight": weight,
        "datacenter": dc,
        "online": online,
    }


def make_payload(replicas, read_threshold=0, write_threshold=0,
                 read_dcs=None, write_dcs=None):
    return {
        "replicas": replicas,
        "read": {
            "weight_threshold": read_threshold,
            "required_datacenters": read_dcs or [],
        },
        "write": {
            "weight_threshold": write_threshold,
            "required_datacenters": write_dcs or [],
        },
    }


def feasible_masks_brute(payload):
    replicas = sorted(payload["replicas"], key=lambda item: item["id"])
    n = len(replicas)

    def side_masks(side):
        required = set(side["required_datacenters"])
        result = []
        for mask in range(1 << n):
            selected = [
                replicas[i]
                for i in range(n)
                if mask & (1 << i)
            ]
            if any(not item["online"] for item in selected):
                continue
            if sum(item["weight"] for item in selected) < side["weight_threshold"]:
                continue
            if not required.issubset({item["datacenter"] for item in selected}):
                continue
            result.append(mask)
        return result

    return replicas, side_masks(payload["read"]), side_masks(payload["write"])


def ids_for(mask, replicas):
    return [
        replicas[i]["id"]
        for i in range(len(replicas))
        if mask & (1 << i)
    ]


def expected_analysis(payload):
    replicas, raw_reads, raw_writes = feasible_masks_brute(payload)
    reads = sorted(raw_reads, key=lambda mask: (mask.bit_count(), mask))
    writes = sorted(raw_writes, key=lambda mask: (mask.bit_count(), mask))
    if not reads or not writes:
        return {
            "read_possible": bool(reads),
            "write_possible": bool(writes),
            "minimum_intersection": None,
            "witness_read": None,
            "witness_write": None,
            "disjoint_counterexample": None,
            "safe": False,
        }

    min_intersection = min(
        (read_mask & write_mask).bit_count()
        for read_mask in reads
        for write_mask in writes
    )
    witness_read, witness_write = next(
        (read_mask, write_mask)
        for read_mask in reads
        for write_mask in writes
        if (read_mask & write_mask).bit_count() == min_intersection
    )
    expected = {
        "read_possible": True,
        "write_possible": True,
        "minimum_intersection": min_intersection,
        "witness_read": {"replica_ids": ids_for(witness_read, replicas)},
        "witness_write": {"replica_ids": ids_for(witness_write, replicas)},
        "disjoint_counterexample": None,
        "safe": min_intersection > 0,
    }
    if min_intersection == 0:
        disjoint = [
            (read_mask, write_mask)
            for read_mask in reads
            for write_mask in writes
            if read_mask & write_mask == 0
        ]
        read_mask, write_mask = min(
            disjoint,
            key=lambda pair: (
                pair[0].bit_count() + pair[1].bit_count(),
                ids_for(pair[0], replicas),
                ids_for(pair[1], replicas),
            ),
        )
        expected["disjoint_counterexample"] = {
            "read": {"replica_ids": ids_for(read_mask, replicas)},
            "write": {"replica_ids": ids_for(write_mask, replicas)},
        }
    return expected


def test_weight_threshold_alone_misses_region_induced_disjoint_pair():
    payload = make_payload(
        [
            replica("a1", 6, "A"),
            replica("a2", 6, "A"),
            replica("b1", 6, "B"),
            replica("b2", 6, "B"),
        ],
        read_threshold=12,
        write_threshold=12,
        read_dcs=["A", "B"],
        write_dcs=["A", "B"],
    )

    result = analyze(payload)

    assert result["read_possible"] is True
    assert result["write_possible"] is True
    assert result["minimum_intersection"] == 0
    assert result["safe"] is False
    assert result["disjoint_counterexample"] == {
        "read": {"replica_ids": ["a1", "b1"]},
        "write": {"replica_ids": ["a2", "b2"]},
    }


def test_all_pairs_intersect_when_each_side_needs_three_of_four():
    payload = make_payload(
        [
            replica("a1", 4, "A"),
            replica("a2", 4, "A"),
            replica("b1", 4, "B"),
            replica("b2", 4, "B"),
        ],
        read_threshold=12,
        write_threshold=12,
        read_dcs=["A", "B"],
        write_dcs=["A", "B"],
    )

    result = analyze(payload)

    assert result == {
        "read_possible": True,
        "write_possible": True,
        "minimum_intersection": 2,
        "witness_read": {"replica_ids": ["a1", "a2", "b1"]},
        "witness_write": {"replica_ids": ["a1", "a2", "b2"]},
        "disjoint_counterexample": None,
        "safe": True,
    }


def test_offline_replicas_cannot_be_selected():
    payload = make_payload(
        [
            replica("a1", 9, "A", online=True),
            replica("b1", 9, "B", online=False),
        ],
        read_threshold=1,
        write_threshold=1,
        read_dcs=["A", "B"],
        write_dcs=["A", "B"],
    )

    assert analyze(payload) == {
        "read_possible": False,
        "write_possible": False,
        "minimum_intersection": None,
        "witness_read": None,
        "witness_write": None,
        "disjoint_counterexample": None,
        "safe": False,
    }


def test_offline_weight_is_not_counted_but_side_can_still_be_possible():
    payload = make_payload(
        [
            replica("offline-heavy", 9, "A", online=False),
            replica("online-light", 1, "A", online=True),
        ],
        read_threshold=1,
        write_threshold=9,
    )

    result = analyze(payload)

    assert result["read_possible"] is True
    assert result["write_possible"] is False
    assert result["safe"] is False
    assert result["minimum_intersection"] is None


def test_equal_weight_tie_uses_read_ids_then_write_ids():
    payload = make_payload(
        [
            replica("d1", 1, "X"),
            replica("d2", 1, "X"),
            replica("d3", 1, "X"),
            replica("d4", 1, "X"),
        ],
        read_threshold=2,
        write_threshold=2,
    )

    result = analyze(payload)

    assert result["minimum_intersection"] == 0
    assert result["disjoint_counterexample"] == {
        "read": {"replica_ids": ["d1", "d2"]},
        "write": {"replica_ids": ["d3", "d4"]},
    }


def test_tie_total_size_has_priority_over_lexicographic_order():
    payload = make_payload(
        [
            replica("aa", 1, "A"),
            replica("bb", 2, "A"),
            replica("cc", 2, "A"),
        ],
        read_threshold=2,
        write_threshold=2,
    )

    result = analyze(payload)

    # 权重均为 2 的 bb 与 cc 是总成员数 2 的不相交对，优先于单权重 id
    # 字典序更小的三成员对。
    assert result["disjoint_counterexample"] == {
        "read": {"replica_ids": ["bb"]},
        "write": {"replica_ids": ["cc"]},
    }
    assert result["minimum_intersection"] == 0


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(
            make_payload(
                [
                    replica("a", 3, "A"),
                    replica("b", 3, "B"),
                    replica("c", 5, "A"),
                    replica("d", 5, "B", online=False),
                ],
                read_threshold=5,
                write_threshold=3,
                read_dcs=["A", "B"],
                write_dcs=["A"],
            ),
            id="offline-region-constraint",
        ),
        pytest.param(
            make_payload(
                [replica(f"r{i}", i % 3 + 1, f"DC{i % 2}") for i in range(12)],
                read_threshold=10,
                write_threshold=12,
                read_dcs=["DC0"],
                write_dcs=["DC1"],
            ),
            id="twelve-replicas-equal-pattern",
        ),
        pytest.param(
            make_payload(
                [replica("x", 9), replica("y", 9)],
                read_threshold=9,
                write_threshold=9,
            ),
            id="equal-heavy-weights",
        ),
    ],
)
def test_manual_cases_match_bitmask_oracle(payload):
    assert analyze(payload) == expected_analysis(payload)


def random_payload(seed):
    rng = __import__("random").Random(seed)
    n = rng.randint(2, 10)
    ids = [f"id-{index:02d}-{rng.randrange(36):x}" for index in range(n)]
    assert len(set(ids)) == n
    datacenters = [rng.choice(["dc-a", "dc-b", "dc-c"]) for _ in range(n)]
    replicas = [
        replica(
            identifier,
            weight=rng.randint(1, 9),
            dc=datacenter,
            online=rng.random() >= 0.2,
        )
        for identifier, datacenter in zip(ids, datacenters)
    ]

    def side():
        dc_options = sorted(set(datacenters))
        required = rng.sample(dc_options, rng.randrange(len(dc_options) + 1))
        return {
            "weight_threshold": rng.randrange(0, 30),
            "required_datacenters": required,
        }

    return {"replicas": replicas, "read": side(), "write": side()}


@pytest.mark.parametrize("seed", range(45))
def test_random_small_instances_match_bitmask_enumeration(seed):
    payload = random_payload(seed)
    assert analyze(payload) == expected_analysis(payload)


@pytest.mark.parametrize(
    "bad_payload,message",
    [
        ({"replicas": []}, "between 2 and 12"),
        (
            make_payload([replica("same"), replica("same")]),
            "duplicate replica id",
        ),
        (
            make_payload([replica("a", 10), replica("b")]),
            "between 1 and 9",
        ),
        (
            make_payload(
                [replica("a"), replica("b")],
                read_dcs=["A", "A"],
            ),
            "read.required_datacenters",
        ),
    ],
)
def test_invalid_input_raises_validation_error(bad_payload, message):
    with pytest.raises(ValidationError, match=message):
        analyze(bad_payload)


@pytest.mark.parametrize(
    "bad_payload",
    [
        pytest.param(
            make_payload([replica("x"), replica("y"), replica("x")]),
            id="duplicate-id-at-zero-and-two",
        ),
        pytest.param(
            make_payload([replica("a", True), replica("b")]),
            id="boolean-weight",
        ),
        pytest.param(
            make_payload([replica("a", 1.0), replica("b")]),
            id="float-weight",
        ),
        pytest.param(
            make_payload([replica("a", online="true"), replica("b")]),
            id="non-boolean-online",
        ),
        pytest.param(
            make_payload(
                [{"id": "a", "weight": 1, "datacenter": "A"}, replica("b")]
            ),
            id="missing-online",
        ),
        pytest.param(
            make_payload(
                [replica("a"), replica("b")], read_threshold=True
            ),
            id="boolean-threshold",
        ),
    ],
)
def test_malformed_fields_raise_validation_error(bad_payload):
    with pytest.raises(ValidationError):
        analyze(bad_payload)


def test_offline_replica_never_enters_quorum_even_alongside_online_members():
    # 旧实现只排除了只含单个离线副本的掩码，多成员掩码仍可夹带离线副本。
    payload = make_payload(
        [
            replica("a1", 1, "A", online=True),
            replica("a2", 1, "A", online=True),
            replica("heavy-offline", 9, "A", online=False),
        ],
        read_threshold=9,
        write_threshold=9,
    )

    result = analyze(payload)

    assert result["read_possible"] is False
    assert result["write_possible"] is False
    assert result["safe"] is False
    assert result["minimum_intersection"] is None


def test_witness_members_actually_realize_minimum_intersection():
    payload = make_payload(
        [replica("a", 1), replica("b", 1), replica("c", 1)],
        read_threshold=1,
        write_threshold=3,
    )

    result = analyze(payload)

    overlap = len(
        set(result["witness_read"]["replica_ids"])
        & set(result["witness_write"]["replica_ids"])
    )
    assert result["minimum_intersection"] == overlap == 1
    assert result["safe"] is True
    assert result["disjoint_counterexample"] is None


def test_results_are_independent_of_input_ordering():
    replicas = [
        replica("a1", 6, "A"),
        replica("a2", 6, "A"),
        replica("b1", 6, "B"),
        replica("b2", 6, "B"),
    ]
    expected = analyze(
        make_payload(replicas, 12, 12, ["A", "B"], ["A", "B"])
    )

    for permutation in itertools.permutations(replicas):
        payload = make_payload(
            list(permutation), 12, 12, ["A", "B"], ["A", "B"]
        )
        assert analyze(payload) == expected


def test_disjoint_counterexample_members_are_really_disjoint():
    payload = make_payload(
        [
            replica("a1", 6, "A"),
            replica("a2", 6, "A"),
            replica("b1", 6, "B"),
            replica("b2", 6, "B"),
        ],
        read_threshold=12,
        write_threshold=12,
        read_dcs=["A", "B"],
        write_dcs=["A", "B"],
    )

    result = analyze(payload)

    counterexample = result["disjoint_counterexample"]
    assert result["safe"] is False
    assert not (
        set(counterexample["read"]["replica_ids"])
        & set(counterexample["write"]["replica_ids"])
    )


def test_cli_reads_stdin_and_writes_json():
    payload = make_payload(
        [replica("a", 9), replica("b", 9)],
        read_threshold=9,
        write_threshold=9,
    )
    completed = subprocess.run(
        [sys.executable, "-m", "quorum"],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        cwd=ROOT,
        check=False,
    )

    assert completed.returncode == 0
    assert json.loads(completed.stdout) == expected_analysis(payload)


def test_cli_invalid_input_exits_with_code_2_and_message_on_stderr():
    completed = subprocess.run(
        [sys.executable, "-m", "quorum"],
        input=json.dumps(make_payload([replica("x"), replica("x")])),
        text=True,
        capture_output=True,
        cwd=ROOT,
        check=False,
    )

    assert completed.returncode == 2
    assert "error" in completed.stderr
    assert completed.stdout == ""


def test_cli_malformed_json_exits_with_code_2():
    completed = subprocess.run(
        [sys.executable, "-m", "quorum"],
        input="{not-json",
        text=True,
        capture_output=True,
        cwd=ROOT,
        check=False,
    )

    assert completed.returncode == 2
    assert "error" in completed.stderr


def test_http_service_accepts_json_and_reports_validation_errors():
    server = build_server("127.0.0.1", 0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        payload = make_payload(
            [replica("a", 9), replica("b", 9)],
            read_threshold=9,
            write_threshold=9,
        )
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/analyze",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            assert response.status == 200
            assert json.loads(response.read()) == expected_analysis(payload)

        bad_request = urllib.request.Request(
            f"http://127.0.0.1:{port}/analyze",
            data=b"{not-json}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(bad_request, timeout=5)
        assert exc_info.value.code == 400

        invalid_request = urllib.request.Request(
            f"http://127.0.0.1:{port}/analyze",
            data=json.dumps(
                make_payload([replica("x"), replica("x")])
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(invalid_request, timeout=5)
        assert exc_info.value.code == 400
        assert "error" in json.loads(exc_info.value.read())
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
