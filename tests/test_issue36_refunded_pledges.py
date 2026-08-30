from scripts.analyze_nash_lindahl_fairness import game3_funded_payment_distance


def test_refunded_pledges_do_not_change_funded_payment_distance() -> None:
    costs = [10.0, 10.0]
    funded = [1]
    benchmark = {
        "Agent1": [0.0, 10.0],
        "Agent2": [0.0, 0.0],
    }
    without_refund = {
        "Agent1": [0.0, 10.0],
        "Agent2": [0.0, 0.0],
    }
    with_refund = {
        "Agent1": [25.0, 10.0],
        "Agent2": [40.0, 0.0],
    }

    clean = game3_funded_payment_distance(
        without_refund, benchmark, costs, funded
    )
    refunded = game3_funded_payment_distance(
        with_refund, benchmark, costs, funded
    )

    assert clean["lindahl_distance_norm"] == 0.0
    assert refunded["lindahl_distance_norm"] == 0.0
    assert refunded["actual_total_contribution"] == 10.0
    assert refunded["actual_total_pledged"] == 75.0
    assert refunded["refunded_pledges"] == 65.0
    assert refunded["overfunding"] == 0.0
