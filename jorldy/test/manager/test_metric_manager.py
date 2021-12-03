from manager.metric_manager import MetricManager


def test_metric_manager():
    metric_manager = MetricManager()

    mock_results = [
        {
            "mock_metric1": 1.0,
            "mock_metric2": 2.0,
        },
        {
            "mock_metric1": 1.0,
            "mock_metric2": 2.0,
        },
        {
            "mock_metric1": 7.0,
            "mock_metric2": -1.0,
        },
    ]

    # test append
    for mock_result in mock_results:
        metric_manager.append(result=mock_result)

    # test get_statistics
    statistics = metric_manager.get_statistics()
    assert statistics["mock_metric1"] == round(9.0 / 3, 4)
    assert statistics["mock_metric2"] == round(3.0 / 3, 4)

    mock_results = [
        {
            "mock_metric2": 2.0,
            "mock_metric3": 3.0,
        },
        {
            "mock_metric2": 2.0,
            "mock_metric3": 3.0,
        },
        {
            "mock_metric2": 5.0,
            "mock_metric3": 6.0,
        },
    ]
    # test append
    for mock_result in mock_results:
        metric_manager.append(result=mock_result)

    # test get_statistics
    statistics = metric_manager.get_statistics()
    assert statistics["mock_metric2"] == round(9.0 / 3, 4)
    assert statistics["mock_metric3"] == round(12.0 / 3, 4)

    # test clear
    assert not "mock_metric1" in statistics.keys()
