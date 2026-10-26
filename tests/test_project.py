def test_project_directories_can_be_created(tmp_path):
    raw_dir = tmp_path / "data" / "raw"
    output_dir = tmp_path / "outputs"

    raw_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    assert raw_dir.is_dir()
    assert output_dir.is_dir()


def test_binary_mask_contains_only_zero_and_one():
    import numpy as np

    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)

    assert mask.shape == (2, 2)
    assert set(np.unique(mask)).issubset({0, 1})


def test_binary_mask_pixel_count():
    import numpy as np

    mask = np.array([[0, 1], [1, 1]], dtype=np.uint8)

    assert np.count_nonzero(mask) == 3
    assert mask.size == 4



import numpy as np


def test_perfect_mask_overlap():
    ground_truth = np.array([[1, 0], [1, 0]], dtype=bool)
    prediction = np.array([[1, 0], [1, 0]], dtype=bool)

    intersection = np.logical_and(ground_truth, prediction).sum()
    union = np.logical_or(ground_truth, prediction).sum()

    iou = intersection / union
    dice = 2 * intersection / (
        ground_truth.sum() + prediction.sum()
    )

    assert iou == 1.0
    assert dice == 1.0


def test_no_overlap_between_masks():
    ground_truth = np.array([[1, 1], [0, 0]], dtype=bool)
    prediction = np.array([[0, 0], [1, 1]], dtype=bool)

    intersection = np.logical_and(ground_truth, prediction).sum()
    union = np.logical_or(ground_truth, prediction).sum()

    assert intersection == 0
    assert intersection / union == 0.0


def test_mask_shape_mismatch_is_detected():
    ground_truth = np.zeros((10, 10), dtype=bool)
    prediction = np.zeros((8, 10), dtype=bool)

    assert ground_truth.shape != prediction.shape



from src.validation_metrics import calculate_metrics


def test_perfect_segmentation_metrics():
    truth = np.array([[1, 0], [1, 0]])
    prediction = np.array([[1, 0], [1, 0]])

    metrics = calculate_metrics(truth, prediction)

    assert metrics["iou"] == 1.0
    assert metrics["dice"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0


def test_false_positive_and_false_negative():
    truth = np.array([[1, 1], [0, 0]])
    prediction = np.array([[1, 0], [1, 0]])

    metrics = calculate_metrics(truth, prediction)

    assert metrics["true_positive"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["iou"] == 1 / 3
    assert metrics["dice"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5


def test_different_mask_shapes_raise_error():
    truth = np.zeros((10, 10), dtype=bool)
    prediction = np.zeros((8, 10), dtype=bool)

    try:
        calculate_metrics(truth, prediction)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for mismatched mask shapes")
