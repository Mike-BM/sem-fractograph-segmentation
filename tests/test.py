
def test_basic_python_operation():
    assert 2 + 2 == 4


def test_project_directories_can_be_created(tmp_path):
    raw_dir = tmp_path / "data" / "raw"
    output_dir = tmp_path / "outputs"

    raw_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    assert raw_dir.is_dir()
    assert output_dir.is_dir()
