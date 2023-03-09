from data import load_data

def test_yaml():
    # Depends on being on Nero to run
    path_to_dataset_config = '../prompts/test.yaml'
    path_to_dataset_dir = '/local-scratch/nigam/projects/clinical_llm/data/mednli'
    dataset, tasks, is_classification = load_data(path_to_dataset_config, path_to_dataset_dir)
    assert is_classification, f"Expected classification task, got is_classification={is_classification}"
    assert dataset.num_rows['train'] == 11232, f"Expected 11232 rows, got {dataset.num_rows['train']}"
    assert dataset.num_rows['test'] == 1422, f"Expected 1422 rows, got {dataset.num_rows['test']}"
    assert dataset.num_rows['validation'] == 1395, f"Expected 1395 rows, got {dataset.num_rows['validation']}"
    assert dataset.column_names['train'] == ['id', 'premise', 'hypothesis', 'label'], f"Expected ['id', 'premise', 'hypothesis', 'label'], got {dataset.column_names['train']}"
    # Task 1
    assert tasks[0].label_map == {'entailment': ['yes', 'true'], 'not entailment': ['no', 'false'] }
    assert tasks[0].template == '{{premise}} Therefore, we are licensed to say that {{hypothesis}} yes or no'
    assert tasks[0].output_template == 'Here is my output: {{output}}'
    assert tasks[0].output_column == 'label'
    # Task 2
    # Test `outcome_template` default
    assert tasks[1].output_template == '{{output}}'
    # Test blank terms
    assert tasks[1].label_map == {'entailment': [], 'not entailment': [] }
    # Task 3
    # Test loading 'yes' and 'no' as strings
    assert tasks[2].label_map == {'yes': ['true'], 'no': ['no', 'false'] }

if __name__ == '__main__':
    test_yaml()