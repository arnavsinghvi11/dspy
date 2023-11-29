import random
import json
from dspy.datasets.dataset import Dataset
from collections import defaultdict

from collections import defaultdict
import json

class QuoteSum(Dataset):  # Assuming Dataset is the base class with the provided functionality
    def __init__(self, path, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        def load_data(file_path):
            with open(file_path, 'r') as file:
                return [json.loads(line) for line in file]
            
        def preprocess_dataset(dataset):
            grouped = defaultdict(lambda: {'question': None, 'entries': []})
            for entry in dataset:
                qid = entry['qid']
                # Skip entries with 'AMBIG_val' in the qid
                if 'AMBIG_val' in qid:
                    continue
                # Initialize 'question' once per qid group
                if grouped[qid]['question'] is None:
                    grouped[qid]['question'] = entry['question']
                # Add the individual entry, excluding 'qid' and 'question'
                individual_entry = {k: v for k, v in entry.items() if k not in ['qid', 'question']}
                grouped[qid]['entries'].append(individual_entry)
            # Create a list with the question and associated entries
            preprocessed_dataset = [{'question': data['question'], 'entries': data['entries']} for qid, data in grouped.items()]
            return preprocessed_dataset
        
        self._train = preprocess_dataset(load_data(f"{path}/train.jsonl"))
        self._dev = preprocess_dataset(load_data(f"{path}/dev.jsonl"))
        self._test = preprocess_dataset(load_data(f"{path}/test.jsonl"))

        # # For debugging
        # print(self._train[0])




    # The train, dev, and test properties will be handled by the base class
    # You can add additional methods as needed

if __name__ == '__main__':
    from dsp.utils import dotdict

    path = '../../QuoteSum/v1/'
    data_args = dotdict(train_seed=1, train_size=300, eval_seed=2023, dev_size=300, test_size=0)
    dataset = QuoteSum(path, **data_args)

    # Example usage
    print(dataset.train[0]['question'])
    print(dataset.train[0]['summary'])
    # Add more print statements as needed