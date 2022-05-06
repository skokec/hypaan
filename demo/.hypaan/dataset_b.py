
def get_dataset_b_result_struct():
    return [
           'exp',
           'dataset_b',
           {
               'subfolder1' : ['subfolder=1','attribute2=abc'],
               'subfolder2' : ['subfolder=2','attribute2=xyz']
           },
           'subsubfolder',
           {
               '{}results_type1' : ['type=1'],
               '{}results_type2' : ['type=2']
           },
           'results.json'
       ]