from datasets import load_dataset, Dataset, concatenate_datasets, DatasetDict
import json
from zhconv import convert
from collections import defaultdict


def load_hf_dataset(data_id):
    '''
    加载huggingface上提供的数据集
    '''
    dataset = load_dataset(data_id)
    return dataset


def process_guanaco_dataset(examples):
    '''
    处理guanaco数据集，将中文转变为简体中文
    合并instruction和input为input
    '''
    instructions = examples['instruction']
    inputs = examples['input']
    outputs = examples['output']
    
    res = defaultdict(list)
    for instruction, input, output in zip(instructions, inputs, outputs):
        if input:
            input = instruction + " " + input
        else:
            input = instruction
        input = convert(input, 'zh-cn')
        output = convert(output, 'zh-cn')
        res['input'].append(input)
        res['target'].append(output)
    return res
    

def load_json_dataset(data_path):
    '''
    加载json类型数据集，包含instruction、input和output属性
    '''
    texts = []
    with open(data_path, 'r') as f:
        json_lines = json.load(f)
    for item in json_lines:
        instruction = item['instruction']
        input = item['input']
        output = item['output']
        if input:
            prompt = instruction + " " + input
        else:
            prompt = instruction
        texts.append({'input': prompt, 'target': output})
    return texts


def group_text(examples):
    '''
    处理数据，包含繁体中文转为简体，以及去除超长文本。
    '''
    inputs = examples['input']
    targets = examples['target']
    
    res = defaultdict(list)
    for input, target in zip(inputs, targets):
        input = convert(input, 'zh-cn')
        target = convert(target, 'zh-cn')
        
        text = "User: {}\nAI: {}".format(input, target)
        if len(text) < 500:
            res['instruction'].append(input)
            res['output'].append(target)
    return res


# belle chinese dataset
belle_id1 = "BelleGroup/generated_train_1M_CN"
belle_dataset1 = load_hf_dataset(belle_id1)['train']
print("belle1 data info: ", belle_dataset1)

# belle chinese dataset
belle_id2 = "BelleGroup/generated_train_0.5M_CN"
belle_dataset2 = load_hf_dataset(belle_id2)['train']
print("belle2 data info: ", belle_dataset2)

# guanaco dataset
guanaco_path = "../datasets/chinese_guanaco.json"
guanaco_dataset = load_json_dataset(guanaco_path)
guanaco_dataset = Dataset.from_list(guanaco_dataset)
print("guanaco data info: ", guanaco_dataset)

# alpaca chinese dataset 
alpaca_path1 = "../datasets/trans_chinese_alpaca_data.json"
alpaca_dataset1 = load_json_dataset(alpaca_path1)
# print(len(alpaca_dataset1))

# alpaca english dataset
alpaca_path2 = "../datasets/alpaca_en_data.json"
alpaca_dataset2 = load_json_dataset(alpaca_path2)
# print(len(alpaca_dataset3))

alpaca_dataset = alpaca_dataset1 + alpaca_dataset2
alpaca_dataset = Dataset.from_list(alpaca_dataset)
print("alpaca data info: ", alpaca_dataset)

instruction_dataset = concatenate_datasets([belle_dataset1, belle_dataset2, guanaco_dataset, alpaca_dataset])
instruction_dataset = instruction_dataset.map(group_text,
                                              batched=True,
                                              batch_size=50,
                                              num_proc=10,
                                              remove_columns=instruction_dataset.column_names)

instruction_df = instruction_dataset.to_pandas()
print("instruction df shape: {}".format(instruction_df.shape))
instruction_df = instruction_df.drop_duplicates(subset=['instruction', 'output'], keep='first', inplace=False)
print("new instruction df shape: {}".format(instruction_df.shape))
instruction_dataset = Dataset.from_pandas(instruction_df)

instruction = DatasetDict({'train': instruction_dataset})
print("instruction data info: ", instruction)

save_path = "../instructions/"
instruction.save_to_disk(save_path)