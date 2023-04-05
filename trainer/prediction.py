import torch
  
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "bigscience/bloomz-7b1-mt"
LORA_WEIGHTS = "bloom-lora"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16, # 加载半精度
        device_map={"":0}, # 指定GPU 0
    )
model.eval()

# 加载LoRA权重
model = PeftModel.from_pretrained(model, LORA_WEIGHTS, torch_dtype=torch.float16)
model.half()

history = []
while True:
    inputs = input("请输入你的问题: ")
    instruction = ""
    for pair in history:
        his_inputs = pair.get("inputs")
        his_outputs = pair.get("outputs")
        instruction += "用户: {}\n系统: {}\n".format(his_inputs, his_outputs)
    instruction += "用户: {}\n系统: ".format(inputs)
    encodings = tokenizer(instruction, max_length=512, return_tensors="pt").to("cuda")
    outputs = model.generate(input_ids=encodings["input_ids"], max_new_tokens=256)
    preds = tokenizer.decode(outputs[0])
    print(preds)
    item = {
        "inputs": inputs.strip(),
        "outputs": preds.strip()
    }
    history.append(item)