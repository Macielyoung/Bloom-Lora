# Bloom-Lora
### 项目介绍

受到[BELLE项目](https://github.com/LianjiaTech/BELLE)的启发，我们希望基于Bloomz+instrcution数据构建一个中文指令问答模型，同时可以根据自己领域内任务生成相关数据，训练出领域相关的指令问答模型。

项目包括：

- 公开中文指令数据集
- 根据自定义种子任务生成相关数据
- finetune模型代码
- 模型推理代码



### 公开中文指令数据集

以下是公开的一些中文指令数据集。

| 数据集                     | 下载                                                         | 项目                                                         |
| -------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Zhihu-KOL                  | [Zhihu-dataset](https://huggingface.co/datasets/wangrui6/Zhihu-KOL) | [Zhihu-KOL](https://github.com/wangrui6/Zhihu-KOL)           |
| Guanaco                    | [Guanaco-dataset](https://huggingface.co/datasets/JosephusCheung/GuanacoDataset) | [Guanaco](https://guanaco-model.github.io/)                  |
| Alpaca-Traditional-Chinese | [Alpaca-dataset](https://github.com/ntunlplab/traditional-chinese-alpaca/tree/main/data) | [Traditional-Chinese-Alpaca](https://github.com/ntunlplab/traditional-chinese-alpaca) |
| Alpaca-Chinese             | [Alpaca-Chinese-dataset](https://github.com/hikariming/alpaca_chinese_dataset/tree/main/%E7%BF%BB%E8%AF%91%E5%90%8E%E7%9A%84%E4%B8%AD%E6%96%87%E6%95%B0%E6%8D%AE) | [alpaca-chinese-dataset](https://github.com/hikariming/alpaca_chinese_dataset) |
| Belle-0.5M                 | [Belle-0.5M](https://huggingface.co/datasets/BelleGroup/train_0.5M_CN) | [BELLE](https://github.com/LianjiaTech/BELLE)                |
| Belle-1M                   | [Belle-1M](https://huggingface.co/datasets/BelleGroup/train_1M_CN) | [BELLE](https://github.com/LianjiaTech/BELLE)                |
| Belle-Multiturn-0.8M       | [Belle-multiturn-0.8M](https://huggingface.co/datasets/BelleGroup/multiturn_chat_0.8M) | [BELLE](https://github.com/LianjiaTech/BELLE)                |
| Belle-Math-0.25M           | [Belle-Math-0.25M](https://huggingface.co/datasets/BelleGroup/school_math_0.25M) | [BELLE](https://github.com/LianjiaTech/BELLE)                |

数据集主要包含三个字段，instruction、input和output。

```json
instruction： 指令
input：输入
output：输出

案例1:
{
		"instruction": "列举电动车相对传统汽油车的优点。",
		"input": "",
		"output": "电动车相对传统汽油车的优点包括：\n1. 更低的排放和更高的能源效率 - 电动车所产生的有害排放物质远少于汽油车，并且它们的能源利用效率更高。\n2. 更低的维护成本 - 电动车需要更少的保养和通常拥有较少的运动部件，从而降低了总体维护成本。\n3. 更低的燃料成本 - 电动车需要比汽油车少得多的燃料，因此随着时间的推移，可以节省成本。\n4. 更长的续航里程 - 电动车单次充电可以行驶比汽油车更远的距离，非常适合长途通勤。\n5. 更为安静的运行 - 电动车比汽油车要安静得多，使驾驶更加愉悦。"
}

案例2:
{
		"instruction": "对以下两个句子进行分类。",
		"input": "- 那场演出太棒了\n- 我永远不会再见到他们",
		"output": "积极：那场演出太棒了\n消极：我永远不会再见到他们"
}
```



### 自定义任务

如果想要结合自己的任务领域，那也可以提供种子任务，使用GPT接口生成相关数据集。种子任务提供在customization目录下。

生成脚本如下：

```shell
export OPENAI_API_KEY=your_openai_key

# 使用GPT3的接口生成数据
python3 generate_instruction.py generate_instruction_following_data --num_instructions_to_generate=10000

# 使用GPT3.5的接口生成数据
python3 generate_instruction.py generate_instruction_following_data --api=chat --model_name=gpt-3.5-turbo --num_instructions_to_generate=10000
```



### 模型微调

模型微调使用[Lora方法](https://arxiv.org/pdf/2106.09685.pdf)来撬动大模型，通过一些指令数据集来更好的理解用户指令，带动大模型给出相应的回答。

```shell
# 处理指令数据集(processor目录下)
python processing.py

# 模型微调(trainer目录下)
deepspeed --include=localhost:0,1 deepspeed_finetune_lora.py
```



### 模型推理

微调好模型后，可以加载lora和大模型进行推理，查看效果。

```python
python prediction.py
```



### 参考项目

1. https://github.com/LianjiaTech/BELLE

2. https://github.com/hikariming/alpaca_chinese_dataset

3. https://github.com/Facico/Chinese-Vicuna

4. https://github.com/mymusise/ChatGLM-Tuning

   
