# 基本流程介绍

## 总体原则

先局部测试，做成类似jupyter那种的pipline,之后逐步封装，明确input,output写成批量脚本

## 读取tag数据

### tag数据格式

```json
{"data": {"instruction": "What does DNA stand for?", "input": "", "output": "DNA stands for deoxyribonucleic acid."}, "key_0": "[\n    {\"tag\": \"Biology\", \"explanation\": \"The instruction is asking for the full form of DNA, which is a biological term.\"},\n    {\"tag\": \"General Knowledge\", \"explanation\": \"The instruction seeks information that is commonly known or should be known by a well-informed person.\"}\n]"}
```

```json
{
    "data": {
      "instruction": "What is the capital of France?",
      "input": "",
      "output": "The capital of France is Paris."
    },
    "index": 11,
    "tags": [
      {
        "tag": "Geography Inquiry",
        "explanation": "The user is asking for information related to the geographical location of the capital city of France."
      }
    ]
}
```

## 基于频率的tag过滤

## 基于规则的过滤

## 基于语义聚类

## 基于关联分析的聚合

## 基于tag的聚类

这里再看一下论文的数据筛选是怎么做的

## 输出结果

之后拿这个数据集进行微调

## 当前问题

（1）alpha与k的调参——alpha决定筛除多少tag,k表征tag的话题数量（但是k不好定，alpha倒是随便筛，根据数据分布以及需要多少数据搞）

（2）得到的标签没有一种导向性，即没有人的干预引导，聚出来的或生成的tag不一定是我们想要的（目前是保证了话题的多样性，但是不能保证任务的多样性），即粒度粗细不好把握（可借鉴teacher-student范式：让teacher设计课程表，然后细化，预生成标签）

## 怎么使用

1. 运行single_pipline.py
2. 运行la
