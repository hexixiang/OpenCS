 
# 设置环境、定义自定义模块、编译模型以及使用提供的数据集和提词器配置严格评估其性能，利用GSM8K 数据集和 OpenAI GPT-3.5-turbo 模型来模拟 DSPy 中的提示任务
 
 
# 设置：在进入示例之前，我们先确保环境已正确配置。我们将从导入必要的模块并配置我们的语言模型开始
# 数据gsm8k_trainset集gsm8k_devset包含列表dspy.Examples，每个示例都有question和answer字段。
import dspy
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric
 
# 设置语言模型。
turbo = dspy.OpenAI(model='gpt-3.5-turbo-instruct', max_tokens=250)
dspy.settings.configure(lm=turbo)
 
# 从 GSM8K 数据集中加载数学问题。
gsm8k = GSM8K()
gsm8k_trainset, gsm8k_devset = gsm8k.train[:10], gsm8k.dev[:10]
print(gsm8k_trainset)
 
 
 
 
# 定义模块：设置环境后，我们定义一个自定义程序，利用 ChainOfThought 模块进行逐步推理生成答案
class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought("question -> answer")
 
    def forward(self, question):
        return self.prog(question=question)
 
 
# 编译并评估：继续使用BootstrapFewShot提词器对其进行编译
'''
请注意，这BootstrapFewShot不是优化提词器，即它只是创建并验证流程步骤的示例（在本例中为思路链推理），但不会优化指标。
其他提词器类似BootstrapFewShotWithRandomSearch并将MIPRO应用直接优化。
'''
from dspy.teleprompt import BootstrapFewShot  # noqa: E402
 
# 设置优化器：我们希望“引导”（即自生成）4-shot 示例的 CoT 程序。
config = dict(max_bootstrapped_demos=4, max_labeled_demos=4)
 
# 优化！这里使用 `gsm8k_metric`。通常，度量标准会告诉优化器它的表现如何。
teleprompter = BootstrapFewShot(metric=gsm8k_metric, **config)
optimized_cot = teleprompter.compile(CoT(), trainset=gsm8k_trainset)
 
 
 
# 评估：现在我们有了一个编译（优化）后的 DSPy 程序，让我们评估它在开发数据集上的表现
from dspy.evaluate import Evaluate  # noqa: E402
 
# 设置评估器，可以多次使用。
evaluate = Evaluate(devset=gsm8k_devset, metric=gsm8k_metric, num_threads=4, display_progress=True, display_table=0)
# 评估我们的 `optimized_cot` 程序。
evaluate(optimized_cot)
 
 
 
# 检查模型的历史记录：为了更深入地了解模型的交互，我们可以通过检查模型的历史记录来查看最近的生成结果
turbo.inspect_history(n=1)