import openai
import json

def call_openai_format_data(api_key, base_url, model, user_message, function_name, function_description, function_parameters):
    client = openai.OpenAI(api_key=api_key, base_url=base_url)

    messages = [{"role": "user", "content": user_message}]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        functions=[
            {
                "name": function_name,
                "description": function_description,
                "parameters": function_parameters,
            }
        ],
        function_call={"name": function_name}
    )

    if response.choices[0].message.function_call is not None:
        json_data = json.loads(response.choices[0].message.function_call.arguments)
        return json_data
    return None

# 示例调用
if __name__ == "__main__":
    api_key = "sk-LORwSL0exKUgxtlh6aDc3e9f49Bd4cE6969a7813Ef354336"
    base_url = "https://rerverseapi.workergpt.cn/v1"
    model = "gpt-4o-2024-08-06"
    user_message = "格式化输入的数据：小明今年18岁,是一名高三学生。他身高175cm,体重65kg。小明喜欢打篮球,每周都会和朋友一起打球。他的梦想是成为一名软件工程师。"
    function_name = "format_the_input_data"
    function_description = "格式化输入的数据"
    function_parameters = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "姓名"},
            "age": {"type": "number", "description": "年龄"},
            "education": {
                "grade": {"type": "string", "description": "年级"},
                "status": {
                    "type": "enum",
                    "enum": ["在读", "毕业", "休学", "退学"],
                    "description": "学习状态",
                }
            },
            "physicalAttribute": {
                "height": {"type": "string", "description": "身高"},
                "weight": {"type": "string", "description": "体重"}
            },
            "hobbies": {
                "type": "array",
                "items": {"type": "string", "description": "爱好"}
            },
            "frequency": {"type": "string", "description": "每周打球的频率"},
            "careerAspiration": {"type": "string", "description": "职业目标对应的职业名称"}
        },
        "required": ["name", "age", "education", "physicalAttribute", "hobbies", "frequency", "careerAspiration"],
    }

    json_data = call_openai_format_data(api_key, base_url, model, user_message, function_name, function_description, function_parameters)
    if json_data:
        with open("test.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(json_data, indent=4, ensure_ascii=False))