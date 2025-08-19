import os
import sys
import requests
parent_dir = os.path.dirname(os.path.abspath(__file__))
up_dir = parent_dir
for i in range(3):
    sys.path.append(up_dir)
    up_dir = os.path.dirname(up_dir)
from kutils import DEBUG, INFO, WARN, ERROR
import utils as u

class CommonHttpRequestor():
    def __init__(self, url):
        self.url = url
        self.models = ['Qwen2_5_72B_Instruct']
        self.body = \
        {
            "msg":"",
            "model":""
        }
        # self.body = \
        # {
        #     "functionName":"myjf.common.sofaaifunction.demo.actllm.actmaya",
        #     "env": "PRE",
        #     "params": {
        #         "msg":"",
        #         "model":""
        #     },
        # }

        self.headers = {
            "Content-Type": "application/json;charset=utf-8",
        }
        self.timeout = 30

    def request(self, prompt, model):
        self.body['msg'] = prompt
        self.body['model'] = model
        r = requests.post(url=self.url, json=self.body, headers=self.headers)
        res = r.json()
        return res

def request_common_http(prompt, model):
    common = CommonHttpRequestor("https://actmng-gray.alipay.com/maya/call.json")
    result = common.request(prompt, model)
    if 'data' in result:
        return result['data']['result']
    else:
        raise ValueError(result)

if __name__ == "__main__":
    result = request_common_http("system - role_definition: 你是一个planner。你可以访问以下工具：\njson_reader(file: str) -> str - Read json file, args: {'file': {'title': 'File', 'type': 'string'}}\ntxt_reader(file: str) -> str - Read txt file, args: {'file': {'title': 'File', 'type': 'string'}}。\n你可以选择是否使用工具来帮助你得到更多信息，从而回答问题。\n如果你判断要使用工具，则返回如下JSON格式：\n{\n    \"tool_calls\": [\n        {\n            \"name\": \"\",\n            \"args\": {\n                <arg 1>: \"\",\n                <arg 2>: \"\"\n                ...\n            }\n        }\n    ]\n}\n其中name是工具名称，args是工具的参数，需要按照你想调用的工具参数格式写好。\n如果你判断不需要工具，则直接正常回复即可。\n需要调用多个工具的话一起并行调用。\n如果你觉得当前任务已经完成，就在回答最后写上\"__FINISH__\"。\n如果你的工具已经返回，就执行后续任务，不要重复调用工具!!!!!\n如果你的工具已经返回，就执行后续任务，不要重复调用工具!!!!!\n如果你的工具已经返回，就执行后续任务，不要重复调用工具!!!!!\n读取并理解示例数据/Users/kevin/kevin_ai_studio_repo//ca_py_simulation/rpa_simulation_20241106//Manner咖啡_training_data.json，然后帮下游程序员思考程序流程。\n要求：\n1. 只写步骤，不写其他内容\n2. 步骤要明确分为点餐前、门店选择、点餐时、点餐后四个部分\n3. 点餐前、门店选择、点餐后都需要if判断在哪个页面，则点击什么按钮，点餐时不需要判断页面\n4. 点餐前按钮名称和顺序应当与用例中完全一致\n5. 门店选择流程先判断页面名称，再按照json中的店铺名称点击\n6. 点餐时，根据query_json格式，从点击套餐，到点击餐点，再到点击餐点属性，最后点加入购物车或者类似操作\n7. 点餐后按钮名称应当与用例中完全一致\n8. 你输出的步骤要兼容所有用户的动作，你只能使用页面名作为条件进行判断，不要列举按钮名称\n\nassistant - master: 根据订单内容，写下程序执行步骤，包括读取订单信息，定位按钮，以及点击按钮的逻辑。\ntool - json_reader: json_reader{'file': '/Users/kevin/kevin_ai_studio_repo//ca_py_simulation/rpa_simulation_20241106//Manner咖啡_training_data.json'}返回：\n{\"Manner咖啡\": [{\"data_id\": \"6100192\", \"query\": {\"shopname\": \"杭州阿里云谷店\", \"order\": [{\"meal_set_name\": \"1份汤力美式\", \"meals\": [{\"meal_name\": \"汤力美式\", \"attr\": [{\"attribute_name\": \"杯型\", \"value\": \"超大冰杯 473ml\"}, {\"attribute_name\": \"冰度\", \"value\": \"正常冰\"}]}]}, {\"meal_set_name\": \"1份拿铁咖啡 Latte\", \"meals\": [{\"meal_name\": \"拿铁咖啡 Latte\", \"attr\": [{\"attribute_name\": \"杯型\", \"value\": \"小热杯 237ml\"}, {\"attribute_name\": \"奶搭配\", \"value\": \"牛奶\"}]}]}]}, \"actions\": {\"点餐前\": \"在小程序首页,点击开始点单;\", \"门店选择\": \"在商品列表页面,点击请选择下单门店;在选择地区页面,点击杭州市;在选择地区页面,点击西湖区;在选择具体门店页面,点击未获取到您的定位，去授权;在选择具体门店页面,点击去下单;\", \"点餐\": \"在超大杯系列商品选择页面,点击汤力美式;在商品详情页面,点击加入购物袋;在意式咖啡系列商品选择页面,点击拿铁咖啡 Latte;在商品详情页面,点击小热杯237ml;在商品规格页面,点击加入购物袋;\", \"点餐后\": \"在意式咖啡系列商品选择页面,点击结算;在确认支付页面,点击支付;\"}}, {\"data_id\": \"6100190\", \"query\": {\"shopname\": \"杭州蚂蚁A空间店\", \"order\": [{\"meal_set_name\": \"1份清橙风味拿铁（超大杯）\", \"meals\": [{\"meal_name\": \"清橙风味拿铁（超大杯）\", \"attr\": [{\"attribute_name\": \"杯型\", \"value\": \"冰杯473ml\"}, {\"attribute_name\": \"奶搭配\", \"value\": \"牛奶\"}, {\"attribute_name\": \"冰度\", \"value\": \"正常冰\"}]}]}, {\"meal_set_name\": \"1份玫瑰风味冰拿铁\", \"meals\": [{\"meal_name\": \"玫瑰风味冰拿铁\", \"attr\": [{\"attribute_name\": \"杯型\", \"value\": \"冰杯473ml\"}, {\"attribute_name\": \"奶搭配\", \"value\": \"牛奶\"}, {\"attribute_name\": \"冰度\", \"value\": \"正常冰\"}]}]}, {\"meal_set_name\": \"1份冰橙美式\", \"meals\": [{\"meal_name\": \"冰橙美式\", \"attr\": [{\"attribute_name\": \"杯型\", \"value\": \"超大冰杯 473ml\"}, {\"attribute_name\": \"冰度\", \"value\": \"正常冰\"}]}]}]}, \"actions\": {\"点餐前\": \"在开始点单页面,点击开始点单;\", \"门店选择\": \"\", \"点餐\": \"在季节限定商品选择页面,点击清橙风味拿铁（超大杯）;在清橙风味拿铁（超大杯）规格选择页面,点击加入购物袋;在点单页面,点击玫瑰风味冰拿铁;在玫瑰风味冰拿铁规格选择页面,点击加入购物袋;在超大杯商品选择页面,点击冰橙美式;在冰橙美式规格选择页面,点击加入购物袋;\", \"点餐后\": \"在点单页面,点击结算;在订单结算页面,点击支付;\"}}]}\n", 'Qwen2_5_72B_Instruct')
    # result = request_common_http("你好", 'Qwen2_5_72B_Instruct')
    DEBUG(result)