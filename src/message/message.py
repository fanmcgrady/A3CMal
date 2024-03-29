from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.request import CommonRequest

class Message:
    def __init__(self):
        key = ''
        secret = ''

        self.client = AcsClient(key, secret, 'cn-hangzhou')

    def send(self, experment, result):
        # 模版内容:
        # ${experment}的实验结果为：${result}
        contents = "{" + "'experment': '{}', 'result': '{}'".format(experment, result) + "}"
        # 方老师 = 
        # 耿大宝 = 
        self.__sendTo('', contents)
        self.__sendTo('', contents)

    def __sendTo(self, phone, contents):
        request = CommonRequest()
        request.set_accept_format('json')
        request.set_domain('dysmsapi.aliyuncs.com')
        request.set_method('POST')
        request.set_protocol_type('https')  # https | http
        request.set_version('2017-05-25')
        request.set_action_name('SendSms')

        request.add_query_param('RegionId', "cn-hangzhou")
        request.add_query_param('PhoneNumbers', phone)
        request.add_query_param('SignName', "四川大学网络空间安全学院")
        request.add_query_param('TemplateCode', "SMS_185812485")
        request.add_query_param('TemplateParam', contents)
        print(contents)

        response = self.client.do_action(request)
        print(str(response, encoding='utf-8'))

# if __name__ == '__main__':
#     msg = Message()
#     msg.send('novel', '0.902')
