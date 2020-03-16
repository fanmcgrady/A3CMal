import configparser


class ReadConfig:
    def __init__(self, filepath='message/config.ini'):
        self.cf = configparser.ConfigParser()
        self.cf.read(filepath)

    def get_db(self, param):
        value = self.cf.get("Database", param)
        return value

    def get_key(self, param):
        value = self.cf.get("Message", param)
        return value

    def get_users(self):
        return self.cf.items("Users")