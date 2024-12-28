import threading
import requests

class MyLogger:
    def __init__(self, server_url:str = None, server_username:str = None, server_folder:str = None, model_name:str = None, path_localFile:str = None):
        self.SERVER_URL = server_url
        self.SERVER_USERNAME = server_username
        self.SERVER_FOLDER = server_folder
        self.MODEL_NAME = model_name
        self.PATH_LOCAL_LOG_FILE = path_localFile
        self.IS_SERVER_WORKING = True
        self._lock1 = threading.Lock()
        self._lock2 = threading.Lock()

    def _local_write(self, string):
        with self._lock2:
            with open(self.PATH_LOCAL_LOG_FILE, "a") as log_file:
                log_file.write(string + "\n")
    
    def _server_write(self, data):
        with self._lock1:
            if(self.IS_SERVER_WORKING):
                try:
                    response = requests.post(url=self.SERVER_URL, data=data)
                    if response.status_code != 200:
                        print(f"Error Writing on Monitor Server: {response.status_code}")
                        print(f"Response: {response.text}")
                        self.IS_SERVER_WORKING = False
                    else:
                        self.IS_SERVER_WORKING = True
                except Exception as e:
                    if(self.IS_SERVER_WORKING):
                        print(f"Error while logging on server: {e}")
                        self.IS_SERVER_WORKING = False

    def log(self, string):
        # print to console
        print(string)
        
        # write to local file
        if(self.PATH_LOCAL_LOG_FILE is not None and self.PATH_LOCAL_LOG_FILE != ""):
            local_write_thread = threading.Thread(target = self._local_write, args=(string, ))
            local_write_thread.start()
        
        # write on server
        data = {
            "msg": str(string),
            "main_folder": self.SERVER_FOLDER,
            "model_name": self.MODEL_NAME,
            "user_name": self.SERVER_USERNAME
        }
        if self.SERVER_URL is not None and self.SERVER_URL != "":
            server_write_thread = threading.Thread(target = self._server_write, args=(data, ))
            server_write_thread.start()


if __name__ == "__main__":
    
    SERVER_USERNAME = "AaratSatsangi"
    SERVER_FOLDER = "Test"
    SERVER_URL = "https://www.aaratsatsangi.in/logger.php"
    MODEL_NAME = "TestModel"
    
    logger = MyLogger(
        server_url = SERVER_URL,
        server_username= SERVER_USERNAME,
        server_folder = SERVER_FOLDER,
        model_name = MODEL_NAME
    )
    logger.log("hello world!")
    logger.log("HELLO WORLD 2!!")