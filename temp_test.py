import time

from pymysqlpool.pool import Pool

HOST = "172.24.30.6"
PORT = 3306
USER = "root"
PASSWORD = "jianke@test123"
DB = "sdxl"
pool = Pool(host=HOST, port=PORT, user=USER, password=PASSWORD, db=DB, min_size=3, max_size=5, autocommit=True,
            init_command="SET SESSION TRANSACTION ISOLATION LEVEL READ COMMITTED", ping_check=1)
pool.init()
connection2 = pool.get_conn()
cur2 = connection2.cursor()
while True:
    time.sleep(0.01)
    res = cur2.execute("UPDATE busy SET isbusy = 1 WHERE isbusy = 0")
    print(res)
    if res != 0:
        break
