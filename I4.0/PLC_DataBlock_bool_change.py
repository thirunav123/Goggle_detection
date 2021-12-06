import snap7
from snap7.types import *

client=snap7.client.Client()
client.connect("10.73.11.212",0,1,102)
print(bool(client.get_connected))


print("Before change:")
data1=client.db_read(1,0,1)
print(data1)
data2=client.db_read(1,1,1)
print(data2)

print("After change:")
client.db_write(1,0,b'\x01')
client.db_write(1,1,b'\x01')
data1=client.db_read(1,0,1)
print(data1)
data2=client.db_read(1,1,1)
print(data2)

