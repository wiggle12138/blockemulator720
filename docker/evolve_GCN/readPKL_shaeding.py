import pickle

# 以二进制模式打开节点嵌入.pkl文件并加载数据
with open('../outputs/sharding_results.pkl', 'rb') as file:  # 'rb'表示二进制读取模式
    data = pickle.load(file)  # 反序列化数据

for key, value in data.items():
    print(f"{key}: {value}")  # 打印每个键值对

# print(data)