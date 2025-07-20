#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
区块链分片负载均衡数据集生成器
生成针对性能异构分片环境的交易数据集

实验目标：
1. 验证负载均衡算法在性能异构环境中的表现
2. 测试分片间性能差异对系统整体吞吐量的影响
3. 观察交易分配策略对性能洼地分片的影响

分片性能设置（对应docker-compose.yml）：
- 分片0: 0.5 CPU, 512MB 内存 (性能洼地)
- 分片1: 1.0 CPU, 1GB 内存   (中低性能)
- 分片2: 1.5 CPU, 2GB 内存   (中高性能)  
- 分片3: 2.0 CPU, 4GB 内存   (高性能)

数据集组成：
1. 基础交易(40%): 模拟正常业务交易
2. 热点交易(25%): 模拟热门地址的高频交易
3. 跨分片交易(20%): 测试跨分片通信开销
4. 批量交易(10%): 模拟批处理场景
5. 压力测试交易(5%): 高价值交易测试极限性能
"""

import csv
import random
import hashlib
import time
from datetime import datetime
from typing import List, Tuple, Dict

class TransactionGenerator:
    def __init__(self):
        self.num_shards = 4
        self.addresses_per_shard = 5000  # 每分片5000地址
        self.total_addresses = self.num_shards * self.addresses_per_shard
        
        # 生成地址池
        self.addresses = self._generate_addresses()
        
        # 分片地址映射
        self.shard_addresses = {
            i: self.addresses[i * self.addresses_per_shard:(i + 1) * self.addresses_per_shard]
            for i in range(self.num_shards)
        }
        
        # 热点地址 - 每分片选择100个作为热点
        self.hot_addresses = {
            i: random.sample(self.shard_addresses[i], 100)
            for i in range(self.num_shards)
        }
        
        # 交易计数器
        self.tx_counter = 0
        
    def _generate_addresses(self) -> List[str]:
        """生成地址池"""
        addresses = []
        for i in range(self.total_addresses):
            # 生成42字符的以太坊风格地址
            hash_input = f"address_{i}_{time.time()}"
            addr_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:40]
            addresses.append(f"0x{addr_hash}")
        return addresses
    
    def _get_address_shard(self, address: str) -> int:
        """根据地址确定所属分片"""
        for shard_id, shard_addrs in self.shard_addresses.items():
            if address in shard_addrs:
                return shard_id
        return 0  # 默认分片0
    
    def _generate_amount(self, tx_type: str) -> int:
        """根据交易类型生成金额（wei）"""
        amounts = {
            'basic': random.randint(1000000000000000000, 100000000000000000000),      # 1-100 ETH
            'hot': random.randint(100000000000000000, 10000000000000000000),         # 0.1-10 ETH
            'cross': random.randint(5000000000000000000, 50000000000000000000),      # 5-50 ETH
            'batch': random.randint(500000000000000000, 5000000000000000000),        # 0.5-5 ETH
            'stress': random.randint(100000000000000000000, 1000000000000000000000), # 100-1000 ETH
        }
        return amounts.get(tx_type, amounts['basic'])
    
    def generate_basic_transactions(self, count: int) -> List[Dict]:
        """生成基础交易 - 模拟正常业务交易"""
        transactions = []
        
        for _ in range(count):
            # 80%分片内交易，20%跨分片交易
            if random.random() < 0.8:
                # 分片内交易
                shard_id = random.randint(0, 3)
                from_addr = random.choice(self.shard_addresses[shard_id])
                to_addr = random.choice(self.shard_addresses[shard_id])
            else:
                # 跨分片交易
                from_addr = random.choice(self.addresses)
                to_addr = random.choice(self.addresses)
                # 确保跨分片
                while self._get_address_shard(from_addr) == self._get_address_shard(to_addr):
                    to_addr = random.choice(self.addresses)
            
            tx = {
                'type': 'basic',
                'from': from_addr,
                'to': to_addr,
                'amount': self._generate_amount('basic'),
                'gas_limit': random.randint(21000, 100000),
                'gas_price': random.randint(20000000000, 100000000000),  # 20-100 Gwei
            }
            transactions.append(tx)
            
        return transactions
    
    def generate_hot_transactions(self, count: int) -> List[Dict]:
        """生成热点交易 - 模拟热门地址的高频交易"""
        transactions = []
        
        for _ in range(count):
            # 选择热点地址参与的交易
            shard_id = random.randint(0, 3)
            hot_addr = random.choice(self.hot_addresses[shard_id])
            
            # 50%热点地址作为发送方，50%作为接收方
            if random.random() < 0.5:
                from_addr = hot_addr
                to_addr = random.choice(self.shard_addresses[shard_id])
            else:
                from_addr = random.choice(self.shard_addresses[shard_id])
                to_addr = hot_addr
            
            tx = {
                'type': 'hot',
                'from': from_addr,
                'to': to_addr,
                'amount': self._generate_amount('hot'),
                'gas_limit': random.randint(21000, 80000),
                'gas_price': random.randint(30000000000, 150000000000),  # 30-150 Gwei
            }
            transactions.append(tx)
            
        return transactions
    
    def generate_cross_shard_transactions(self, count: int) -> List[Dict]:
        """生成跨分片交易 - 测试跨分片通信开销"""
        transactions = []
        
        # 定义跨分片模式
        cross_patterns = [
            (0, 1), (0, 2), (0, 3),  # 从性能洼地到其他分片
            (1, 2), (1, 3),          # 中低到中高、高性能
            (2, 3),                  # 中高到高性能
            (3, 0), (2, 0), (1, 0),  # 反向：高性能到性能洼地
        ]
        
        for _ in range(count):
            from_shard, to_shard = random.choice(cross_patterns)
            from_addr = random.choice(self.shard_addresses[from_shard])
            to_addr = random.choice(self.shard_addresses[to_shard])
            
            tx = {
                'type': 'cross',
                'from': from_addr,
                'to': to_addr,
                'amount': self._generate_amount('cross'),
                'gas_limit': random.randint(50000, 200000),  # 跨分片交易gas消耗更高
                'gas_price': random.randint(50000000000, 200000000000),  # 50-200 Gwei
                'cross_shard': True,
                'from_shard': from_shard,
                'to_shard': to_shard,
            }
            transactions.append(tx)
            
        return transactions
    
    def generate_batch_transactions(self, count: int) -> List[Dict]:
        """生成批量交易 - 模拟批处理场景"""
        transactions = []
        batch_size = count // 20  # 分20批
        
        for batch_id in range(20):
            # 选择一个分片进行批量操作
            shard_id = random.randint(0, 3)
            batch_sender = random.choice(self.shard_addresses[shard_id])
            
            for _ in range(batch_size):
                to_addr = random.choice(self.shard_addresses[shard_id])
                
                tx = {
                    'type': 'batch',
                    'from': batch_sender,
                    'to': to_addr,
                    'amount': self._generate_amount('batch'),
                    'gas_limit': random.randint(21000, 60000),
                    'gas_price': random.randint(25000000000, 80000000000),  # 25-80 Gwei
                    'batch_id': batch_id,
                }
                transactions.append(tx)
                
        return transactions
    
    def generate_stress_transactions(self, count: int) -> List[Dict]:
        """生成压力测试交易 - 高价值交易测试极限性能"""
        transactions = []
        
        for _ in range(count):
            # 压力测试倾向于发送到性能较好的分片
            shard_weights = [0.1, 0.2, 0.3, 0.4]  # 分片0-3的权重
            target_shard = random.choices(range(4), weights=shard_weights)[0]
            
            from_addr = random.choice(self.addresses)
            to_addr = random.choice(self.shard_addresses[target_shard])
            
            tx = {
                'type': 'stress',
                'from': from_addr,
                'to': to_addr,
                'amount': self._generate_amount('stress'),
                'gas_limit': random.randint(100000, 500000),  # 高gas消耗
                'gas_price': random.randint(100000000000, 500000000000),  # 100-500 Gwei
                'target_shard': target_shard,
            }
            transactions.append(tx)
            
        return transactions
    
    def generate_dataset(self, total_transactions: int = 100000) -> List[Dict]:
        """生成完整数据集"""
        print(f"开始生成 {total_transactions} 条交易的数据集...")
        
        # 按比例分配交易类型
        basic_count = int(total_transactions * 0.40)      # 40% 基础交易
        hot_count = int(total_transactions * 0.25)        # 25% 热点交易
        cross_count = int(total_transactions * 0.20)      # 20% 跨分片交易
        batch_count = int(total_transactions * 0.10)      # 10% 批量交易
        stress_count = int(total_transactions * 0.05)     # 5% 压力测试交易
        
        all_transactions = []
        
        print(f"生成 {basic_count} 条基础交易...")
        all_transactions.extend(self.generate_basic_transactions(basic_count))
        
        print(f"生成 {hot_count} 条热点交易...")
        all_transactions.extend(self.generate_hot_transactions(hot_count))
        
        print(f"生成 {cross_count} 条跨分片交易...")
        all_transactions.extend(self.generate_cross_shard_transactions(cross_count))
        
        print(f"生成 {batch_count} 条批量交易...")
        all_transactions.extend(self.generate_batch_transactions(batch_count))
        
        print(f"生成 {stress_count} 条压力测试交易...")
        all_transactions.extend(self.generate_stress_transactions(stress_count))
        
        # 打乱交易顺序
        print("打乱交易顺序...")
        random.shuffle(all_transactions)
        
        print(f"数据集生成完成！总计 {len(all_transactions)} 条交易")
        return all_transactions
    
    def save_to_csv(self, transactions: List[Dict], filename: str = "selectedTxs_100K.csv"):
        """保存交易到CSV文件（兼容原始格式）"""
        print(f"保存交易数据到 {filename}...")
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            # 使用原始CSV格式：空列，空列，空列，from，to，空列，空列，空列，amount，空列...
            writer = csv.writer(csvfile)
            
            for tx in transactions:
                row = [
                    '',                    # 空列1
                    '',                    # 空列2  
                    '',                    # 空列3
                    tx['from'],           # from地址
                    tx['to'],             # to地址
                    '',                    # 空列6
                    '0',                   # 空列7（通常是0）
                    '0',                   # 空列8（通常是0）
                    str(tx['amount']),     # 金额
                    '',                    # 空列10
                    '',                    # 空列11
                    '',                    # 空列12
                    '',                    # 空列13
                    '',                    # 空列14
                    '',                    # 空列15
                    '',                    # 空列16
                    '',                    # 空列17
                    '',                    # 空列18
                    '',                    # 空列19
                ]
                writer.writerow(row)
        
        print(f"数据保存完成！文件：{filename}")
    
    def generate_analysis_report(self, transactions: List[Dict]):
        """生成数据集分析报告"""
        print("\n" + "="*60)
        print("数据集分析报告")
        print("="*60)
        
        # 按类型统计
        type_count = {}
        shard_distribution = {i: 0 for i in range(4)}
        cross_shard_count = 0
        total_amount = 0
        
        for tx in transactions:
            tx_type = tx['type']
            type_count[tx_type] = type_count.get(tx_type, 0) + 1
            
            # 统计目标分片分布
            to_shard = self._get_address_shard(tx['to'])
            shard_distribution[to_shard] += 1
            
            # 统计跨分片交易
            from_shard = self._get_address_shard(tx['from'])
            if from_shard != to_shard:
                cross_shard_count += 1
                
            total_amount += tx['amount']
        
        print(f"交易类型分布：")
        for tx_type, count in type_count.items():
            percentage = (count / len(transactions)) * 100
            print(f"  {tx_type:10s}: {count:6d} 条 ({percentage:5.1f}%)")
        
        print(f"\n分片负载分布：")
        for shard_id, count in shard_distribution.items():
            percentage = (count / len(transactions)) * 100
            performance = ["性能洼地", "中低性能", "中高性能", "高性能"][shard_id]
            print(f"  分片{shard_id} ({performance}): {count:6d} 条 ({percentage:5.1f}%)")
        
        print(f"\n跨分片交易统计：")
        print(f"  跨分片交易: {cross_shard_count} 条 ({(cross_shard_count/len(transactions)*100):5.1f}%)")
        print(f"  分片内交易: {len(transactions)-cross_shard_count} 条 ({((len(transactions)-cross_shard_count)/len(transactions)*100):5.1f}%)")
        
        print(f"\n金额统计：")
        print(f"  总金额: {total_amount/1e18:.2f} ETH")
        print(f"  平均金额: {total_amount/len(transactions)/1e18:.6f} ETH")
        
        print("\n数据集特点：")
        print("1. 基础交易(40%): 模拟日常业务，80%分片内+20%跨分片")
        print("2. 热点交易(25%): 集中在100个热点地址，测试热点处理能力")  
        print("3. 跨分片交易(20%): 专门测试跨分片通信开销")
        print("4. 批量交易(10%): 20个批次，测试批处理性能")
        print("5. 压力测试(5%): 高价值交易，倾向发送到高性能分片")
        print("\n实验目标：")
        print("- 验证负载均衡算法在异构环境中的表现")
        print("- 测试性能洼地分片对系统整体吞吐量的影响")
        print("- 观察不同交易模式下的分片性能差异")
        print("="*60)

def main():
    """主函数"""
    print("区块链分片负载均衡数据集生成器")
    print("目标：生成10万+条交易用于性能异构分片测试")
    print("-" * 50)
    
    # 设置随机种子以确保可重现性
    random.seed(42)
    
    # 创建生成器
    generator = TransactionGenerator()
    
    # 生成数据集
    transactions = generator.generate_dataset(total_transactions=100000)
    
    # 保存到CSV文件
    generator.save_to_csv(transactions, "selectedTxs_100K.csv")
    
    # 生成分析报告
    generator.generate_analysis_report(transactions)
    
    print(f"\n数据集生成完成！")
    print(f"文件：selectedTxs_100K.csv")
    print(f"请将该文件复制到 docker/Files/ 目录下")

if __name__ == "__main__":
    main()