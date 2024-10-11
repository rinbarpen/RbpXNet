import concurrent.futures
import threading
import time
import psutil
import subprocess
import networkx as nx

PROJECT_NAME = 'Segment-Drive'
direct_graph = nx.DiGraph()

# 添加任务节点
direct_graph.add_node('1', config_file=f'configs/{PROJECT_NAME}/UNet_DRIVE.json')
direct_graph.add_node('2', config_file=f'configs/{PROJECT_NAME}/ML-UNet_DRIVE.json')
direct_graph.add_node('3', config_file=f'configs/{PROJECT_NAME}/RWA-UNet_DRIVE.json')

direct_graph.add_edges_from([('1', '2'), ('2', '3')])  # 假设任务之间有依赖关系

# 全局锁
graph_lock = threading.Lock()

# 定义任务节点启动函数
def run_task(node):
    config_file = direct_graph.nodes[node]['config_file']
    try:
        print(f"Running task {node} with config: {config_file}")
        result = subprocess.run(['python', 'train.py', '--config', config_file], check=True)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Task {node} failed with error: {e}")
        raise e  # 让上层捕获异常，处理资源不足和异常情况

# 检查显存、内存等资源使用情况
def check_resources():
    mem = psutil.virtual_memory()
    # 假设显存使用情况通过nvidia-smi获取, 或其他工具
    gpu_mem = 0  # 根据你的环境更新显存检测逻辑
    print(f"Memory used: {mem.percent}%, GPU memory used: {gpu_mem}%")
    
    if mem.percent > 85 or gpu_mem > 90:  # 如果内存或显存超过一定阈值
        return False  # 资源不足
    return True  # 资源充足

# 等待资源恢复
def wait_for_resources(sleep_time=10):
    print(f"Resources are insufficient. Waiting for resources to free up...")
    while not check_resources():  # 如果资源不足，等待并继续检测
        time.sleep(sleep_time)  # 等待指定的秒数后再检查资源情况
    print("Resources are now sufficient. Resuming tasks...")

# 从有向图中获取所有入度为0的节点
def get_zero_in_degree_nodes(graph):
    with graph_lock:  # 确保并发安全
        zero_in_degree_nodes = [n for n, d in graph.in_degree() if d == 0]
    return zero_in_degree_nodes

# 在任务完成后移除结点并更新图
def remove_node_from_graph(graph, node):
    with graph_lock:  # 确保并发安全
        if node in graph:
            graph.remove_node(node)
            print(f"Node {node} has been removed from the graph.")

# 任务调度管理函数
def execute_pipeline():
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {}
        
        while True:
            zero_in_degree_nodes = get_zero_in_degree_nodes(direct_graph)
            
            # 如果没有入度为0的结点且任务池为空，说明任务已完成
            if not zero_in_degree_nodes and not futures:
                print("All tasks have been completed.")
                break
            
            # 如果没有入度为0的结点，等待一个任务完成再继续
            if not zero_in_degree_nodes:
                print("No zero in-degree nodes available, waiting for a task to complete...")
                done, _ = concurrent.futures.wait(futures.values(), return_when=concurrent.futures.FIRST_COMPLETED)
                
                # 移除已完成的任务，并更新图
                for future in done:
                    node = [k for k, v in futures.items() if v == future][0]
                    try:
                        future.result()  # 获取任务结果，检查是否有异常
                        remove_node_from_graph(direct_graph, node)  # 任务成功完成，移除结点
                    except Exception as e:
                        print(f"Task {node} failed. Retrying after checking resources...")
                        wait_for_resources()  # 任务失败后等待资源恢复
                        futures[node] = executor.submit(run_task, node)  # 重试任务

                # 移除已完成的任务
                for future in done:
                    del futures[[k for k, v in futures.items() if v == future][0]]

                continue  # 返回循环，重新检查入度为0的结点
            
            # 检查资源情况，并提交任务
            for node in zero_in_degree_nodes:
                if check_resources():  # 如果资源充足，提交任务
                    futures[node] = executor.submit(run_task, node)
                else:
                    wait_for_resources()  # 资源不足时等待

# 启动pipeline
if __name__ == "__main__":
    execute_pipeline()
