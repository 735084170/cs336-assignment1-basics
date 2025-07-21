import time

def slow_function():
    """一个比较耗时的函数"""
    print("进入 slow_function...")
    total = 0
    for i in range(10**7):
        total += i
    print("离开 slow_function。")
    return total

def fast_function():
    """一个执行很快的函数"""
    print("执行 fast_function。")
    time.sleep(0.1) # 模拟少量I/O操作

def main():
    """主函数，调用其他函数"""
    print("程序开始。")
    slow_function()
    for _ in range(5):
        fast_function()
    print("程序结束。")

if __name__ == "__main__":
    main()