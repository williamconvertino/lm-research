import time

count = 0
while True:
    count += 1
    time.sleep(1)
    print(f"\rIdle for {count} seconds", end="")