#!/usr/bin/env python3
import subprocess
import sys
from typing import Generator

def run(args: list[str], inputgen: Generator[str]) -> Generator[str]:
    print("args:", args)

    with subprocess.Popen(["./run.sh", *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as p:
        for input in inputgen:
            p.stdin.write((input + "\n").encode("utf-8"))
            p.stdin.flush()

            output = p.stdout.readline().decode("utf-8").rstrip()
            yield output

        p.stdin.close()

def inputgen() -> Generator[str]:
    for i in range(10):
        yield str(2**i)

if __name__ == "__main__":
    outputgen = run(sys.argv[1:], inputgen())

    for output in outputgen:
        print(output)
