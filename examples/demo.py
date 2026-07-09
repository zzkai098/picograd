"""Minimal demo: build a graph and (once backward is implemented) read gradients."""

from zgrad import Value


def main():
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)
    d = a * b + c
    print(f"forward: d = a*b + c = {d.data}")  # -> -6.0 + 10.0 = 4.0

    # Uncomment once Value.backward() is implemented:
    # d.backward()
    # print(f"a.grad = {a.grad}")  # -3.0
    # print(f"b.grad = {b.grad}")  #  2.0
    # print(f"c.grad = {c.grad}")  #  1.0


if __name__ == "__main__":
    main()
