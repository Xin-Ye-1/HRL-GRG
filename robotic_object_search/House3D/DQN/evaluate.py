#!/usr/bin/env python


from worker import *
from train import *

def test():
    graph, workers = set_up()
    with graph.as_default():
        workers[0].evaluate(read_file=FLAGS.evaluate_file)


if __name__ == '__main__':
    test()
