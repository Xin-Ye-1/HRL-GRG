#!/usr/bin/env python


from worker import *
from train import *

def test():
    graph, workers = set_up()
    with tf.Session(graph=graph) as sess:
        workers[0].evaluate(sess)


if __name__ == '__main__':
    test()
