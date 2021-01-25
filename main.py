# Jarosław Zabuski, Jakub Strawa

import sys
import id3
import gc
import time

if __name__ == "__main__":
    # python main.py iterations percentage if_print_tree
    args = sys.argv
    if len(args) == 1:
        iterations = 100
        percentage_of_data = 2
        if_print_tree = False
    else:
        if len(args) != 4:
            print("Not enough arguments!")
            exit(1)
        iterations = int(args[1])
        percentage_of_data = int(args[2])
        if_print_tree = False
        if(int(args[3]) == 1):
            if_print_tree = True

    list1 = []
    times1 = [0, 0]
    tree1depth = 0
    tree1_attr_used = 0
    for i in range(0, iterations):
        timer1 = time.time()
        dtree = id3.ID3(percentage_of_data)
        dtree.prepare_data("mushroom.txt")
        dtree.root = dtree.modified_ID3(dtree.learningDataSet, dtree.attributes)
        timer2 = time.time()
        times1[0] += timer2 - timer1
        if if_print_tree is True:
            dtree.print_tree(dtree.root)
            print("\n\n")
        timer1 = time.time()
        percent = dtree.test_tree()
        timer2 = time.time()
        times1[1] += timer2 - timer1
        list1.append(percent)
        tree1depth += dtree.max_tree_depth/2
        tree1_attr_used += (22 - len(dtree.attributes))
    list2 = []
    times2 = [0, 0]
    tree2depth = 0
    tree2_attr_used = 0
    for j in range(0, iterations):
        timer1 = time.time()
        dtree = id3.ID3(percentage_of_data)
        dtree.prepare_data("mushroom.txt")
        dtree.root = dtree.classic_ID3(dtree.learningDataSet, dtree.attributes)
        timer2 = time.time()
        times2[0] += timer2 - timer1
        if if_print_tree is True:
            dtree.print_tree(dtree.root)
            print("\n\n")
        timer1 = time.time()
        percent = dtree.test_tree()
        timer2 = time.time()
        times2[1] += timer2 - timer1
        list2.append(percent)
        tree2depth += dtree.max_tree_depth/2
        tree2_attr_used += (22 - len(dtree.attributes))

    print(f'Decision tree was created {iterations} times for each algorithm with {percentage_of_data}% of data set used to train it')
    print(f'Calculated average complience for modified ID3 is: {sum(list1)/len(list1)}')
    print(f'Calculated average complience for classic ID3 is: {sum(list2)/len(list2)}')
    results = open("result1.txt", "a")
    results.write("\n" + str(percentage_of_data) + " " + str(sum(list1)/len(list1)) + " " + str(sum(list2)/len(list2))
                  + " " + str(times1[0]/iterations) + " " + str(times1[1]/iterations) + " " + str(times2[0]/iterations)
                  + " " + str(times2[1]/iterations) + " " + str(tree1depth/iterations) + " " + str(tree2depth/iterations)
                  + " " + str(tree1_attr_used/iterations) + " " + str(tree2_attr_used/iterations))
    results.close()

    gc.collect()

