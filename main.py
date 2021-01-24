import sys
import id3
import gc

if __name__ == "__main__":
    # python main.py iterations percentage if_print_tree
    args = sys.argv
    if len(args) == 1:
        iterations = 100
        percentage_of_data = 2
        if_print_tree = False
    else:
        iterations = int(args[1])
        percentage_of_data = int(args[2])
        if_print_tree = False
        if(int(args[3]) == 1):
            if_print_tree = True

    list1 = []
    for i in range(0,iterations):
        dtree = id3.ID3(percentage_of_data)
        dtree.prepare_data("mushroom.txt")
        dtree.root = dtree.modified_ID3(dtree.learningDataSet, dtree.attributes)
        dtree.print_tree(dtree.root)
        print("\n\n")
        percent = dtree.test_tree()
        list1.append(percent)
        gc.collect()
    list2 = []
    for j in range(0,iterations):
        dtree = id3.ID3(percentage_of_data)
        dtree.prepare_data("mushroom.txt")
        dtree.root = dtree.classic_ID3(dtree.learningDataSet, dtree.attributes)
        dtree.print_tree(dtree.root)
        print("\n\n")
        percent = dtree.test_tree()
        list2.append(percent)
        gc.collect()

    print(f'Decision tree was created {iterations} times for each algorithm with {percentage_of_data}% of data set used to train it')
    print(f'Calculated average complience for modified ID3 is: {sum(list1)/len(list1)}')
    print(f'Calculated average complience for classic ID3 is: {sum(list2)/len(list2)}')
