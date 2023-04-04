
with open("aaa.tsv", 'r', encoding="utf-8") as input_file:
    count_label_0 = 0
    count_label_1 = 0
    class_0_word_count = 0
    class_1_word_count = 0
    for k, line in enumerate(input_file):
        if k == 0:
            continue
        try:
            sentence, label = line.strip().split("\t")
        except:
            print(line)
            continue

        if label == "0":
            count_label_0 += 1
            class_0_word_count += len(sentence.split())
        else:
            count_label_1 += 1
            class_1_word_count += len(sentence.split())
    print("count_label_0: ", count_label_0)
    print("count_label_1: ", count_label_1)
    print("class_0_word_count: ", class_0_word_count)
    print("class_1_word_count: ", class_1_word_count)