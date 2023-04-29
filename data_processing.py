import json
from model import XLMRBinaryClassifier
import numpy as np

def get_data(json_file):
    with open(json_file, 'r', encoding="utf-8") as input_file:
        data = []
        lexicon, _ = get_frmt_lexicon()
        for k, line in enumerate(input_file):
            json_line = json.loads(line)
            sentence = json_line["sentence"]
            for term in lexicon:
                translations = lexicon[term]
                if term in sentence or translations in sentence:
                    label = json_line["label"]
                    data.append((sentence, label))

    # only keep first 50 sentences
    #data = data

    return data

def get_frmt_lexicon():
    # Format: English: (Simp-CN, Simp-TW, Trad-TW, Trad-CN)
    orginal_zh_terms = {
        "Pineapple": ("菠萝", "凤梨", "鳳梨", "菠蘿"),
        "Computer mouse": ("鼠标", "滑鼠", "滑鼠", "鼠標"),
        # Original source had CN:牛油果, but translator used 鳄梨.
        # "Avocado": ("鳄梨", "酪梨", "酪梨", "鱷梨"),
        "Band-Aid": ("创可贴", "OK绷", "OK繃", "創可貼"),
        "Blog": ("博客", "部落格", "部落格", "博客"),
        "New Zealand": ("新西兰", "纽西兰", "紐西蘭", "新西蘭"),
        "Printer (computing)": ("打印机", "印表机", "印表機", "打印機"),
        # Original source has TW:月臺, but translator used 月台.
        "Railway platform": ("站台", "月台", "月台", "站台"),
        "Roller coaster": ("过山车", "云霄飞车", "雲霄飛車", "過山車"),
        "Salmon": ("三文鱼", "鲑鱼", "鮭魚", "三文魚"),
        "Shampoo": ("洗发水", "洗发精", "洗髮精", "洗髮水"),
        # From Wikipedia page "Software testing"
        "Software": ("软件", "软体", "軟體", "軟件"),
        "Sydney": ("悉尼", "雪梨", "雪梨", "悉尼"),
        "test1": ("测试1", "测试2", "测试11", "测试4"),
        "test2": ("测试2", "测试6", "测试22", "测试8"),

        # The following two are excluded because they underpin the first 100
        # lexical exemplars used for priming the models.
        # "Flip-flops": ("人字拖", "夹脚拖", "夾腳拖", "人字拖"),
        "Paper clip": ("回形针", "回纹针", "迴紋針", "回形針")
    }

    # Portuguese terms.
    # Format: English: (BR, PT)
    # The Portuguese corpus is lowercased before matching these terms.
    orginal_pt_terms = {
        # "Bathroom": ("banheiro", "casa de banho"),
        # Original source had "pequeno almoço" but translator used "pequeno-almoço".
        # "Breakfast": ("café da manhã", "pequeno-almoço"),
        "Bus": ("ônibus", "autocarro"),
        "Cup": ("xícara", "chávena"),
        "Computer mouse": ("mouse", "rato"),
        # "Drivers license": ("carteira de motorista", "carta de condução"),
        # From Wikipedia page "Ice cream sandwich"
        "Ice cream": ("sorvete", "gelado"),
        "Juice": ("suco", "sumo"),
        "Mobile phone": ("celular", "telemóvel"),
        "Pedestrian": ("pedestre", "peão"),
        # From Wikipedia page "Pickpocketing"
        # "Pickpocket": ("batedor de carteiras", "carteirista"),
        "Pineapple": ("abacaxi", "ananás"),
        "Refrigerator": ("geladeira", "frigorífico"),
        "Suit": ("terno", "fato"),
        "Train": ("trem", "comboio"),
        "Video game": ("videogame", "videojogos"),

        # Terms updated after original selection.

        # For BR, replaced "menina" (common in speech) with "garota" (common in
        # writing, matching the human translators.
        "Girl": ("garota", "rapariga"),

        # Replace original "Computer monitor": ("tela de computador", "ecrã") with
        # the observed use for just screen:
        "Screen": ("tela", "ecrã"),

        # Terms excluded.

        # The following three are excluded because they underpin the first 100
        # lexical exemplars used for priming the models.
        "Gym": ("academia", "ginásio"),
        "Stapler": ("grampeador", "agrafador"),
        # "Nightgown": ("camisola", "camisa de noite"),

        # The following are excluded for other reasons:

        # BR translator primarily used 'comissário de bordo' and hardly ever
        # 'aeromoça'. PT translator used 'comissários/assistentes de bordo' or just
        # 'assistentes de bordo' Excluding the term as low-signal for now.
        ## "Flight attendant": ("aeromoça", "comissário ao bordo"),

        # Both regions' translators consistently used "presunto", so the term has
        # low signal.
        ## "Ham": ("presunto", "fiambre"),
    }

    zh_lexicon = {}
    pt_lexicon = {}

    # for each language, create a dictionary of terms
    for orginal_terms in [orginal_zh_terms, orginal_pt_terms]:
        for term, translations in orginal_terms.items():
            # if it's chinese
            if len(translations) == 4:
                # only care about Simp-CN and Trad-TW
                zh_lexicon[translations[0]] = translations[2]
            else:
                # only care about BR and PT
                pt_lexicon[translations[0]] = translations[1]

    return zh_lexicon, pt_lexicon


def main():
    # Create XLM-Roberta binary classifier

    data = get_data("data/all/zh/test_with_parse.json")

    train_texts = [d[0] for d in data]
    train_labels = [int(d[1]) for d in data]

    # Create XLM-Roberta binary classifier
    clf = XLMRBinaryClassifier()

    # Train classifier on example data
    clf.train(train_texts, train_labels)
    clf.save_model("model")


    clf.load_model("model")

    # Evaluate contrastive_experiment
    #contrastive_experiment(train_texts, train_labels, clf)

    # Evaluate leave_one_out_experiment
    leave_one_out_experiment_per_sent(train_texts[0], clf)



def contrastive_experiment(data, label, model):

    predictions = []

    for sentence in data:

        original_text = sentence
        contrastive_text = ""
        lexicon = get_frmt_lexicon()[0]
        reversed_lexicon = {v: k for k, v in lexicon.items()}

        for word in original_text.split():

            if word in lexicon:
                translation = lexicon[word]
                # swap the word with its translation
                contrastive_text = (original_text.replace(word, translation))
            elif word in reversed_lexicon:
                translation = reversed_lexicon[word]
                # swap the word with its translation
                contrastive_text = (original_text.replace(word, translation))

        # compare the probability difference of the model prediction between original text with the contrastive text

        for original_text, contrastive_text in zip([original_text], [contrastive_text]):


            original_pred_label, original_proba = model.predict(original_text)
            contrastive_pred_label, contrastive_proba = model.predict(contrastive_text)

            predictions.append(int(original_pred_label))

            if original_text == contrastive_text or original_text == "" or contrastive_text == "":
                print("skipped")
                print(original_text)
                print(contrastive_text)
                print("\n")
                continue

            if abs(original_proba - contrastive_proba) > 0.2:
                print("\n")

                print("Original text: ", original_text)
                print("Original prediction: ", original_pred_label)
                print("Original probability: ", original_proba)
                print("\n")
                print("Contrastive text: ", contrastive_text)
                print("Contrastive prediction: ", contrastive_pred_label)
                print("Contrastive probability: ", contrastive_proba)
                print("\n")

                print("---------------------------------")
    # compare the predictions with the train_labels
    print("Accuracy: ", accuracy(predictions, label))
    print("F1: ", f1(predictions, label))
    print("Precision: ", precision(predictions, label))
    print("Recall: ", recall(predictions, label))

def leave_one_out_experiment_per_sent(original_sent, model):

    original_pred_label, original_proba = model.predict(original_sent)
    print(f"Original sentence: {original_sent}")
    print(f"Original sentence prediction: {original_pred_label}")
    print(f"Original sentence probability: {original_proba}")
    print("\n")

    # Split sentence into a list of words
    words = original_sent.split()

    # Perform LOO on each word in the sentence
    for i in range(len(words)):
        # Remove one word from the sentence
        left_out_word = words.pop(i)
        left_out_sentence = " ".join(words)

        # Evaluate the impact on the sentence
        # (In this example, we're just printing the left-out sentence for simplicity)
        llo_pred_label, llo_proba = model.predict(left_out_sentence)
        print(f"Left out word: {left_out_word}")
        print(f"Left out sentence: {left_out_sentence}")
        print(f"Left out sentence prediction: {llo_pred_label}")
        print(f"Left out sentence probability: {llo_proba}")
        print("\n")

        # Add the left-out word back into the sentence for the next iteration
        words.insert(i, left_out_word)


def accuracy(predictions, labels):
    """
    Calculates the accuracy of the predictions given the true labels.

    Parameters:
    predictions (list): A list of predicted labels.
    labels (list): A list of true labels.

    Returns:
    float: The accuracy of the predictions.
    """
    correct = 0
    total = len(predictions)
    for i in range(total):
        if predictions[i] == labels[i]:
            correct += 1
    return float(correct) / total


def f1(predictions, labels):
    """
    Calculates the F1 score of the predictions given the true labels.

    Parameters:
    predictions (list): A list of predicted labels.
    labels (list): A list of true labels.

    Returns:
    float: The F1 score of the predictions.
    """
    tp = 0
    fp = 0
    fn = 0
    for i in range(len(predictions)):
        if predictions[i] == labels[i]:
            if predictions[i] == 1:
                tp += 1
        else:
            if predictions[i] == 1:
                fp += 1
            else:
                fn += 1
    if tp == 0:
        return 0
    precision = float(tp) / (tp + fp)
    recall = float(tp) / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def precision(predictions, labels):
    """
    Calculates the precision of the predictions given the true labels.

    Parameters:
    predictions (list): A list of predicted labels.
    labels (list): A list of true labels.

    Returns:
    float: The precision of the predictions.
    """
    tp = 0
    fp = 0
    for i in range(len(predictions)):
        if predictions[i] == labels[i]:
            if predictions[i] == 1:
                tp += 1
        else:
            if predictions[i] == 1:
                fp += 1
    if tp == 0:
        return 0
    return float(tp) / (tp + fp)


def recall(predictions, labels):
    """
    Calculates the recall of the predictions given the true labels.

    Parameters:
    predictions (list): A list of predicted labels.
    labels (list): A list of true labels.

    Returns:
    float: The recall of the predictions.
    """
    tp = 0
    fn = 0
    for i in range(len(predictions)):
        if predictions[i] == labels[i]:
            if predictions[i] == 1:
                tp += 1
        else:
            if predictions[i] == 0:
                fn += 1
    if tp == 0:
        return 0
    return float(tp) / (tp + fn)


if __name__ == '__main__':
    main()
