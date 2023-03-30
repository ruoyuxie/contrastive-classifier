
def get_frmt_lexicon():

    # Format: English: (Simp-CN, Simp-TW, Trad-TW, Trad-CN)
    orginal_zh_terms = {
        "Pineapple": ("菠萝", "凤梨", "鳳梨", "菠蘿"),
        "Computer mouse": ("鼠标", "滑鼠", "滑鼠", "鼠標"),
        # Original source had CN:牛油果, but translator used 鳄梨.
        "Avocado": ("鳄梨", "酪梨", "酪梨", "鱷梨"),
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

        # The following two are excluded because they underpin the first 100
        # lexical exemplars used for priming the models.
        "Flip-flops": ("人字拖", "夹脚拖", "夾腳拖", "人字拖"),
        "Paper clip": ("回形针", "回纹针", "迴紋針", "回形針")
    }

    # Portuguese terms.
    # Format: English: (BR, PT)
    # The Portuguese corpus is lowercased before matching these terms.
    orginal_pt_terms = {
        "Bathroom": ("banheiro", "casa de banho"),
        # Original source had "pequeno almoço" but translator used "pequeno-almoço".
        "Breakfast": ("café da manhã", "pequeno-almoço"),
        "Bus": ("ônibus", "autocarro"),
        "Cup": ("xícara", "chávena"),
        "Computer mouse": ("mouse", "rato"),
        "Drivers license": ("carteira de motorista", "carta de condução"),
        # From Wikipedia page "Ice cream sandwich"
        "Ice cream": ("sorvete", "gelado"),
        "Juice": ("suco", "sumo"),
        "Mobile phone": ("celular", "telemóvel"),
        "Pedestrian": ("pedestre", "peão"),
        # From Wikipedia page "Pickpocketing"
        "Pickpocket": ("batedor de carteiras", "carteirista"),
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
        "Nightgown": ("camisola", "camisa de noite"),

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

def evaluate_frmt_lexicon_pt(exp_word_list):
    _, pt_lexicon = get_frmt_lexicon()

    # get a list of words, check every word if it is mik or dik or mi or di
    for word in exp_word_list:
        if word in pt_lexicon:
            print(word, pt_lexicon[word])

def evaluate_frmt_lexicon_zh(exp_word_list):
    zh_lexicon, _ = get_frmt_lexicon()



# mik_count = 0
# dik_count = 0
#
# mi_count = 0
# di_count = 0
#
# # get a list of words, check every word if it is mik or dik or mi or di
# for word in word_list:
#     if word == "mik":
#         mik_count += 1
#     elif word == "dik":
#         dik_count += 1
#     elif word == "mi":
#         mi_count += 1
#     elif word == "di":
#         di_count += 1
#
# print("total number of words: ", len(word_list))
# print("number of words [mik]: ", mik_count)
# print("number of words [dik]: ", dik_count)
# print("number of words [mi]: ", mi_count)
# print("number of words [di]: ", di_count)
