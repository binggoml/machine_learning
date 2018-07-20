import pickle  # 导入持久化类
import os
import jieba
import re
from xml.dom import minidom


def get_stopwords(path):
    return set([line.strip() for line in open(path, 'r', encoding='utf-8')])


def get_files(path):
    # 获取该路径下所有文件的列表
    filelist = []
    for (root, dirs, files) in os.walk(path):
        temp = [os.path.join(root, files) for files in files]
        filelist.extend(temp)
    # print(filelist)
    return filelist


# 对列表进行分词并用空格连接
def segmentWord(txt):
    text = ""
    word_list = list(jieba.cut(txt))
    for word in word_list:
        word = word.strip()
        if word not in stopwords_list and word != '':
            text += word
            text += ' '
    return text


# 获取xml 对应标签的组合文本
def get_xml_paragraph_txt(file):
    reg = '&amp;|&rdquo;|&ldquo;|&times;|rdquo;|ldquo;|times;|&bull;'
    pattern = re.compile(reg)
    lst_paragraph = ['原告诉称', '被告辩称', '被告诉称', '本院查明', '本院认为', '当事人信息']

    try:
        doc = minidom.parse(file)
        root = doc.documentElement
    except Exception as err:
        print(file)
        return ''

    lines = []
    for paragraph in lst_paragraph:
        nodes = root.getElementsByTagName(paragraph)
        if nodes:
            txt = nodes[0].childNodes[0].nodeValue
            txt = str(txt)
            if not txt.endswith("。"):
                txt += "。"
            if len(txt) > 5:
                txt = re.sub(pattern, '', txt)
                lines.append(txt)
    tmp_txt = ''.join(lines)
    # 统一车牌
    txt = mark_plate_number(tmp_txt)
    return txt


# 标记车牌，统一车牌为CHEPAI1~CHEPAI20
def mark_plate_number(line):
    result = []
    # 车牌号正则
    reg1 = '[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领A-Z]{1}[A-Z]{1}[A-Z0-9]{4}[A-Z0-9挂学警港澳]{1}'
    reg2 = '[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领]{1}.{1,3}[×xX＊*☆★○]{3,5}'
    pattern1 = re.compile(reg1)
    pattern2 = re.compile(reg2)

    # 统计车牌
    line = line.strip()
    line_tmp = line
    dic = dict()
    # print(line_tmp)
    tmp_list = re.findall(pattern1, line_tmp)
    for v in tmp_list:
        if v not in dic.keys():
            dic[v] = len(v)

    tmp_list = re.findall(pattern2, line_tmp)
    for v in tmp_list:
        if v not in dic.keys():
            dic[v] = len(v)

    # 车牌长度排序，降序
    sorted_key_list = sorted(dic.items(), key=lambda e: e[1], reverse=True)
    i = 1
    for tp in sorted_key_list:
        string = 'CHEPAI' + str(i)
        line = line.replace(str(tp[0]), string)
        i += 1

    return line


if __name__ == "__main__":
    # 添加自定义词典
    # print('load userdict...')
    # jieba.load_userdict("D:\\test\\000_tonganzhitui\\多辆机动车致人损害\\法律词典.dic")
    # print('load userdict finish!')

    # 停用词
    stopwords_path = 'stop_words_ch.txt'
    # 模型路径
    model_path = 'data/models/xgboost.pkl'
    # vectorizer_pkl对象保存路径
    vectorizer_path = 'data/models/vectorizer.pkl'

    # 获取停用词
    stopwords_list = get_stopwords(stopwords_path)
    # 加载TfidfVectorizer对象
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    # 加载模型
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # 预测结果输出文件
    output = 'data/result/result.txt'
    # 测试文本目录
    input = 'data/testText/'
    print('Loading Data...')
    filelist = get_files(input)

    count = 1
    lst_pred = []
    for file in filelist:
        print(count)
        # if count < 6:
        #     count += 1
        #     continue
        txt = get_xml_paragraph_txt(file)
        if txt == '':
            continue
        test_content = segmentWord(txt)
        cont = []
        cont.append(test_content)
        test_tfidf = vectorizer.transform(cont)
        # print(test_weight.shape)
        y_pred = model.predict(test_tfidf)
        lst_pred.append(str(file) + '\t' + str(y_pred[0]))
        count += 1

    with open(output, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lst_pred))
    print('完成数据预测！')
