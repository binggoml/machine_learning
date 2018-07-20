import sys
import os
import jieba
# 引入Bunch类
from sklearn.datasets.base import Bunch
# 引入持久化类
import pickle


##############################################################
# 分类语料预处理的类
# 语料目录结构：
# corpus
#   |-catergory_A
#     |-01.txt
#     |-02.txt
#   |-catergory_B
#   |-catergory_C
#   ...
##############################################################



# 文本预处理类
class TextPreprocess:
    # 定义词袋对象:data_set
    # Bunch类提供一种key,value的对象形式
    # target_name:所有分类集名称列表
    # label:每个文件的分类标签列表
    # filenames:文件名称
    # contents:文件内容
    data_set = Bunch(target_name=[], label=[], filenames=[], contents=[])

    def __init__(self):  # 构造方法
        print('调用构造函数')

        self.user_dic_path = ''  # 自定义词典
        self.corpus_path = ""  # 原始语料路径
        self.segment_path = ""  # 分词后语料路径
        self.wordbag_path = ""  # 词袋模型路径
        self.trainset_name = ""  # 训练集文件名


    # 对预处理后语料进行分词,并持久化。
    # 处理后在segment_path下建立与pos_path相同的子目录和文件结构
    def segment(self):
        if (self.segment_path == "" or self.corpus_path == ""):
            print('segment_path或pos_path不能为空')
            return
        dir_list = os.listdir(self.corpus_path)
        # 获取每个目录下所有的文件
        for mydir in dir_list:
            class_path = self.corpus_path + mydir + "/"  # 拼出分类子目录的路径
            file_list = os.listdir(class_path)  # 获取class_path下的所有文件
            for file_path in file_list:  # 遍历所有文件
                file_name = class_path + file_path  # 拼出文件名全路径
                file_read = open(file_name, 'r', encoding='utf-8')  # 打开一个文件
                raw_corpus = file_read.read()  # 读取未分词语料
                seg_corpus = jieba.cut(raw_corpus)  # 结巴分词操作
                # 拼出分词后语料分类目录
                seg_dir = self.segment_path + mydir + "/"
                if not os.path.exists(seg_dir):  # 如果没有创建
                    os.makedirs(seg_dir)
                file_write = open(seg_dir + file_path, 'w', encoding='utf-8')  # 创建分词后语料文件，文件名与未分词语料相同
                file_write.write(" ".join(seg_corpus))  # 用空格将分词结果分开并写入到分词后语料文件中
                file_read.close()  # 关闭打开的文件
                file_write.close()  # 关闭写入的文件
        print('中文语料分词成功完成!')

    # 分词训练语料进行打包
    def create_dat_file(self):
        if (self.segment_path == "" or self.wordbag_path == "" or self.trainset_name == ""):
            print('segment_path或wordbag_path,trainset_name不能为空!')
            return

        # 获取corpus_path下的所有子分类
        dir_list = os.listdir(self.segment_path)
        self.data_set.target_name = dir_list
        # 获取每个目录下所有的文件
        for mydir in dir_list:
            class_path = self.segment_path + mydir + "/"  # 拼出分类子目录的路径
            file_list = os.listdir(class_path)  # 获取class_path下的所有文件
            for file_path in file_list:  # 遍历所有文档
                file_name = class_path + file_path  # 拼出文件名全路径
                self.data_set.filenames.append(file_name)  # 把文件路径附加到数据集中
                self.data_set.label.append(self.data_set.target_name.index(mydir))  # 把文件分类标签附加到数据集中
                file_read = open(file_name, 'r', encoding='utf-8')  # 打开一个文件
                seg_corpus = file_read.read()  # 读取语料
                self.data_set.contents.append(seg_corpus)  # 构建分词文本内容列表
                file_read.close()

        # 词袋对象持久化
        file_obj = open(self.wordbag_path + self.trainset_name, "wb")
        pickle.dump(self.data_set, file_obj)
        file_obj.close()
        print('分词语料打包成功完成!')

    # 导出训练语料集
    def load_trainset(self):
        file_obj = open(self.wordbag_path + self.trainset_name, 'rb')
        self.data_set = pickle.load(file_obj)
        file_obj.close()



# 基本业务流程：
# 1. 分类语料预处理
# 2. 预处理后语料分词
# 3. 分词训练语料进行打包，即创建dat文件
# 4. 计算tf-idf权值并持久化为词袋文件
# 5. 检测打包后的分词训练语料
# 6. 检测词袋文件

if __name__ == '__main__':
    print('start.....')
    # 实例化这个类

    tp = TextPreprocess()
    tp.user_dic_path = 'data/自定义法律词典.txt'  # 自定义词典
    tp.corpus_path = "data/corpus/"  # 预处理后语料路径
    tp.segment_path = "data/segment/"  # 分词后语料路径
    tp.wordbag_path = "data/wordbag/"  # 词袋模型路径
    tp.trainset_name = "trainset.dat"  # 训练集文件名

    # 添加自定义词典
    print('load userdict...')
    jieba.load_userdict(tp.user_dic_path)
    print('load userdict finish!')

    # 2. 预处理后语料分词
    tp.segment()
    # 3.分词训练语料进行打包: data_set = Bunch(target_name=[], label=[], filenames=[], contents=[])
    tp.create_dat_file()


