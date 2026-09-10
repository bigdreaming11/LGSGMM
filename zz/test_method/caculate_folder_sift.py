import os
import time
import re

from zz.test_method.score import single_scores
from zz.test_method.process_pair_sift import process_pair_sift

from duibi_ceshi.zz_sift import plot_sift

# def extract_number(filename):
#     # 使用正则表达式匹配文件名中的数字部分
#     match = re.search(r'\d+', filename)  # 匹配第一个连续的数字
#     return int(match.group()) if match else float('inf')
def extract_number(filename):
    # 匹配文件名中所有数字
    numbers = re.findall(r'\d+', filename)
    # 取最后一个数字作为排序依据
    return int(numbers[-1]) if numbers else float('inf')



def test_folder_sift(img_folder_a, txt_folder_a, img_folder_b, txt_folder_b, output_folder, if_pic):
        """
        测试两个文件夹中的图片和 txt 文件，并保存结果
        :param img_folder_a: 图片 A 文件夹路径
        :param txt_folder_a: 文本文件 A 文件夹路径
        :param img_folder_b: 图片 B 文件夹路径
        :param txt_folder_b: 文本文件 B 文件夹路径
        :param output_folder: 输出结果保存文件夹路径
        """
        os.makedirs(output_folder, exist_ok=True)

        # 按照文件名中的数字排序
        img_files_a = sorted([f for f in os.listdir(img_folder_a) if f.endswith((".jpg", ".png"))],key=extract_number)
        txt_files_a = sorted([f for f in os.listdir(txt_folder_a) if f.endswith(".txt")],key=extract_number)
        img_files_b = sorted([f for f in os.listdir(img_folder_b) if f.endswith((".jpg", ".png"))],key=extract_number)
        txt_files_b = sorted([f for f in os.listdir(txt_folder_b) if f.endswith(".txt")],key=extract_number)
        # 确保文件数量一致
        assert len(img_files_a) == len(txt_files_a), "Mismatch between images and texts in folder A"
        assert len(img_files_b) == len(txt_files_b), "Mismatch between images and texts in folder B"
        assert len(img_files_a) == len(img_files_b), "Mismatch between folder A and folder B"

        time_each=[]    
        num_pict=0
        test_scores=[]
        test_precision=[]
        test_recall=[]

        time_each_new=[]
        test_scores_new=[]
        test_precision_new=[]
        test_recall_new=[]
        # 遍历每对图片和 txt 文件
        for img_file_a, txt_file_a, img_file_b, txt_file_b in zip(img_files_a, txt_files_a, img_files_b, txt_files_b):
            print(f"Processing: {img_file_a} & {img_file_b}")

            # 构造完整路径
            img_path_a = os.path.join(img_folder_a, img_file_a)
            img_path_b = os.path.join(img_folder_b, img_file_b)
            txt_path_a = os.path.join(txt_folder_a, txt_file_a)
            txt_path_b = os.path.join(txt_folder_b, txt_file_b)

            # 输出结果路径
            output_path = os.path.join(output_folder, f"{os.path.splitext(img_file_a)[0]}_match.png")

            # 处理单对图片
            start_time=time.time()
            boxes0 ,boxes1,image0,image1,matches,H,index=process_pair_sift(img_path_a, txt_path_a, img_path_b, txt_path_b, output_path)
            end_time=time.time()
            dd=(end_time-start_time)*1000
            time_each.append((end_time-start_time)*1000)
            num_pict+=1
            print(f"算法的运推理时间：{(end_time-start_time)*1000}ms")

            single_score,single_precision,single_recall=single_scores(boxes0 ,boxes1,matches)

            # print(f"过滤前:{single_score}")
            # print(f"过滤后:{single_score_new}")
            test_scores.append(single_score)
            test_precision.append(single_precision)
            test_recall.append(single_recall)

####################################################################画图方便
            if if_pic==1:
                plot_sift(matches, image0, image1, boxes0, boxes1, output_folder, img_file_a,single_score,single_precision,single_recall,dd)
            ##############################################################
            #计算单个损失，然后append



        
        mean_time = sum(time_each) / len(time_each)
        mean_score=sum(test_scores) / len(test_scores)
        mean_precision=sum(test_precision)/len(test_precision)
        mean_reacll=sum(test_recall)/len(test_recall)
        print(f"**************测试完毕*******************")
        print(f"测试集平均推理速度：{mean_time}ms")
        print(f"测试集平均MDA分数：{mean_score}")
        print(f"测试集平均准确率：{mean_precision}")
        print(f"测试集平均召回率：{mean_reacll}")