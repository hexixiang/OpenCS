打分程序包含五个

1、api_score_pairs    用于成对打分，需要提示词路径，源文件路径

2、api_client                api接口类函数

3、sorted_1000                用于排序提取不重复的前一千条数据

4、trans                训练数据格式转换

5、combine                    用于对并行打分的程序的整合

成对打分prompt提示词路径在

prompt/score/myprompt

论文WHAT：

prompt/score/ALIGNMENT

论文alpagasus：

prompt/score/alpagasus

文件运行：

首先api_score_pairs，生成打分结果，多开进程并行，

soretd_1000, 提取分数并排序，每类提取前20，去重

score_trans, 文件转换为训练数据格式