## 2016CCF [大数据精准营销中搜狗用户画像挖掘](http://www.wid.org.cn/data/science/player/competition/detail/description/239)  final winner solution

###李恒超、[李裕礞](https://github.com/liyumeng)、王安然、钱凌飞、任璐 
###大连理工大学信息检索实验室

重现实验
========
把原始数据　user_tag_query.10W.TRAIN　和　user_tag_query.10W.TEST 放在`./data/`目录下，然后运行`run.sh`
最后能生成　tfidf_dm_dbow_20W.csv,为方便大家阅读，代码已整理成最简洁的形式，该结果B榜成绩会在**0.724**左右。
建议在**ubuntu**环境下运行，windows环境下，也可以运行run_cv.ipynb

数据下载
=========
https://pan.baidu.com/s/1bpGIfxX
提取码：kcnm

依赖
=======
* Anaconda 4.2.0(Python 3.5 version)
* jieba 0.38
* keras 1.1.0
* xgboost 0.6
* gensim 0.13.2

平台
============
**ubuntu 16.04**

硬件
=======
本代码是在8core,i7CPU,8gb RAM 的电脑上开发的.总共运行时间大概需要5个小时

其他
=======
如果觉得不错的话，欢迎大家点击右上角的star和fork,谢谢！

我们参加的其他竞赛：

[2nd Place Solution for SMP CUP 2016](https://github.com/liyumeng/SmpCup2016)

![](data/队旗.jpg)

