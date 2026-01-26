1. 在当前的Result目录当中有我们在四台设备上得到的实验结果。综合结果放在result_parsed.csv当中，现在我们需要画柱状图来展示我们的实验结果。现在，为我编写一个python脚本，画出柱状图。要求画出符合学术论文标准的专业柱状图，注意各种图例与配色。并且，python脚本可以通过输入参数决定画哪个配置的柱状图，决定每个配置中哪些优化方案会被图中。在Result目录当中实现draw_column.py脚本实现我要求的功能。
   完成

2. 修改当前的draw_column.py脚本，现在使用C:\workspace\Workspace\Workspace\EdgeMamba\Result\result_final.xlsx作为实验结果绘图

当前的图例字体好像有些太小了，字体放大一点，确保在论文的缩小图上也能看清
完成

3. C:\workspace\Workspace\Workspace\EdgeMamba\Result\Log 在此目录当中，我们有一些训练过程中的日志文件，我想将模型的准确率随着训练epochs数增长提取出来。我们先来提取log文件当中的内容，其中，auto vim.log当中可能每20轮epoch就会测试多种配置，对于这种情况，你也应该将结果提取出来并且保存。现在写一个python脚本提取全部log文件中的准确率与epochs，然后将其保存到此目录下result.xlsx文件当中。
   完成

4. 现在在result.xlsx当中有一个summary的sheet是我整理好的数据，现在你需要根据C:\workspace\Workspace\Workspace\EdgeMamba\draw.md当中的画图要求开始画图。一共包括EdgeVim Max Model	EdgeVim Random Model 1	EdgeVim Random Model 2	EdgeVim Min Model	Vim tiny	Vim Small
这几项，其中，EdgeVim Random Model 1	EdgeVim Random Model 2画成散点图，其余全部画成折线图，EdgeVim Min Model是每隔20个epoch才有一个点，需要注意以下。那么在Reuslt目录下写一个traning_curve.py脚本实现画图的功能。

在最后299 epoch的时候，给max model和vim small加一个标记，显示max model在vim small上面
完成

5. 现在我们来整理C:\workspace\Workspace\Workspace\EdgeMamba\Result\Log\random_search_all_results.json当中的内容，在result.xlsx中增加一个sheet，提取文件当中每个模型的params，flops以及准确率即可。写一个extract_random_config.py脚本完成这件事
   完成

6. 当前的绘图放在双栏上还是太小了，把绘图脚本的字号放大一些，确保双栏也能看清楚，