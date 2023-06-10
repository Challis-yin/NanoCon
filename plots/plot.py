import matplotlib.pyplot as plt
import numpy as np
from pyecharts.render import make_snapshot
from snapshot_pyppeteer import snapshot
from pyecharts import options as opts
from pyecharts.charts import Bar
# from pyecharts.charts import Barh
from pyecharts.charts import Pie
from pyecharts.globals import ThemeType

def draw_pyecharts(xaxis, data, config, name):
    """
    @param data: 二维数组，每行对应一个方法抛出的结果数据，每列对应lable
    @param data_temp: 字典类型，存储了一些配置信息，画图需要用到 “names”对应的列表，存储了画图生成的图片的方法名称
    @param config: 与上面相同
    """
    """demo
    data = [[0.665, 0.6266666666666667, 0.7033333333333334, 0.7083666666666667, 0.33097412973151036]]
    lable = ['ACC', 'AUPRC', 'AUROC', 'Precision', 'Recall', 'F1Score']
    """
    bar = Bar(init_opts=opts.InitOpts(width="600px", height="350px", theme=ThemeType.LIGHT))
    bar.add_xaxis(xaxis)
    bar.add_yaxis("Ours", data['Ours'], stack="stack1")
    bar.add_yaxis("RNN", data['RNN'], stack="stack2")
    bar.add_yaxis("methBERT_weighted", data['methBERT_weighted'], stack="stack3")
    bar.set_global_opts(toolbox_opts=opts.ToolboxOpts(is_show=False), title_opts=opts.TitleOpts(title=name))
    bar.set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    # bar.transpose() 
    make_snapshot(snapshot, bar.render(), '{}/{}.{}'.format(config['savepath'], name, 'svg'))
    make_snapshot(snapshot, bar.render(), '{}/{}.{}'.format(config['savepath'], name, 'png'))
    make_snapshot(snapshot, bar.render(), '{}/{}.{}'.format(config['savepath'], name, 'pdf'))
    # make_snapshot(snapshot, bar.render(), '{}/{}.{}'.format(config['savepath'], 'length_statistics_train_pyecharts', 'svg'))
    print('svg绘制成功')
    bar.render("/mnt/sde/ycl/Nanopore_program_copy/util/name.svg")

def draw_pyecharts2(xaxis, data, config, name):
    """
    @param data: 二维数组，每行对应一个方法抛出的结果数据，每列对应lable
    @param data_temp: 字典类型，存储了一些配置信息，画图需要用到 “names”对应的列表，存储了画图生成的图片的方法名称
    @param config: 与上面相同
    @return: NAN（快跑，不要维护）
    """
    """demo
    data = [[0.665, 0.6266666666666667, 0.7033333333333334, 0.7083666666666667, 0.33097412973151036]]
    lable = ['ACC', 'AUPRC', 'AUROC', 'Precision', 'Recall', 'F1Score']
    """
    bar = Bar(init_opts=opts.InitOpts(width="400px", height="350px", theme=ThemeType.LIGHT))
    bar.add_xaxis(xaxis)
    bar.add_yaxis("Ours", data['Ours'], stack="stack1")
    bar.add_yaxis("RNN", data['RNN'], stack="stack2")
    bar.add_yaxis("methBERT_weighted", data['methBERT_weighted'], stack="stack3")
    bar.set_global_opts(toolbox_opts=opts.ToolboxOpts(is_show=False))
    bar.set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    # bar.transpose() 
    make_snapshot(snapshot, bar.render(), '{}/{}.{}'.format(config['savepath'], name, 'svg'))
    make_snapshot(snapshot, bar.render(), '{}/{}.{}'.format(config['savepath'], name, 'png'))
    make_snapshot(snapshot, bar.render(), '{}/{}.{}'.format(config['savepath'], name, 'pdf'))
    # make_snapshot(snapshot, bar.render(), '{}/{}.{}'.format(config['savepath'], 'length_statistics_train_pyecharts', 'svg'))
    print('svg绘制成功')
    bar.render("/mnt/sde/ycl/Nanopore_program_copy/util/name.svg")


if __name__ == '__main__':
    models = ['ACC', 'AUPRC', 'AUROC', 'Precition', 'Recall', 'F1Score']
    metrics1 = {
        'Ours': [0.9737,0.9766,0.9934,0.9371,0.9131,0.9254],
        'RNN': [0.9594,0.9503,0.9867,0.8991,0.8703,0.8844],
        'methBERT': [0.9549,0.9318,0.9804,0.8913,0.8517,0.8708]
    }
    metrics2 = {
        'Ours': [0.9924,0.7223,0.9809,0.7865,0.5699,0.6557],
        'RNN': [0.9909,0.6383,0.9763,0.7318,0.4608,0.5579],
        'methBERT_weighted': [0.9819,0.6676,0.9824,0.4147,0.7978,0.5424]
    }
    config = {
        'savepath': '/mnt/sde/ycl/Nanopore_program_copy/util'
    }
    name1 = 'O.sativa'
    name2 = 'A.thaliana'
    # draw_pyecharts(models, metrics2, config, name2)
    # draw_pyecharts(models, metrics2, config, name2)
    
    
    # metrics3 = {
    #     'Ours': [0.7522,0.9848,0.4538,0.8212],
    #     'RNN': [0.6458,0.9831,0.4039,0.7863],
    #     'methBERT': [0.7386,0.9813,0.4714,0.8163]
    # }
    # name3 = 'rice_tha'
    models1 = ['AUPRC', 'AUROC', 'Precision', 'Recall']
    # draw_pyecharts2(models1, metrics3, config, name3)
    metrics4 = {
        'Ours': [0.9011,0.9688,0.9625,0.5156],
        'RNN': [0.6669,0.9815,0.3391,0.8393],
        'methBERT_weighted': [0.8970,0.9711,0.8480,0.7897]
    }
    name4 = 'tha_rice'
    draw_pyecharts2(models1, metrics4, config, name4)
