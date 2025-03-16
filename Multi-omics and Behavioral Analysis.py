"""
多组学与行为学联合分析系统 - 完整版（含肠道菌群）
作者：王杰林 
邮箱：2390448675@qq.com
版本：6.7
功能：数据预处理、样本对齐、相关性分析、网络分析、可视化
最后更新：2025-03-17-03:43
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import logging

# ------------------------- 配置参数 -------------------------
CONFIG = {
    'correlation_threshold': 0.6,  # 相关系数阈值
    'p_threshold': 0.05,  # p值阈值
    'top_features': 30,  # 相关性分析中显示的前N个特征
    'output_dir': 'results',  # 输出目录
    'cmap': 'RdYlBu_r',  # 色彩映射
    'dpi': 300,  # 图像分辨率
    'font_family': 'Arial',  # 字体
    'cbar_kws': {  # colorbar参数
        'shrink': 0.6,  # 缩放比例
        'aspect': 15  # 长宽比
    },
    'cbar_label_fontsize': 12,  # colorbar标签字体大小
    'cbar_tick_fontsize': 10,  # colorbar刻度字体大小
    'p_thresholds': [0.001, 0.01, 0.05],  # p值阈值
    'p_symbols': ['***', '**', '*']  # p值阈值对应的符号
}
# ------------------------- 配置参数更新 -------------------------
CONFIG.update({
    'network_layout': 'circular',  # 新增网络布局选项
    'integrated_title_fontsize': 24  # 增大综合热图标题
})

# 设置基本绘图样式
def set_plotting_style():
    """设置绘图样式"""
    plt.style.use('default')  # 重置为默认样式
    plt.rcParams.update({
        'font.family': CONFIG['font_family'],
        'figure.dpi': CONFIG['dpi'],
        'savefig.dpi': CONFIG['dpi'],
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.titlesize': 16,
        'figure.figsize': [10, 8]
    })
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)

def load_and_process_data():
    """加载并预处理所有数据"""
    print("正在加载数据...")
    try:
        # 加载转录组数据
        trans = pd.read_csv("transcriptomics.csv")
        if 'gene_id' not in trans.columns:
            raise ValueError("转录组数据缺少gene_id列")
        trans.set_index('gene_id', inplace=True)
        trans = trans.T  # 转置使样本为行

        # 加载代谢组数据
        metab = pd.read_csv("metabolomics.csv")
        if 'metabolite_id' not in metab.columns:
            raise ValueError("代谢组数据缺少metabolite_id列")
        metab.set_index('metabolite_id', inplace=True)
        metab = metab.T  # 转置使样本为行

        # 加载肠道菌群数据
        gut = pd.read_csv("gut_microbiome.csv")
        if 'microbe_id' not in gut.columns:
            raise ValueError("肠道菌群数据缺少microbe_id列")
        gut.set_index('microbe_id', inplace=True)
        gut = gut.T  # 转置使样本为行

        # 加载行为学数据
        behav = pd.read_csv("behavior.csv")
        if 'MouseID' not in behav.columns:
            raise ValueError("行为学数据缺少MouseID列")
        behav.set_index('MouseID', inplace=True)

        # 获取共同样本
        common_samples = ['Con1', 'Con2', 'Con3',
                          'CUMS1', 'CUMS2', 'CUMS3',
                          'NG1', 'NG2', 'NG3']

        # 检查样本是否存在
        for sample in common_samples:
            if sample not in trans.index:
                raise ValueError(f"转录组数据中缺少样本: {sample}")
            if sample not in metab.index:
                raise ValueError(f"代谢组数据中缺少样本: {sample}")
            if sample not in gut.index:
                raise ValueError(f"肠道菌群数据中缺少样本: {sample}")
            if sample not in behav.index:
                raise ValueError(f"行为学数据中缺少样本: {sample}")

        # 提取共同样本的数据
        trans = trans.loc[common_samples]
        metab = metab.loc[common_samples]
        gut = gut.loc[common_samples]
        behav = behav.loc[common_samples]

        print("\n数据维度:")
        print(f"转录组: {trans.shape} (样本数 × 基因数)")
        print(f"代谢组: {metab.shape} (样本数 × 代谢物数)")
        print(f"肠道菌群: {gut.shape} (样本数 × 微生物数)")
        print(f"行为学: {behav.shape} (样本数 × 行为指标数)")

        # 检查数据是否包含缺失值
        if trans.isnull().any().any():
            print("警告：转录组数据包含缺失值")
        if metab.isnull().any().any():
            print("警告：代谢组数据包含缺失值")
        if gut.isnull().any().any():
            print("警告：肠道菌群数据包含缺失值")
        if behav.isnull().any().any():
            print("警告：行为学数据包含缺失值")

        # 标准化数据
        scaler = StandardScaler()
        trans_scaled = pd.DataFrame(scaler.fit_transform(trans), index=trans.index, columns=trans.columns)
        metab_scaled = pd.DataFrame(scaler.fit_transform(metab), index=metab.index, columns=metab.columns)
        gut_scaled = pd.DataFrame(scaler.fit_transform(gut), index=gut.index, columns=gut.columns)
        behav_scaled = pd.DataFrame(scaler.fit_transform(behav), index=behav.index, columns=behav.columns)

        return trans_scaled, metab_scaled, gut_scaled, behav_scaled
    except Exception as e:
        print(f"数据加载失败：{str(e)}")
        raise

def calculate_correlations(data1, data2, prefix=""):
    """计算两个数据集之间的相关性"""
    print(f"\n正在计算{prefix}相关性...")
    try:
        corr_matrix = pd.DataFrame(index=data1.columns, columns=data2.columns, dtype=float)
        pval_matrix = corr_matrix.copy()
        for feat1 in data1.columns:
            for feat2 in data2.columns:
                if data1[feat1].std() == 0 or data2[feat2].std() == 0:
                    corr, pval = np.nan, np.nan
                else:
                    corr, pval = pearsonr(data1[feat1], data2[feat2])
                corr_matrix.loc[feat1, feat2] = corr
                pval_matrix.loc[feat1, feat2] = pval
        return corr_matrix, pval_matrix
    except Exception as e:
        print(f"相关性计算失败：{str(e)}")
        raise

def generate_annotation_symbols(pval_matrix):
    """基于p值生成显著性标记"""
    symbols = pd.DataFrame('', index=pval_matrix.index, columns=pval_matrix.columns)
    symbols[:] = np.where(pval_matrix < 0.001, '***',
                         np.where(pval_matrix < 0.01, '**',
                                 np.where(pval_matrix < 0.05, '*', '')))
    return symbols

def plot_correlation_heatmap(corr_matrix, pval_matrix, title, output_path, figsize=None):
    """修改后的热图绘制函数"""
    try:
        set_plotting_style()
        if figsize is None:
            figsize = (max(12, corr_matrix.shape[1]*0.5), max(10, corr_matrix.shape[0]*0.5))
        plt.figure(figsize=figsize)
        mask = np.triu(np.ones_like(corr_matrix)) if corr_matrix.shape[0]==corr_matrix.shape[1] else None
        annot_matrix = generate_annotation_symbols(pval_matrix)
        sns.heatmap(
            corr_matrix,
            mask=mask,
            cmap=CONFIG['cmap'],
            center=0,
            annot=annot_matrix,
            fmt='',
            linewidths=0.5,
            xticklabels=True,
            yticklabels=True,
            square=True,
            cbar_kws={"shrink": .8}
        )
        plt.title(title, fontsize=14, pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"热图绘制失败：{str(e)}")
        raise

def plot_4x4_correlation_grid(corr_dict, pval_dict, output_path):
    """修复版4x4热图网格"""
    try:
        set_plotting_style()
        fig = plt.figure(figsize=(28, 28))  # 增大画布尺寸
        gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.4)
        axes = gs.subplots()

        # 数据类型定义（使用统一前缀）
        data_pairs = [
            ('Tran', 'Transcriptomics'),
            ('Meta', 'Metabolomics'),
            ('Gut', 'GutMicrobiome'),  # 关键修改：使用Gut作为前缀
            ('Beha', 'Behavior')
        ]

        for i in range(4):
            for j in range(4):
                ax = axes[i, j]
                prefix1, label1 = data_pairs[i]
                prefix2, label2 = data_pairs[j]
                key_forward = f"{prefix1}_{prefix2}"
                key_reverse = f"{prefix2}_{prefix1}"

                # 获取相关性和p值矩阵
                if key_forward in corr_dict:
                    corr_matrix = corr_dict[key_forward]
                    pval_matrix = pval_dict[key_forward]
                elif key_reverse in corr_dict:
                    corr_matrix = corr_dict[key_reverse].T
                    pval_matrix = pval_dict[key_reverse].T
                else:
                    raise KeyError(f"Missing keys: {key_forward} and {key_reverse}")

                # 生成星号标注
                annot_matrix = generate_annotation_symbols(pval_matrix)

                # 绘制热图
                sns.heatmap(
                    corr_matrix,
                    mask=np.triu(np.ones_like(corr_matrix)) if i > j else None,
                    cmap=CONFIG['cmap'],
                    center=0,
                    annot=annot_matrix,  # 使用星号标注
                    fmt='',  # 空格式
                    linewidths=0.5,
                    xticklabels=True,
                    yticklabels=True,
                    ax=ax,
                    cbar=False,
                    square=True
                )

                # 动态调整标签显示
                ax.set_xticklabels(
                    [label.split('_')[-1][:15] for label in corr_matrix.columns],  # 截断长标签
                    rotation=45,
                    ha='right',
                    fontsize=8
                )
                ax.set_yticklabels(
                    [label.split('_')[-1][:15] for label in corr_matrix.index],
                    fontsize=8
                )

                # 添加子标题说明维度
                ax.text(0.5, -0.15, 
                       f'({corr_matrix.shape[0]}x{corr_matrix.shape[1]})',
                       transform=ax.transAxes,
                       ha='center',
                       fontsize=8)

        plt.suptitle("Multi-omics Correlation Analysis", fontsize=16, y=1.02)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
    except Exception as e:
        logging.error(f"网格图绘制失败：{str(e)}")
        raise
    
def plot_integrated_heatmap(trans, metab, gut, behav,pval_dict, output_path):
    """综合大热图"""
    try:
        set_plotting_style()
        combined = pd.concat([
            trans.add_prefix('Gene_'),
            metab.add_prefix('Metab_'),
            gut.add_prefix('Gut_'),
            behav.add_prefix('Behav_')
        ], axis=1)

         # 新的p值矩阵拼接逻辑
        def get_pval_matrix(prefix1, prefix2):
            """智能获取p值矩阵"""
            key = f"{prefix1}_{prefix2}"
            reverse_key = f"{prefix2}_{prefix1}"
            if key in pval_dict:
                return pval_dict[key]
            elif reverse_key in pval_dict:
                return pval_dict[reverse_key].T
            else:
                raise KeyError(f"Missing both {key} and {reverse_key}")

        # 按区块拼接p值矩阵
        pval_blocks = [
            [get_pval_matrix('Tran', 'Tran'), get_pval_matrix('Tran', 'Meta'), 
             get_pval_matrix('Tran', 'Gut'), get_pval_matrix('Tran', 'Beha')],
            [get_pval_matrix('Meta', 'Tran'), get_pval_matrix('Meta', 'Meta'),
             get_pval_matrix('Meta', 'Gut'), get_pval_matrix('Meta', 'Beha')],
            [get_pval_matrix('Gut', 'Tran'), get_pval_matrix('Gut', 'Meta'),
             get_pval_matrix('Gut', 'Gut'), get_pval_matrix('Gut', 'Beha')],
            [get_pval_matrix('Beha', 'Tran'), get_pval_matrix('Beha', 'Meta'),
             get_pval_matrix('Beha', 'Gut'), get_pval_matrix('Beha', 'Beha')]
        ]
        
        full_pval = pd.concat(
            [pd.concat(row, axis=1) for row in pval_blocks],
            axis=0
        )
         # 生成标注
        annot_matrix = generate_annotation_symbols(full_pval)
        plt.figure(figsize=(20, 18))
        ax = sns.heatmap(
            combined.corr(),
            mask=np.triu(np.ones_like(combined.corr(), dtype=bool)),
            cmap=CONFIG['cmap'],
            center=0,
            annot=annot_matrix,  # 添加p值标注
            fmt='',
            linewidths=0.5,
            cbar_kws={**CONFIG['cbar_kws'], 'label': 'Correlation Coefficient'},
            xticklabels=False
        )
        plt.title("Integrated Multi-omics and Behavior Analysis", 
                 fontsize=CONFIG['integrated_title_fontsize'],  # 使用配置的字体大小
                 pad=30)  # 增加标题间距
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"综合热图错误: {str(e)}")
        raise

def plot_network(gene_corr, gene_p, metab_corr, metab_p, gut_corr, gut_p, output_path):
    """最终修复版网络图绘制"""
    print("\n构建关联网络...")
    try:
        set_plotting_style()
        plt.figure(figsize=(20, 16))
        G = nx.Graph()

        def add_significant_correlations(corr_df, pval_df, node_type):
            for feat in corr_df.index:
                for behav in corr_df.columns:
                    if (abs(corr_df.loc[feat, behav]) > CONFIG['correlation_threshold'] 
                        and pval_df.loc[feat, behav] < CONFIG['p_threshold']):
                        G.add_node(feat, type=node_type)
                        G.add_node(behav, type='Behavior')
                        G.add_edge(feat, behav, weight=corr_df.loc[feat, behav])

        add_significant_correlations(gene_corr, gene_p, 'Gene')
        add_significant_correlations(metab_corr, metab_p, 'Metabolite')
        add_significant_correlations(gut_corr, gut_p, 'Microbe')

        if len(G.nodes) == 0:
            print("警告：没有找到显著相关性，网络为空")
            return

        # 使用 spring_layout 布局
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

        # 计算节点大小（基于连接度）
        degrees = dict(G.degree())
        node_sizes = [degrees[node] * 100 + 500 for node in G.nodes()]

        # 设置节点颜色
        node_colors = ['#8ecae6' if G.nodes[node]['type'] == 'Gene' else
                       '#95d5b2' if G.nodes[node]['type'] == 'Metabolite' else
                       '#ff7f0e' if G.nodes[node]['type'] == 'Microbe' else
                       '#ffb4a2' for node in G.nodes()]

        # 绘制节点
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes)

        # 绘制边（根据相关性系数调整边的宽度）
        edges = G.edges(data='weight')
        edge_widths = [abs(w) * 5 for u, v, w in edges]
        edge_colors = [w for u, v, w in edges]

        # 绘制边并添加颜色条
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu_r, norm=plt.Normalize(vmin=-1, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca(), shrink=0.5)
        cbar.set_label('Correlation Coefficient', fontsize=20)

        nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, edge_cmap=plt.cm.RdYlBu_r, alpha=0.7)

        # 绘制节点标签
        nx.draw_networkx_labels(G, pos, font_size=12, font_family='Arial')

        # 创建分类图例（关键修改）
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#8ecae6', markersize=15, label='Gene'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#95d5b2', markersize=15, label='Metabolite'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff7f0e', markersize=15, label='Microbe'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ffb4a2', markersize=15, label='Behavior')
        ]
        plt.legend(
            handles=legend_elements,
            loc='upper left',
            bbox_to_anchor=(0.90, 0.05),
            title='Node Types (Size indicates connection count)',
            frameon=True,
            fontsize=20,
            title_fontsize=20
        )

        plt.title("Multi-omics Network and Behavior Analysis", fontsize=30, pad=20)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
    except Exception as e:
        logging.error(f"网络图绘制失败：{str(e)}")
        raise

def main():
    try:
        # 创建输出目录
        output_dir = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(output_dir, exist_ok=True)
        CONFIG['output_dir'] = output_dir
        print(f"分析开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"输出目录: {os.path.abspath(output_dir)}")

        # 加载和处理数据
        trans, metab, gut, behav = load_and_process_data()

        # 计算所有可能的相关性组合
        corr_results = {}
        pval_results = {}

        # 转录组内部相关性
        corr, pval = calculate_correlations(trans, trans, "转录组内部")
        corr_results['Tran_Tran'] = corr
        pval_results['Tran_Tran'] = pval

        # 代谢组内部相关性
        corr, pval = calculate_correlations(metab, metab, "代谢组内部")
        corr_results['Meta_Meta'] = corr
        pval_results['Meta_Meta'] = pval

        # 肠道菌群内部相关性
        corr, pval = calculate_correlations(gut, gut, "肠道菌群内部")
        corr_results['Gut_Gut'] = corr
        pval_results['Gut_Gut'] = pval

        # 行为学内部相关性
        corr, pval = calculate_correlations(behav, behav, "行为学内部")
        corr_results['Beha_Beha'] = corr
        pval_results['Beha_Beha'] = pval

        # 转录组-代谢组相关性
        corr, pval = calculate_correlations(trans, metab, "转录组-代谢组")
        corr_results['Tran_Meta'] = corr
        pval_results['Tran_Meta'] = pval

        # 转录组-肠道菌群相关性
        corr, pval = calculate_correlations(trans, gut, "转录组-肠道菌群")
        corr_results['Tran_Gut'] = corr
        pval_results['Tran_Gut'] = pval

        # 转录组-行为学相关性
        corr, pval = calculate_correlations(trans, behav, "转录组-行为学")
        corr_results['Tran_Beha'] = corr
        pval_results['Tran_Beha'] = pval

        # 代谢组-肠道菌群相关性
        corr, pval = calculate_correlations(metab, gut, "代谢组-肠道菌群")
        corr_results['Meta_Gut'] = corr
        pval_results['Meta_Gut'] = pval

        # 代谢组-行为学相关性
        corr, pval = calculate_correlations(metab, behav, "代谢组-行为学")
        corr_results['Meta_Beha'] = corr
        pval_results['Meta_Beha'] = pval

        # 肠道菌群-行为学相关性
        corr, pval = calculate_correlations(gut, behav, "肠道菌群-行为学")
        corr_results['Gut_Beha'] = corr
        pval_results['Gut_Beha'] = pval
         # 系统化计算所有组合
        prefixes = ['Tran', 'Meta', 'Gut', 'Beha']
        datasets = {
            'Transcriptomics': trans,
            'Metabolomics': metab,
            'GutMicrobiome': gut,  # 确保使用正确的名称
            'Behavior': behav
        }
        
        for data1_name, data1 in datasets.items():
            for data2_name, data2 in datasets.items():
                prefix1 = data1_name[:4]  # 截取前4个字符作为前缀
                prefix2 = data2_name[:4]
                key = f"{prefix1}_{prefix2}"
                corr, pval = calculate_correlations(data1, data2, f"{data1_name}-{data2_name}")
                corr_results[key] = corr
                pval_results[key] = pval
        # 生成可视化结果
        print("\n生成可视化结果...")

        # 单独的相关性热图
        for name, corr in corr_results.items():
            plot_correlation_heatmap(
                corr_results[name],
                pval_results[name],
                f"{name} Correlation Analysis",
                f"{CONFIG['output_dir']}/{name}_heatmap.png"
            )

         # 生成可视化结果时传递pval_dict
        plot_4x4_correlation_grid(corr_results, pval_results, f"{CONFIG['output_dir']}/4x4_correlation_grid.png")
        
        # 综合热图需要pval_dict
        plot_integrated_heatmap(trans, metab, gut, behav, pval_results, f"{output_dir}/integrated.png")
        # 网络图
        plot_network(
            corr_results['Tran_Beha'], pval_results['Tran_Beha'],
            corr_results['Meta_Beha'], pval_results['Meta_Beha'],
            corr_results['Gut_Beha'], pval_results['Gut_Beha'],
            f"{CONFIG['output_dir']}/network.png"
        )

       
        print("\n保存相关性矩阵...")
        for name, corr in corr_results.items():
            corr.to_csv(f"{CONFIG['output_dir']}/{name}_correlations.csv")
            pval_results[name].to_csv(f"{CONFIG['output_dir']}/{name}_pvalues.csv")

        print("\n分析成功完成！")
        print(f"结果保存在：{os.path.abspath(CONFIG['output_dir'])}")
        print(f"分析结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    except Exception as e:
        print(f"\n错误发生：{str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    main()