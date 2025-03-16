# if (!requireNamespace("BiocManager", quietly = TRUE))
#     install.packages("BiocManager")
# BiocManager::install("DESeq2")

library(DESeq2)
library(readxl)  # 用于读取 Excel 文件
library(ggplot2) # 可选，用于可视化

# 读取 Excel 文件
file_path <- "C:\\Users\\23904\\Desktop\\转录组学\\final_processed_gene_count.xlsx"
data <- read_excel(file_path)

# 查看数据结构
head(data)

# 提取样本名和基因名
gene_names <- data$Gene  # 第一列为基因名
count_matrix <- as.matrix(data[, -1])  # 去掉第一列（基因名）
rownames(count_matrix) <- gene_names  # 设置基因为行名

# 创建样本分组信息
sample_info <- data.frame(
    Sample = colnames(count_matrix),
    Condition = gsub("_\\d+$", "", colnames(count_matrix)) # 提取组名（去掉数字后缀）
)

# 构建 DESeq2 数据集
dds <- DESeqDataSetFromMatrix(
    countData = count_matrix,
    colData = sample_info,
    design = ~ Condition
)

# keep <- rowSums(counts(dds)) >= 10
# dds <- dds[keep, ]

# 运行 DESeq2
dds <- DESeq(dds)

# 定义比较组
comparison_pairs <- list(c("Con", "CUMS"), c("NG", "CUMS"))

for (pair in comparison_pairs) {
    group1 <- pair[1]
    group2 <- pair[2]
    
    # 提取差异分析结果
    res <- results(dds, contrast = c("Condition", group1, group2))
    
    # 转换为数据框
    res_df <- as.data.frame(res)
    
    # 添加基因名列
    res_df <- cbind(Gene = rownames(res_df), res_df)  # 将行名作为基因名列
    
    # 保存结果到 CSV 文件
    output_file <- paste0(group1, "_VS_", group2, "_DESeq2_results.csv")
    write.csv(res_df, file = output_file, row.names = FALSE)  # 不保存行名
    
    cat("完成分析：", output_file, "\n")
}