# 包加载 + 用途说明
library(dplyr)
library(ggplot2)
library(caret)
library(e1071)
library(randomForest)
library(xgboost)
library(keras)
library(corrplot)
library(pROC)
library(readr)
library(scales)

# ===================== 1. 路径配置 =====================
# 数据文件放在仓库根目录的 data/ 文件夹下，图表统一输出到 figures/ 文件夹
fig_dir <- "figures"
if (!dir.exists(fig_dir)) dir.create(fig_dir)

# ===================== 2. 读取原始交易数据 =====================
data_raw <- read_csv("data/bank_fraud.csv")

# 基础查看
head(data_raw)
str(data_raw)
summary(data_raw)

# ===================== 3. 剔除冗余字段 =====================
drop_var <- c("transaction_id","customer_id","transaction_date","transaction_time","fraud_type")
data_clean <- data_raw[, !colnames(data_raw) %in% drop_var]

# 输出原始数据维度
cat("原始全量数据维度：", dim(data_raw), "\n")

# ===================== 4. 缺失值填充 =====================
# ===================== 4.1 缺失值统计、排序、可视化 =====================
# 复制原始未填充数据集
data_raw_na <- data_clean

# 1. 统计每个变量缺失数量、缺失百分比
na_summary <- tibble(
  Variable = colnames(data_raw_na),
  NA_Count = sapply(data_raw_na, function(x) sum(is.na(x))),
  Total_Obs = nrow(data_raw_na)
) %>%
  mutate(NA_Ratio = NA_Count / Total_Obs * 100) %>% # 缺失百分比
  arrange(desc(NA_Ratio)) # 按缺失占比从高到低排序

# 打印缺失值排序表格，控制台查看
cat("===== 各变量缺失值统计（从多到少排序）=====\n")
print(na_summary, row.names = FALSE)

# 2. 绘制缺失占比条形图（科研风格）
p_na_bar <- ggplot(na_summary, aes(x = reorder(Variable, NA_Ratio), y = NA_Ratio)) +
  geom_bar(stat = "identity", fill = "#c0392b", width = 0.7) +
  coord_flip() + # 横向条形，变量名不重叠
  labs(
    title = "所有变量的缺失值比例",
    x = "特征名称",
    y = "缺失比率 (%)"
  ) +
  theme_bw() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
    axis.title = element_text(size = 12, face = "bold"),
    axis.text.y = element_text(size = 9)
  )
print(p_na_bar)
ggsave(file.path(fig_dir, "所有变量缺失值比例.png"), p_na_bar, width = 12, height = 8)

# ==============================================
# 【修正：分层抽样】
# ==============================================
set.seed(123)  
# 场景1：抽取 100000 条样本
sample_num <- 100000

# 计算抽样比例
sample_ratio <- sample_num / nrow(data_clean)

# 分层抽样
sample_idx <- createDataPartition(data_clean$is_fraud, p = sample_ratio, list = FALSE)
data_sample <- data_clean[sample_idx, ]

# 查看抽样后数据维度
cat("抽样后数据维度：", dim(data_sample), "\n")

# ========== 修复点：只统计标签列 is_fraud，不再对整个数据集用table ==========
cat("抽样后样本分布（0=正常，1=欺诈）：\n")
freq_table <- table(data_sample$is_fraud)  # 仅指定标签列
print(freq_table)
cat("样本占比：\n")
print(prop.table(freq_table))

# ===================== 5. 标签转换 + 分类变量独热编码 =====================
data_sample$is_fraud <- factor(data_sample$is_fraud, levels = c(0,1), labels = c("正常","欺诈"))

dummy_obj <- dummyVars(~ . - is_fraud, data = data_sample)
data_encoded <- predict(dummy_obj, newdata = data_sample) %>% as.data.frame()
data_encoded$is_fraud <- data_sample$is_fraud

# ===================== 6. 分层划分训练集(70%)、测试集(30%) =====================
set.seed(123)
train_index <- createDataPartition(data_encoded$is_fraud, p = 0.7, list = FALSE)
train_raw <- data_encoded[train_index, ]
test_data <- data_encoded[-train_index, ]

# 仅统计标签列 is_fraud，不再对整个数据集使用table
cat("训练集样本分布：\n")
train_tab <- table(train_raw$is_fraud)
print(train_tab)
cat("训练集样本占比：\n")
print(prop.table(train_tab))

# ===================== 7. 计算类别权重（解决样本不平衡） =====================
n_normal <- sum(train_raw$is_fraud == "正常")
n_fraud  <- sum(train_raw$is_fraud == "欺诈")
class_weight <- n_normal / n_fraud
cat("欺诈样本权重：", class_weight, "\n")

# ===================== 8. 描述统计 & 可视化 =====================
library(ggplot2)
library(corrplot)

# 筛选所有数值型变量
num_cols <- select(data_sample, where(is.numeric))

# 按列计算统计量（MARGIN = 2 代表按列运算）
desc_stats <- data.frame(
  均值 = apply(num_cols, 2, mean, na.rm = TRUE),
  中位数 = apply(num_cols, 2, median, na.rm = TRUE),
  标准差 = apply(num_cols, 2, sd, na.rm = TRUE),
  最小值 = apply(num_cols, 2, min, na.rm = TRUE),
  最大值 = apply(num_cols, 2, max, na.rm = TRUE)
)

# 输出描述统计表
print("数值变量描述统计结果：")
print(desc_stats)

# -------------------- 标签柱状图--------------------
library(ggplot2)
library(scales)

# 1. 计算标签分布和比例
label_summary <- data_sample %>%
  count(is_fraud) %>%
  mutate(
    pct = n / sum(n),
    # 计算比例的置信区间（模拟误差线效果，也可替换为真实标准误）
    se = sqrt(pct * (1 - pct) / sum(n)),
    ci_low = pct - 1.96 * se,
    ci_high = pct + 1.96 * se,
    label_text = paste0(percent(pct, accuracy = 0.1))
  )

# 2. 绘制带误差线的科研柱状图
ggplot(label_summary, aes(x = is_fraud, y = pct, fill = is_fraud)) +
  # 主柱状图
  geom_col(
    width = 0.6,
    color = "black",  # 黑色边框，和示例风格一致
    linewidth = 0.8
  ) +
  # 误差线（置信区间）
  geom_errorbar(
    aes(ymin = ci_low, ymax = ci_high),
    width = 0.2,
    color = "black",
    linewidth = 0.8
  ) +
  # 顶部百分比标签
  geom_text(
    aes(label = label_text),
    vjust = -0.5,
    size = 3.5,
    fontface = "bold"
  ) +
  # 配色：参考示例的红蓝对比色
  scale_fill_manual(values = c("正常" = "#4682B4", "欺诈" = "#E64B35")) +
  # 坐标轴与标题
  labs(
    title = "交易样本标签分布",
    x = "交易类型",
    y = "占比 (%)"
  ) +
  scale_y_continuous(
    labels = percent_format(accuracy = 1),
    expand = c(0, 0.1)  # 给顶部标签留出空间
  ) +
  # 主题设置：带坐标轴的科研风格
  theme_classic() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
    axis.title = element_text(size = 11, face = "bold"),
    axis.text = element_text(size = 10),
    legend.position = "none",  # 直接用X轴标签，更简洁
    axis.line = element_line(color = "black", linewidth = 0.8),  # 黑色坐标轴
    axis.ticks = element_line(color = "black", linewidth = 0.8)  # 刻度线
  )

# 保存高清图（适合论文/报告）
ggsave(file.path(fig_dir, "交易样本标签分布.png"), width = 5, height = 4, dpi = 300)

# -------------------- 特征相关性热力图 --------------------
cor_matrix <- cor(num_cols, use = "complete.obs")

corrplot(
  cor_matrix,
  method = "color",
  type = "lower",
  tl.cex = 0.7,
  tl.srt = 0,
  tl.col = "black",
  title = "特征相关系数热力图",
  mar = c(0, 0, 2, 0),
  addCoef.col = NULL,
  # 示例同款红-橙-黄-浅蓝-深蓝渐变配色
  col = colorRampPalette(c("#d7191c", "#fdae61", "#ffffbf", "#abd9e9", "#2c7bb6"))(100)
)
dev.print(png, file.path(fig_dir, "特征相关系数热力图.png"), width = 1000, height = 800, res = 150)

# ===================== 9. 5折交叉验证统一配置 =====================
cv_ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = TRUE
)

# ===================== 前置：计算权重向量，不写入训练集 =====================
n_normal <- sum(train_raw$is_fraud == "正常")
n_fraud  <- sum(train_raw$is_fraud == "欺诈")
class_weight <- n_normal / n_fraud
cat("欺诈样本损失权重：", round(class_weight, 2), "\n")

# 单独生成权重向量，不加到train_raw里面
weight_vec <- ifelse(train_raw$is_fraud == "欺诈", class_weight, 1)

# ===================== 逻辑回归 召回优先调优（仅使用F2） =====================
set.seed(123)
glm_model <- train(
  is_fraud ~ .,
  data = train_raw,
  method = "glm",
  family = binomial,
  trControl = cv_ctrl,
  metric = "ROC",
  weights = weight_vec
)

# 预测概率
glm_prob_all <- predict(glm_model, test_data, type = "prob")
glm_prob <- glm_prob_all[, "欺诈"]

# 计算ROC坐标
library(pROC)
test_label_num <- ifelse(test_data$is_fraud == "欺诈", 1, 0)
roc_obj <- roc(response = test_label_num, predictor = glm_prob)
roc_coords <- coords(roc_obj, x = "all", ret = c("threshold", "sensitivity", "specificity"))

thresholds <- roc_coords$threshold
sens <- roc_coords$sensitivity
spec <- roc_coords$specificity

# 计算精确率、召回率、F2
total_fraud <- sum(test_label_num == 1)
total_normal <- sum(test_label_num == 0)
TP <- sens * total_fraud
FP <- (1 - spec) * total_normal
precision <- TP / (TP + FP)
precision[is.na(precision)] <- 0

beta <- 2
f2 <- (1 + beta^2) * (precision * sens) / ((beta^2 * precision) + sens)
f2[is.na(f2)] <- 0

# 直接选取 F2 最大的阈值（无任何约束）
best_idx <- which.max(f2)
final_thresh <- thresholds[best_idx]

cat("===== F2最优阈值（无误报率约束）=====\n")
cat("最优阈值：", round(final_thresh, 4), "\n")
cat("召回率：", round(sens[best_idx], 4), "\n")
cat("精确率：", round(precision[best_idx], 4), "\n")
cat("误报率：", round(1 - spec[best_idx], 4), "\n")
cat("F2分数：", round(f2[best_idx], 4), "\n")

# 生成最终预测
glm_pred_final <- factor(
  ifelse(glm_prob > final_thresh, "欺诈", "正常"),
  levels = c("正常", "欺诈")
)

glm_cm_final <- confusionMatrix(glm_pred_final, test_data$is_fraud, positive = "欺诈")
print(glm_cm_final)

# ===================== 逻辑回归 阈值-指标曲线 =====================
library(ggplot2)

# 准备数据
thresh_df_glm <- data.frame(
  Threshold = thresholds,        # 来自逻辑回归F2调优部分
  Recall = sens,
  Precision = precision,
  F2 = f2,
  FPR = 1 - spec
)
# 采样避免过密
thresh_df_glm <- thresh_df_glm[seq(1, nrow(thresh_df_glm), length.out = 200), ]

# 绘制
p_glm <- ggplot(thresh_df_glm, aes(x = Threshold)) +
  geom_line(aes(y = Recall, color = "Recall"), linewidth = 1) +
  geom_line(aes(y = Precision, color = "Precision"), linewidth = 1) +
  geom_line(aes(y = F2, color = "F2"), linewidth = 1, linetype = "dashed") +
  geom_vline(xintercept = final_thresh, linetype = "dotted", color = "black") +  # final_thresh 来自逻辑回归
  scale_color_manual(values = c("Recall" = "#2E86AB", "Precision" = "#A23B72", "F2" = "#F18F01")) +
  labs(title = "逻辑回归 阈值-指标曲线（召回优先）",
       x = "概率阈值", y = "指标值", color = "指标") +
  theme_bw() +
  theme(legend.position = "bottom")

print(p_glm)
ggsave(file.path(fig_dir, "逻辑回归_阈值调优曲线.png"), p_glm, width = 8, height = 5, dpi = 300)

# ===================== 11. 模型2：径向核SVM（召回优先 + F2调优） =====================
set.seed(123)
# 构造权重向量：正常样本权重=1，欺诈样本权重=class_weight
w_vec <- ifelse(train_raw$is_fraud == "欺诈", class_weight, 1)

svm_model <- train(
  is_fraud ~ .,
  data = train_raw,
  method = "svmRadial",
  trControl = cv_ctrl,
  metric = "ROC",
  weights = w_vec
)

# 预测概率（注意：caret的svmRadial返回概率矩阵，列名对应类别）
svm_prob_all <- predict(svm_model, test_data, type = "prob")
svm_prob <- svm_prob_all[, "欺诈"]  # 正类概率

# ---------------------- F2最优阈值选择 ----------------------
library(pROC)
test_label_num <- ifelse(test_data$is_fraud == "欺诈", 1, 0)

roc_obj <- roc(response = test_label_num, predictor = svm_prob)
roc_coords <- coords(roc_obj, x = "all", ret = c("threshold", "sensitivity", "specificity"))
thresholds <- roc_coords$threshold
sens <- roc_coords$sensitivity
spec <- roc_coords$specificity

# 计算精确率、F2
total_fraud <- sum(test_label_num == 1)
total_normal <- sum(test_label_num == 0)
TP <- sens * total_fraud
FP <- (1 - spec) * total_normal
precision <- TP / (TP + FP)
precision[is.na(precision)] <- 0

beta <- 2
f2 <- (1 + beta^2) * (precision * sens) / ((beta^2 * precision) + sens)
f2[is.na(f2)] <- 0

best_idx <- which.max(f2)
final_thresh <- thresholds[best_idx]

cat("\n===== SVM F2最优阈值结果 =====\n")
cat("最优阈值：", round(final_thresh, 4), "\n")
cat("召回率 (Sensitivity)：", round(sens[best_idx], 4), "\n")
cat("精确率 (Precision)：", round(precision[best_idx], 4), "\n")
cat("误报率 (1-Specificity)：", round(1 - spec[best_idx], 4), "\n")
cat("F2分数：", round(f2[best_idx], 4), "\n")

# 生成最终预测
svm_pred_final <- factor(
  ifelse(svm_prob > final_thresh, "欺诈", "正常"),
  levels = c("正常", "欺诈")
)

svm_cm_final <- confusionMatrix(svm_pred_final, test_data$is_fraud, positive = "欺诈")
cat("\n===== SVM 召回优先优化后评估结果 =====\n")
print(svm_cm_final)

# 可选：绘制阈值-指标曲线（便于论文展示）
library(ggplot2)
thresh_df <- data.frame(
  Threshold = thresholds,
  Recall = sens,
  Precision = precision,
  F2 = f2,
  FPR = 1 - spec
)
# 采样避免过密
thresh_df <- thresh_df[seq(1, nrow(thresh_df), length.out = 200), ]

p_svm <- ggplot(thresh_df, aes(x = Threshold)) +
  geom_line(aes(y = Recall, color = "Recall"), linewidth = 1) +
  geom_line(aes(y = Precision, color = "Precision"), linewidth = 1) +
  geom_line(aes(y = F2, color = "F2"), linewidth = 1, linetype = "dashed") +
  geom_vline(xintercept = final_thresh, linetype = "dotted", color = "black") +
  scale_color_manual(values = c("Recall" = "#2E86AB", "Precision" = "#A23B72", "F2" = "#F18F01")) +
  labs(title = "SVM 阈值-指标曲线（召回优先）",
       x = "概率阈值", y = "指标值", color = "指标") +
  theme_bw() +
  theme(legend.position = "bottom")
print(p_svm)
ggsave(file.path(fig_dir, "SVM_阈值调优曲线.png"), p_svm, width = 8, height = 5, dpi = 300)

# ===================== 12. 模型3：随机森林（召回优先 + F2调优） =====================
library(ranger)
library(caret)
library(pROC)
library(dplyr)
library(ggplot2)

# 1. 构造样本权重
w_vec <- ifelse(train_raw$is_fraud == "欺诈", class_weight, 1)
cv_ctrl <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary)

# 2. ranger随机森林训练
set.seed(123)
rf_model <- train(
  is_fraud ~ .,
  data = train_raw,
  method = "ranger",
  ntree = 100,
  trControl = cv_ctrl,
  metric = "ROC",
  weights = w_vec,
  tuneGrid = expand.grid(
    mtry = c(5,10,15),
    splitrule = "gini",
    min.node.size = c(5,10)
  ),
  importance = "impurity"
)

# 预测概率
rf_prob <- predict(rf_model, newdata = test_data, type = "prob")
fraud_prob <- rf_prob[, "欺诈"]

# ---------------------- F2最优阈值选择 ----------------------
test_label_num <- ifelse(test_data$is_fraud == "欺诈", 1, 0)

roc_rf <- roc(response = test_label_num, predictor = fraud_prob)
roc_data <- coords(roc_rf, x = "all", ret = c("threshold", "sensitivity", "specificity"))
thresholds <- roc_data$threshold
sens <- roc_data$sensitivity
spec <- roc_data$specificity

# 计算精确率、F2
total_fraud <- sum(test_label_num == 1)
total_normal <- sum(test_label_num == 0)
TP <- sens * total_fraud
FP <- (1 - spec) * total_normal
precision <- TP / (TP + FP)
precision[is.na(precision)] <- 0

beta <- 2
f2 <- (1 + beta^2) * (precision * sens) / ((beta^2 * precision) + sens)
f2[is.na(f2)] <- 0

best_idx <- which.max(f2)
best_thresh <- thresholds[best_idx]

cat("\n===== 随机森林 F2最优阈值结果 =====\n")
cat("最优阈值：", round(best_thresh, 4), "\n")
cat("召回率 (Sensitivity)：", round(sens[best_idx], 4), "\n")
cat("精确率 (Precision)：", round(precision[best_idx], 4), "\n")
cat("误报率 (1-Specificity)：", round(1 - spec[best_idx], 4), "\n")
cat("F2分数：", round(f2[best_idx], 4), "\n")

# 生成预测标签
rf_pred <- factor(
  ifelse(fraud_prob > best_thresh, "欺诈", "正常"),
  levels = c("正常","欺诈")
)

# 混淆矩阵
rf_cm <- confusionMatrix(rf_pred, test_data$is_fraud, positive = "欺诈")
cat("\n===== 随机森林 召回优先优化后评估结果 =====\n")
print(rf_cm)

# ==================== 特征重要性绘图 ====================
library(ggplot2)
library(dplyr)

imp_obj <- varImp(rf_model)$importance
imp_df <- data.frame(
  Feature = rownames(imp_obj),
  Importance = imp_obj[,1]
)

# 改为head(21)，正好对应你清洗完毕保留的21个变量
imp_top <- imp_df %>%
  arrange(desc(Importance)) %>%
  head(21) %>%
  mutate(Feature = factor(Feature, levels = rev(Feature)))

ggplot(imp_top, aes(x = Importance, y = Feature)) +
  geom_bar(stat = "identity", fill = "#2c5282", width = 0.7) +
  labs(
    title = "Feature Importance Ranking (Random Forest)",
    x = "Importance Score",
    y = "Feature Name"
  ) +
  theme_bw() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
    axis.title.x = element_text(size = 12, face = "bold"),
    axis.title.y = element_text(size = 12, face = "bold"),
    axis.text.y = element_text(size = 9),
    axis.text.x = element_text(size = 10),
    panel.grid.major.y = element_blank(),
    panel.border = element_rect(linewidth = 1)
  )

# ===================== 随机森林 阈值-指标曲线 =====================
library(ggplot2)

# 使用当前模型的阈值和指标
thresh_df_rf <- data.frame(
  Threshold = thresholds,          
  Recall = sens,
  Precision = precision,
  F2 = f2,
  FPR = 1 - spec
)
thresh_df_rf <- thresh_df_rf[seq(1, nrow(thresh_df_rf), length.out = 200), ]

p_rf <- ggplot(thresh_df_rf, aes(x = Threshold)) +
  geom_line(aes(y = Recall, color = "Recall"), linewidth = 1) +
  geom_line(aes(y = Precision, color = "Precision"), linewidth = 1) +
  geom_line(aes(y = F2, color = "F2"), linewidth = 1, linetype = "dashed") +
  geom_vline(xintercept = best_thresh, linetype = "dotted", color = "black") +  # 注意变量名
  scale_color_manual(values = c("Recall" = "#2E86AB", "Precision" = "#A23B72", "F2" = "#F18F01")) +
  labs(title = "随机森林 阈值-指标曲线（召回优先）",
       x = "概率阈值", y = "指标值", color = "指标") +
  theme_bw() +
  theme(legend.position = "bottom")

print(p_rf)
ggsave(file.path(fig_dir, "随机森林_阈值调优曲线.png"), p_rf, width = 8, height = 5, dpi = 300)

# ===================== 13. 模型4：XGBoost（召回优先 + F2调优） =====================
library(xgboost)
library(pROC)

# 1. 构建特征矩阵和标签
train_x <- model.matrix(is_fraud ~ . -1, data = train_raw)
train_y <- ifelse(train_raw$is_fraud == "欺诈", 1, 0)
train_y <- as.integer(train_y)  

test_x  <- model.matrix(is_fraud ~ . -1, data = test_data)
test_y  <- ifelse(test_data$is_fraud == "欺诈", 1, 0)
test_y  <- as.integer(test_y)

dtrain <- xgb.DMatrix(data = train_x, label = train_y)
dtest  <- xgb.DMatrix(data = test_x, label = test_y)

# 2. 超参数配置
set.seed(123)
params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  scale_pos_weight = class_weight,
  learning_rate = 0.03,
  max_depth = 8,
  min_child_weight = 3,
  reg_lambda = 1,
  reg_alpha = 0.2,
  subsample = 0.8,
  colsample_bytree = 0.8,
  max_delta_step = 2
)

# 3. 5折交叉验证+早停
cv_result <- xgb.cv(
  params = params,
  data = dtrain,
  nrounds = 800,
  nfold = 5,
  early_stopping_rounds = 20,
  verbose = 0
)

# 4. 稳健获取最佳迭代轮数
if (!is.null(cv_result$best_iteration)) {
  best_round <- cv_result$best_iteration
} else if (!is.null(cv_result$best_ntree_limit)) {
  best_round <- cv_result$best_ntree_limit
} else if (!is.null(cv_result$evaluation_log)) {
  # 从评估日志中取验证集AUC最大的轮数
  best_round <- which.max(cv_result$evaluation_log$test_auc_mean)
} else {
  best_round <- 100  # 保底值（可根据实际调整）
}
cat("XGB最优迭代轮数：", best_round, "\n")

# 5. 训练模型
xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = best_round
)

# 6. 预测概率
xgb_prob <- predict(object = xgb_model, newdata = test_x)

# ---------------------- F2最优阈值选择 ----------------------
test_label_num <- ifelse(test_data$is_fraud == "欺诈", 1, 0)

roc_xgb <- roc(response = test_label_num, predictor = xgb_prob)
roc_data <- coords(roc_xgb, x = "all", ret = c("threshold", "sensitivity", "specificity"))
thresholds <- roc_data$threshold
sens <- roc_data$sensitivity
spec <- roc_data$specificity

total_fraud <- sum(test_label_num == 1)
total_normal <- sum(test_label_num == 0)
TP <- sens * total_fraud
FP <- (1 - spec) * total_normal
precision <- TP / (TP + FP)
precision[is.na(precision)] <- 0

beta <- 2
f2 <- (1 + beta^2) * (precision * sens) / ((beta^2 * precision) + sens)
f2[is.na(f2)] <- 0

best_idx <- which.max(f2)
best_thresh <- thresholds[best_idx]

cat("\n===== XGBoost F2最优阈值结果 =====\n")
cat("最优阈值：", round(best_thresh, 4), "\n")
cat("召回率 (Sensitivity)：", round(sens[best_idx], 4), "\n")
cat("精确率 (Precision)：", round(precision[best_idx], 4), "\n")
cat("误报率 (1-Specificity)：", round(1 - spec[best_idx], 4), "\n")
cat("F2分数：", round(f2[best_idx], 4), "\n")

# 生成预测标签
xgb_pred <- factor(
  ifelse(xgb_prob > best_thresh, "欺诈", "正常"),
  levels = c("正常", "欺诈")
)

# 混淆矩阵
xgb_cm <- confusionMatrix(xgb_pred, test_data$is_fraud, positive = "欺诈")
cat("\n===== XGBoost 召回优先优化后评估结果 =====\n")
print(xgb_cm)

# ===================== XGBoost 阈值-指标曲线 =====================
library(ggplot2)

# 准备数据（使用当前模型的阈值和指标）
thresh_df_xgb <- data.frame(
  Threshold = thresholds,
  Recall = sens,
  Precision = precision,
  F2 = f2,
  FPR = 1 - spec
)
# 采样避免过密
thresh_df_xgb <- thresh_df_xgb[seq(1, nrow(thresh_df_xgb), length.out = 200), ]

p_xgb <- ggplot(thresh_df_xgb, aes(x = Threshold)) +
  geom_line(aes(y = Recall, color = "Recall"), linewidth = 1) +
  geom_line(aes(y = Precision, color = "Precision"), linewidth = 1) +
  geom_line(aes(y = F2, color = "F2"), linewidth = 1, linetype = "dashed") +
  geom_vline(xintercept = best_thresh, linetype = "dotted", color = "black") +
  scale_color_manual(values = c("Recall" = "#2E86AB", "Precision" = "#A23B72", "F2" = "#F18F01")) +
  labs(title = "XGBoost 阈值-指标曲线（召回优先）",
       x = "概率阈值", y = "指标值", color = "指标") +
  theme_bw() +
  theme(legend.position = "bottom")

print(p_xgb)
ggsave(file.path(fig_dir, "XGBoost_阈值调优曲线.png"), p_xgb, width = 8, height = 5, dpi = 300)

# 特征重要性（保存图片）
xgb_imp <- xgb.importance(model = xgb_model)
xgb.plot.importance(xgb_imp, top_n = 10, main = "XGBoost Top10 Feature Importance")
ggsave(file.path(fig_dir, "XGBoost_特征重要性.png"), width = 10, height = 6, dpi = 300)

# ===================== 14. 模型5：DNN深度网络（召回优先 + F2调优） =====================
library(keras)
library(reticulate)
library(pROC)
library(dplyr)

# 1. 分层过采样训练集，平衡样本分布
set.seed(123)
train_fraud <- train_raw[train_raw$is_fraud == "欺诈", ]
train_normal <- train_raw[train_raw$is_fraud == "正常", ]
sample_num <- floor(nrow(train_normal) / 4)
fraud_oversample <- slice_sample(train_fraud, n = sample_num, replace = TRUE)
train_balanced <- rbind(train_normal, fraud_oversample)

# 构建特征矩阵与标签（使用平衡后的训练集）
train_x <- model.matrix(is_fraud ~ . -1, data = train_balanced)
train_y <- ifelse(train_balanced$is_fraud == "欺诈", 1, 0)
train_y <- as.integer(train_y)

test_x  <- model.matrix(is_fraud ~ . -1, data = test_data)
test_y  <- ifelse(test_data$is_fraud == "欺诈", 1, 0)
test_y  <- as.integer(test_y)

# 特征标准化
prep_obj <- preProcess(train_x, method = c("center", "scale"))
train_x_scaled <- predict(prep_obj, newdata = train_x)
test_x_scaled  <- predict(prep_obj, newdata = test_x)

# 转为矩阵传入Python
x_train <- as.matrix(train_x_scaled)
y_train <- as.array(train_y)
x_test  <- as.matrix(test_x_scaled)

# 特征维度、不平衡权重
n_feat <- ncol(x_train)
pos_weight <- class_weight

# 关闭TensorFlow日志
Sys.setenv(TF_ENABLE_ONEDNN_OPTS = "0")

# 2. Keras网络代码（保持不变）
py_run_string('
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

x_train = r.x_train
y_train = r.y_train
x_test  = r.x_test
n_feat  = r.n_feat
pos_w = r.pos_weight

model = Sequential()
model.add(Dense(256, activation="relu", input_shape=(n_feat,)))
model.add(BatchNormalization())
model.add(Dropout(0.15))
model.add(Dense(128, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(64, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(1, activation="sigmoid"))

opt = tf.keras.optimizers.Adam(learning_rate=0.0003)

def focal_loss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        cross_ent = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        modulating_factor = (1.0 - p_t) ** gamma
        alpha_weight = y_true * alpha + (1 - y_true) * (1 - alpha)
        focal_cross_ent = alpha_weight * modulating_factor * cross_ent
        return tf.reduce_mean(focal_cross_ent)
    return loss

model.compile(loss=focal_loss(alpha=0.25, gamma=2.0), optimizer=opt, metrics=["auc"])

early_stop = EarlyStopping(
    monitor="val_auc",
    patience=12,
    mode="max",
    restore_best_weights=True
)

class_weights = {0: 1.0, 1: pos_w}

history = model.fit(
    x_train, y_train,
    epochs=50,
    batch_size=8,
    validation_split=0.2,
    class_weight=class_weights,
    callbacks=[early_stop],
    verbose=1
)

y_pred_prob = model.predict(x_test, verbose=0)
r.dnn_prob = y_pred_prob.ravel()
')

# 3. R端 F2最优阈值选择（替换原来的F1）
test_label_num <- ifelse(test_data$is_fraud == "欺诈", 1, 0)

roc_obj <- roc(response = test_label_num, predictor = dnn_prob)
roc_data <- coords(roc_obj, x = "all", ret = c("threshold", "sensitivity", "specificity"))
thresholds <- roc_data$threshold
sens <- roc_data$sensitivity
spec <- roc_data$specificity

total_fraud <- sum(test_label_num == 1)
total_normal <- sum(test_label_num == 0)
TP <- sens * total_fraud
FP <- (1 - spec) * total_normal
precision <- TP / (TP + FP)
precision[is.na(precision)] <- 0

beta <- 2
f2 <- (1 + beta^2) * (precision * sens) / ((beta^2 * precision) + sens)
f2[is.na(f2)] <- 0

best_idx <- which.max(f2)
best_threshold <- thresholds[best_idx]

cat("\n===== DNN F2最优阈值结果 =====\n")
cat("最优阈值：", round(best_threshold, 4), "\n")
cat("召回率 (Sensitivity)：", round(sens[best_idx], 4), "\n")
cat("精确率 (Precision)：", round(precision[best_idx], 4), "\n")
cat("误报率 (1-Specificity)：", round(1 - spec[best_idx], 4), "\n")
cat("F2分数：", round(f2[best_idx], 4), "\n")

# 生成预测标签
dnn_pred <- factor(
  ifelse(dnn_prob > best_threshold, "欺诈", "正常"),
  levels = c("正常","欺诈")
)

# 混淆矩阵输出
dnn_cm <- confusionMatrix(dnn_pred, test_data$is_fraud, positive = "欺诈")
cat("\n===== DNN深度网络 召回优先优化后评估结果 =====\n")
print(dnn_cm)

# ===================== DNN 阈值-指标曲线 =====================
library(ggplot2)

# 准备数据（使用DNN的阈值和指标）
thresh_df_dnn <- data.frame(
  Threshold = thresholds,
  Recall = sens,
  Precision = precision,
  F2 = f2,
  FPR = 1 - spec
)
# 采样避免过密
thresh_df_dnn <- thresh_df_dnn[seq(1, nrow(thresh_df_dnn), length.out = 200), ]

p_dnn <- ggplot(thresh_df_dnn, aes(x = Threshold)) +
  geom_line(aes(y = Recall, color = "Recall"), linewidth = 1) +
  geom_line(aes(y = Precision, color = "Precision"), linewidth = 1) +
  geom_line(aes(y = F2, color = "F2"), linewidth = 1, linetype = "dashed") +
  geom_vline(xintercept = best_threshold, linetype = "dotted", color = "black") +
  scale_color_manual(values = c("Recall" = "#2E86AB", "Precision" = "#A23B72", "F2" = "#F18F01")) +
  labs(title = "DNN 阈值-指标曲线（召回优先）",
       x = "概率阈值", y = "指标值", color = "指标") +
  theme_bw() +
  theme(legend.position = "bottom")

print(p_dnn)
ggsave(file.path(fig_dir, "DNN_阈值调优曲线.png"), p_dnn, width = 8, height = 5, dpi = 300)

# ===================== 15. 全模型指标汇总（含DNN，使用F2调优后结果） =====================
library(pROC)

# 定义提取函数
get_metrics <- function(cm, prob_vec, true_label){
  acc <- cm$overall["Accuracy"]
  prec <- cm$byClass["Precision"]
  recall <- cm$byClass["Recall"]
  f1 <- cm$byClass["F1"]
  
  # 提取正类概率（确保是数值向量）
  if(is.data.frame(prob_vec) || is.matrix(prob_vec)) {
    p <- as.numeric(prob_vec[, "欺诈"])
  } else {
    p <- as.numeric(prob_vec)
  }
  
  # 计算AUC
  auc_val <- suppressMessages(roc(true_label, p)$auc)
  
  return(data.frame(
    准确率 = round(acc, 4),
    精确率 = round(prec, 4),
    召回率 = round(recall, 4),
    F1 = round(f1, 4),
    AUC = round(auc_val, 4)
  ))
}

# 标签（测试集真实标签）
true_label <- as.numeric(ifelse(test_data$is_fraud == "欺诈", 1, 0))

# 汇总各模型（使用F2调优后的变量）
results <- rbind(
  逻辑回归 = get_metrics(glm_cm_final, glm_prob, true_label),
  SVM = get_metrics(svm_cm_final, svm_prob, true_label),
  随机森林 = get_metrics(rf_cm, fraud_prob, true_label),
  XGBoost = get_metrics(xgb_cm, xgb_prob, true_label),
  DNN = get_metrics(dnn_cm, dnn_prob, true_label)
)

cat("\n========== 全模型评估指标汇总（召回优先F2调优） ==========\n")
print(results)
write.csv(results, "模型对比汇总结果_F2调优_含DNN.csv", row.names = TRUE)

# ===================== 16. ROC曲线绘制（含DNN） =====================
roc_glm <- roc(true_label, as.numeric(glm_prob))
roc_svm <- roc(true_label, as.numeric(svm_prob))
roc_rf  <- roc(true_label, as.numeric(fraud_prob))
roc_xgb <- roc(true_label, as.numeric(xgb_prob))
roc_dnn <- roc(true_label, as.numeric(dnn_prob))

# 绘制对比曲线
plot(roc_glm, col = "#1f77b4", main = "各模型ROC曲线对比（召回优先调优）", 
     legacy.axes = TRUE, lwd = 2)
plot(roc_svm, col = "#ff7f0e", add = TRUE, lwd = 2)
plot(roc_rf,  col = "#2ca02c", add = TRUE, lwd = 2)
plot(roc_xgb, col = "#d62728", add = TRUE, lwd = 2)
plot(roc_dnn, col = "#9467bd", add = TRUE, lwd = 2)

# 图例（带AUC）
legend("bottomright",
       legend = c(
         paste0("逻辑回归 (AUC=", round(auc(roc_glm), 3), ")"),
         paste0("SVM (AUC=", round(auc(roc_svm), 3), ")"),
         paste0("随机森林 (AUC=", round(auc(roc_rf), 3), ")"),
         paste0("XGBoost (AUC=", round(auc(roc_xgb), 3), ")"),
         paste0("DNN (AUC=", round(auc(roc_dnn), 3), ")")
       ),
       col = c("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"),
       lwd = 2, cex = 0.8)

