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
# 数值型变量：中位数填充
data_clean <- data_clean %>%
  mutate(across(where(is.numeric), ~ ifelse(is.na(.), median(., na.rm = TRUE), .)))

# 字符型变量：众数填充
fill_mode <- function(vec){
  if(!any(is.na(vec))) return(vec)
  tbl <- table(vec)
  mode_val <- names(tbl)[which.max(tbl)]
  vec[is.na(vec)] <- mode_val
  return(vec)
}
data_clean <- data_clean %>%
  mutate(across(where(is.character), fill_mode))

# ==============================================
# 【修正：分层抽样】
# ==============================================
set.seed(123)  # 固定随机种子，结果可复现
# 场景1：抽取 100000 条样本（调试用）
sample_num <- 100000

# 计算抽样比例
sample_ratio <- sample_num / nrow(data_clean)

# 分层抽样：按is_fraud分层，保证欺诈/正常比例不变
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
# 假设结果 class_weight ≈ 17.29，含义：
# 1 条欺诈样本产生的损失 = 17.29 条正常样本的损失
# 模型犯错（把欺诈判成正常）的惩罚力度被放大十几倍


# 直观理解:
# 模型错判1 个欺诈样本 → 损失暴增十几倍
# 为了降低总损失，模型必须努力学好欺诈样本的特征
# 最终提升欺诈识别能力（召回率），这正是风控 / 反欺诈的核心需求
# 风控场景优先级：宁可误报正常交易，也不能漏掉欺诈交易，所以必须做加权。

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

# -------------------- 标签分布科研柱状图（带坐标轴/误差线） --------------------
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

# -------------------- 特征相关性热力图（示例同款配色版） --------------------
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

# ===================== 10. 模型1：逻辑回归 =====================
set.seed(123)
glm_model <- train(
  is_fraud ~ ., # 公式含义：以 is_fraud（交易类型：正常 / 欺诈）为目标变量，使用数据集中所有其余列作为特征建模。
  data = train_raw, # 使用你划分好的训练集建模，只用训练数据学习规律，不碰测试集。
  method = "glm", # 指定模型：使用广义线性模型，这里就是逻辑回归。
  family = binomial, # 告诉 R：这是二分类任务（结果只有两类：正常、欺诈），搭配 glm 就是标准逻辑回归。
  trControl = cv_ctrl, # 你提前定义的交叉验证规则（常见 k 折交叉验证）。
  # 求什么：
  # 把训练集拆成 k 份子集，轮流用子集当验证集、其余当训练集；
  # 多次训练取平均效果，避免过拟合、客观评估模型泛化能力。
  metric = "ROC", 
  # 模型训练时，以 ROC 曲线下面积 (AUC) 作为优化和筛选标准。
  # 针对不平衡数据（你欺诈样本极少），AUC 比单纯准确率更靠谱。
  weights = ifelse(train_raw$is_fraud == "欺诈", class_weight, 1)
) # 如果样本标签是 欺诈 → 权重 = class_weight（≈17.29）
# 如果样本标签是 正常 → 权重 = 1

glm_pred <- predict(glm_model, test_data)
glm_prob <- predict(glm_model, test_data, type = "prob")
glm_cm <- confusionMatrix(glm_pred, test_data$is_fraud, positive = "欺诈")
cat("\n===== 逻辑回归 评估结果 =====\n")
print(glm_cm)

# ===================== 11. 模型2：径向核SVM =====================
set.seed(123)
# 构造权重向量：正常样本权重=1，欺诈样本权重=class_weight
w_vec <- ifelse(train_raw$is_fraud == "欺诈", class_weight, 1)

svm_model <- train(
  is_fraud ~ .,
  data = train_raw,
  method = "svmRadial",
  trControl = cv_ctrl,
  metric = "ROC",
  weights = w_vec   # 传入构造好的数值权重向量
)

svm_pred <- predict(svm_model, test_data)
svm_prob <- predict(svm_model, type = "prob")
svm_cm <- confusionMatrix(svm_pred, test_data$is_fraud, positive = "欺诈")
cat("\n===== SVM 评估结果 =====\n")
print(svm_cm)

# ===================== 12. 模型3：随机森林 =====================
set.seed(123)
# 单独构造权重向量
w_vec <- ifelse(train_raw$is_fraud == "欺诈", class_weight, 1)
rf_model <- train(
  is_fraud ~ .,
  data = train_raw,
  method = "rf",
  ntree = 100,
  trControl = cv_ctrl,
  metric = "ROC",
  weights = w_vec
)

# 修正：必须指定 newdata = test_data
rf_pred <- predict(rf_model, newdata = test_data)
rf_prob <- predict(rf_model, newdata = test_data, type = "prob")

# 后续评估不变
rf_cm <- confusionMatrix(rf_pred, test_data$is_fraud, positive = "欺诈")
cat("\n===== 随机森林 评估结果 =====\n")
print(rf_cm)

# 特征重要性
varImp(rf_model)
plot(varImp(rf_model), main="随机森林特征重要性排序")

# ===================== 13. 模型4：XGBoost =====================
# 1. 构建特征矩阵和标签（保留0/1整数类型）
train_x <- model.matrix(is_fraud ~ . -1, data = train_raw)
train_y <- ifelse(train_raw$is_fraud == "欺诈", 1, 0)
train_y <- as.integer(train_y)  # 确保是整数类型

test_x  <- model.matrix(is_fraud ~ . -1, data = test_data)
test_y  <- ifelse(test_data$is_fraud == "欺诈", 1, 0)
test_y  <- as.integer(test_y)

# 2. 新版xgboost参数设置（解决警告+报错）
set.seed(123)
params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  scale_pos_weight = class_weight
)

# 新版xgboost使用 x 和 y 作为参数名，而非 data 和 label
xgb_model <- xgboost(
  x = train_x,
  y = train_y,
  params = params,
  nrounds = 100
)

# 预测
xgb_prob <- predict(xgb_model, test_x)
xgb_pred <- factor(ifelse(xgb_prob > 0.5, "欺诈", "正常"), levels = c("正常", "欺诈"))
xgb_cm <- confusionMatrix(xgb_pred, test_data$is_fraud, positive = "欺诈")
cat("\n===== XGBoost 评估结果 =====\n")
print(xgb_cm)

# 特征重要性
xgb_imp <- xgb.importance(model = xgb_model)
xgb.plot.importance(xgb_imp, top_n = 10, main = "XGBoost Top10 重要特征")

# ===================== 14. 模型5：DNN深度网络 =====================
library(keras)
library(reticulate)

# 1. 特征标准化
prep_obj <- preProcess(train_x, method = c("center", "scale"))
train_x_scaled <- predict(prep_obj, newdata = train_x)
test_x_scaled  <- predict(prep_obj, newdata = test_x)

# 转为矩阵，供Python调用
x_train <- as.matrix(train_x_scaled)
y_train <- as.array(train_y)
x_test  <- as.matrix(test_x_scaled)

# 特征维度（R变量，Python直接读取）
n_feat <- ncol(x_train)

# 关闭TensorFlow冗余日志
Sys.setenv(TF_ENABLE_ONEDNN_OPTS = "0")

# 2. 运行原生Python Keras代码（不再额外传参，直接读R环境变量）
py_run_string('
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# 读取R中的变量
x_train = r.x_train
y_train = r.y_train
x_test  = r.x_test
n_feat  = r.n_feat

# 搭建模型
model = Sequential()
model.add(Dense(64, activation="relu", input_shape=(n_feat,)))
model.add(Dropout(0.3))
model.add(Dense(32, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(1, activation="sigmoid"))

# 编译
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# 训练
model.fit(
    x_train, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# 预测并传回R
y_pred_prob = model.predict(x_test)
r.dnn_prob = y_pred_prob.ravel()
')

# 3. R端生成标签、评估模型
dnn_pred <- factor(ifelse(dnn_prob > 0.5, "欺诈", "正常"), levels = c("正常","欺诈"))
dnn_cm <- confusionMatrix(dnn_pred, test_data$is_fraud, positive = "欺诈")
cat("\n===== DNN深度网络 评估结果 =====\n")
print(dnn_cm)

# ===================== 15. 模型指标汇总 =====================
library(pROC)

get_metrics <- function(cm, prob, true_label){
  acc <- cm$overall["Accuracy"]
  prec <- cm$byClass["Precision"]
  recall <- cm$byClass["Recall"]
  f1 <- cm$byClass["F1"]
  
  # 统一转为数据框，提取正类概率并转数值向量
  prob_df <- as.data.frame(prob)
  if(ncol(prob_df) >= 2){
    prob_vec <- as.numeric(prob_df[, 2])
  } else {
    prob_vec <- as.numeric(prob_df[, 1])
  }
  
  # 强制转为一维向量 + 对齐长度（核心修复）
  prob_vec <- as.vector(prob_vec)
  true_vec <- as.vector(true_label)
  
  # 截断到相同长度，避免行列数不一致
  min_len <- min(length(prob_vec), length(true_vec))
  prob_vec <- prob_vec[1:min_len]
  true_vec <- true_vec[1:min_len]
  
  # 计算AUC
  auc <- suppressMessages(roc(true_vec, prob_vec)$auc)
  
  return(data.frame(准确率=acc, 精确率=prec, 召回率=recall, F1=f1, AUC=auc))
}

true_test_label <- as.vector(ifelse(test_data$is_fraud == "欺诈", 1, 0))

# 逐个计算指标
res_glm <- get_metrics(glm_cm, glm_prob, true_test_label)
res_svm <- get_metrics(svm_cm, svm_prob, true_test_label)
res_rf  <- get_metrics(rf_cm,  rf_prob,  true_test_label)

# XGBoost
res_xgb <- data.frame(
  准确率 = xgb_cm$overall["Accuracy"],
  精确率 = xgb_cm$byClass["Precision"],
  召回率 = xgb_cm$byClass["Recall"],
  F1 = xgb_cm$byClass["F1"],
  AUC = suppressMessages(roc(true_test_label, as.vector(as.numeric(xgb_prob)))$auc)
)

# DNN
res_dnn <- data.frame(
  准确率 = dnn_cm$overall["Accuracy"],
  精确率 = dnn_cm$byClass["Precision"],
  召回率 = dnn_cm$byClass["Recall"],
  F1 = dnn_cm$byClass["F1"],
  AUC = suppressMessages(roc(true_test_label, as.vector(as.numeric(dnn_prob)))$auc)
)

# 合并结果并导出
result_total <- rbind(
  逻辑回归 = res_glm,
  SVM = res_svm,
  随机森林 = res_rf,
  XGBoost = res_xgb,
  DNN深度网络 = res_dnn
)
cat("\n======== 全模型评估指标汇总 ========\n")
print(result_total)
write.csv(result_total, "模型对比汇总结果.csv", row.names = TRUE)

# ===================== 16. ROC曲线绘制 =====================
library(pROC)

# 1. 封装函数：提取概率 + 强制转向量 + 对齐长度
get_pos_prob <- function(prob, label_vec){
  p_df <- as.data.frame(prob)
  # 提取正类概率
  if(ncol(p_df) >= 2){
    p <- as.numeric(p_df[, 2])
  }else{
    p <- as.numeric(p_df[, 1])
  }
  p <- as.vector(p)
  # 按标签长度截断，保证二者等长
  len <- length(label_vec)
  p <- p[1:len]
  return(p)
}

# 标签统一转为一维向量
true_label_vec <- as.vector(true_test_label)

# 提取并对齐所有模型概率
p_glm  <- get_pos_prob(glm_prob, true_label_vec)
p_svm  <- get_pos_prob(svm_prob, true_label_vec)
p_rf   <- get_pos_prob(rf_prob,  true_label_vec)
p_xgb  <- get_pos_prob(xgb_prob, true_label_vec)
p_dnn  <- get_pos_prob(dnn_prob, true_label_vec)

# 计算ROC，屏蔽提示
roc_glm <- suppressMessages(roc(true_label_vec, p_glm))
roc_svm <- suppressMessages(roc(true_label_vec, p_svm))
roc_rf  <- suppressMessages(roc(true_label_vec, p_rf))
roc_xgb <- suppressMessages(roc(true_label_vec, p_xgb))
roc_dnn <- suppressMessages(roc(true_label_vec, p_dnn))

# 绘制对比曲线
plot(roc_glm, col = "blue", main = "各模型ROC曲线对比", legacy.axes = TRUE, lwd = 2)
plot(roc_svm, col = "red", add = TRUE, lwd = 2)
plot(roc_rf,  col = "green", add = TRUE, lwd = 2)
plot(roc_xgb, col = "purple", add = TRUE, lwd = 2)
plot(roc_dnn, col = "orange", add = TRUE, lwd = 2)

legend("bottomright",
       legend = c("逻辑回归","SVM","随机森林","XGBoost","DNN"),
       col = c("blue","red","green","purple","orange"),
       lwd = 2)

