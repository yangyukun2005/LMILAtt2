import json
import re
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
from modelscope import AutoModel, AutoTokenizer
import gc
import os
import time
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# 检查并设置GPU设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"使用设备: {device}")
if device.type == 'cuda':
    logger.info(f"GPU名称: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

# 1. 数据加载函数
def load_user_level_data(depressed_path, normal_path):
    logger.info("开始加载用户数据...")
    
    # 加载抑郁用户数据
    with open(depressed_path, 'r', encoding='utf-8') as f:
        depressed = json.load(f)
    
    # 加载正常用户数据
    with open(normal_path, 'r', encoding='utf-8') as f:
        normal = json.load(f)
    
    # 数据清洗函数
    def clean_text(text):
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^A-Za-z0-9\u4e00-\u9fff]+', ' ', text)
        text = text.lower().strip()
        return text
    
    user_data = []
    # 处理抑郁用户
    for user in depressed:
        tweets = [clean_text(t['tweet_content']) for t in user['tweets']]
        # 只保留非空文本
        tweets = [t for t in tweets if len(t) > 0]
        if len(tweets) > 0:  # 至少有一条有效推文
            user_data.append({
                'tweets': tweets,
                'label': 1
            })
    
    # 处理正常用户
    for user in normal:
        tweets = [clean_text(t['tweet_content']) for t in user.get('tweets', [])]
        tweets = [t for t in tweets if len(t) > 0]
        if len(tweets) > 0:
            user_data.append({
                'tweets': tweets,
                'label': 0
            })
    
    logger.info(f"数据加载完成，共 {len(user_data)} 个用户")
    logger.info(f"抑郁用户: {sum(1 for u in user_data if u['label'] == 1)} 个")
    logger.info(f"正常用户: {sum(1 for u in user_data if u['label'] == 0)} 个")
            
    return pd.DataFrame(user_data)

# 2. 加载Qwen-0.6B模型（使用ModelScope）
def load_qwen_model():
    logger.info("开始加载Qwen-0.6B模型...")
    
    model_kwargs = {
        'device_map': 'auto',
        'trust_remote_code': True,
        'torch_dtype': torch.bfloat16,
        #'revision': 'v0.6.0'
    }
    
    # 根据GPU显存选择加载方式
    '''
    if device.type == 'cuda' and torch.cuda.get_device_properties(0).total_memory < 15 * 1024**3:
        model_kwargs['load_in_4bit'] = True
        logger.info("使用4bit量化加载模型以节省显存")
    else:
        model_kwargs['low_cpu_mem_usage'] = True
        logger.info("使用标准模式加载模型")
    '''

    # 加载模型和tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-Embedding-0.6B", 
            trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            "Qwen/Qwen3-Embedding-0.6B", **model_kwargs
        ).eval()
        logger.info("Qwen-0.6B模型加载成功")
        return model, tokenizer
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        raise

# 3. 使用Qwen生成文本向量
class QwenEmbedder:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.embedding_size = None  # Qwen-0.6B的嵌入维度
    
    def embed_texts(self, texts, batch_size=256):
        """批量生成文本嵌入向量，优化内存使用"""
        logger.info(f"开始生成文本嵌入，共 {len(texts)} 条文本，batch_size={batch_size}")
        
        embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        start_time = time.time()
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            # 过滤空文本
            batch = [text for text in batch if len(text.strip()) > 0]
            if not batch:
                continue
                
            # 编码文本（截断至512token）
            inputs = self.tokenizer(
                batch, 
                padding=True, 
                truncation=True, 
                max_length=512, 
                return_tensors="pt"
            ).to(device)
            
            # 生成嵌入（不计算梯度）
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                last_hidden = outputs.hidden_states[-1]
                attention_mask = inputs['attention_mask'].unsqueeze(-1)
                masked_hidden = last_hidden * attention_mask
                batch_embeddings = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1)
                batch_embeddings = batch_embeddings.to(torch.float32).cpu().numpy()
                # 动态设置embedding_size
                if self.embedding_size is None:
                    self.embedding_size = batch_embeddings.shape[1]
            
            embeddings.append(batch_embeddings)
            
            # 清理中间变量
            del inputs, outputs, last_hidden, masked_hidden
            torch.cuda.empty_cache()
            
            # 进度报告
            if (i // batch_size) % 10 == 0:
                elapsed = time.time() - start_time
                batches_done = i // batch_size + 1
                remaining = (elapsed / batches_done) * (total_batches - batches_done)
                logger.info(f"进度: {i+len(batch)}/{len(texts)} | "
                          f"用时: {elapsed:.1f}s | "
                          f"预计剩余: {remaining:.1f}s")
        
        # 处理特殊情况：所有文本都为空
        if not embeddings:
            return np.zeros((len(texts), self.embedding_size))
            
        embeddings = np.vstack(embeddings)
        logger.info(f"文本嵌入生成完成，总用时: {time.time()-start_time:.1f}秒")
        return embeddings

# 4. 为每个用户生成推文向量矩阵（优化版）
def generate_user_embeddings(user_df, embedder, max_tweets=50, batch_size=256):
    logger.info("开始生成用户嵌入矩阵...")
    
    # 先收集所有推文，批量处理提高效率
    all_tweets = []
    user_tweet_counts = []
    
    for _, row in user_df.iterrows():
        # 过滤过短文本（少于10字符）
        valid_tweets = [t for t in row['tweets'] if len(t.strip()) > 10]
        if len(valid_tweets) > max_tweets:
            valid_tweets = valid_tweets[:max_tweets]
        all_tweets.extend(valid_tweets)
        user_tweet_counts.append(len(valid_tweets))
    
    logger.info(f"共收集 {len(all_tweets)} 条推文，来自 {len(user_df)} 个用户")
    
    # 批量生成所有推文的嵌入向量
    tweet_embeddings = embedder.embed_texts(all_tweets, batch_size=batch_size)
    
    # 为每个用户构建矩阵
    user_embeddings = []
    idx = 0
    for count in user_tweet_counts:
        user_vecs = tweet_embeddings[idx:idx+count]
        idx += count
        
        # 填充或截断
        if len(user_vecs) < max_tweets:
            padding = np.zeros((max_tweets - len(user_vecs), embedder.embedding_size))
            user_vecs = np.vstack([user_vecs, padding])
        elif len(user_vecs) > max_tweets:
            user_vecs = user_vecs[:max_tweets]
            
        user_embeddings.append(user_vecs)
    
    logger.info("用户嵌入矩阵生成完成")
    return np.array(user_embeddings, dtype=np.float32)

# 5. 模型定义（调整为4096维输入）
class MILSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.3):
        super(MILSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 推文级LSTM
        self.tweet_lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True, dropout=dropout
        )
        
        # 注意力池化层
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # 分类层
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x形状: (batch_size, num_tweets, embedding_dim)
        batch_size, num_tweets, _ = x.size()
        
        # 处理每条推文
        # 重塑输入: (batch_size * num_tweets, 1, embedding_dim)
        x_flat = x.view(batch_size * num_tweets, 1, -1)
        
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers * 2, x_flat.size(0), self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers * 2, x_flat.size(0), self.hidden_dim, device=device)
        
        tweet_out, _ = self.tweet_lstm(x_flat, (h0, c0))
        tweet_out = tweet_out[:, -1, :]  # 取最后时间步输出
        
        # 恢复用户维度: (batch_size, num_tweets, hidden_dim*2)
        user_repr = tweet_out.view(batch_size, num_tweets, -1)
        
        # 注意力机制
        attn_weights = self.attention(user_repr)  # (batch_size, num_tweets, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        weighted_repr = torch.sum(attn_weights * user_repr, dim=1)
        
        # 分类
        out = self.fc(weighted_repr)
        out = self.sigmoid(out)
        return out.squeeze()

# 6. 模型评估函数
def evaluate_model(model, data_loader):
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_x)
            probs = outputs.cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(batch_y.cpu().numpy())
    
    return all_labels, all_preds, all_probs

# 主程序
if __name__ == "__main__":
    # 确保输出目录存在
    os.makedirs('results', exist_ok=True)
    
    # 加载模型优先
    qwen_model, qwen_tokenizer = load_qwen_model()
    embedder = QwenEmbedder(qwen_model, qwen_tokenizer)
    
    # 加载数据
    user_df = load_user_level_data('depressed_5000.json', 'normal_5000.json')
    
    # 生成用户嵌入矩阵（使用优化后的方法）
    MAX_TWEETS = 50  # 每个用户最多考虑的推文数（根据内存调整）
    embed_batch_size = 8  # 视显存而定

    # 如果本地已存在嵌入文件则直接加载
    embedding_path = f'results/user_embeddings_{MAX_TWEETS}.npy'
    label_path = f'results/user_labels.npy'
    if os.path.exists(embedding_path) and os.path.exists(label_path):
        logger.info("检测到本地嵌入文件，直接加载...")
        X = np.load(embedding_path)
        y = np.load(label_path)
    else:
        X = generate_user_embeddings(user_df, embedder, MAX_TWEETS, embed_batch_size)
        y = user_df['label'].values
        np.save(embedding_path, X)
        np.save(label_path, y)
        logger.info(f"用户嵌入和标签已保存到 {embedding_path} 和 {label_path}")
    
    # 释放嵌入模型内存
    logger.info("释放嵌入模型内存...")
    del qwen_model, qwen_tokenizer, embedder
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    # 重塑数据为3D张量: (样本数, 推文数, 嵌入维度)
    X = X.reshape(X.shape[0], MAX_TWEETS, X.shape[2])
    
    # 第一次划分：训练集(60%)，临时集(40%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    # 第二次划分：验证集(20%)，测试集(20%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    logger.info(f"训练集大小: {len(X_train)} 用户")
    logger.info(f"验证集大小: {len(X_val)} 用户")
    logger.info(f"测试集大小: {len(X_test)} 用户")
    
    # 转换为PyTorch张量
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    logger.info("数据加载器创建完成")
    
    # 初始化MIL模型
    input_dim = X.shape[2]  # 嵌入维度 (4096)
    hidden_dim = 256        # 隐藏层维度
    output_dim = 1          # 输出维度
    num_layers= 2           # LSTM层数
    
    model = MILSTM(input_dim, hidden_dim, output_dim, num_layers).to(device)
    logger.info(f"模型初始化完成，输入维度: {input_dim}, 隐藏层维度: {hidden_dim}")
    
    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # 训练模型
    num_epochs = 20
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    logger.info(f"开始训练模型，共 {num_epochs} 个epoch...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        epoch_start = time.time()
        # 训练一个epoch
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            # 
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # 计算验证集损失和准确率
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                # 
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                predicted = (outputs > 0.5).float()
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        train_loss = running_loss / len(train_loader)
        val_loss = val_loss / len(test_loader)
        accuracy = correct / total
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(accuracy)
        
        logger.info(f'Epoch [{epoch+1}/{num_epochs}], '
                   f'Train Loss: {train_loss:.4f}, '
                   f'Val Loss: {val_loss:.4f}, '
                   f'Val Accuracy: {accuracy:.4f}, '
                   f'Time: {time.time()-epoch_start:.1f}s')
    
    # 保存模型
    model_path = 'results/mil_lstm_classifier.pth'
    torch.save(model.state_dict(), model_path)
    logger.info(f"模型已保存至: {model_path}")
    
    # 绘制训练过程
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.xticks(range(len(train_losses)))
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.xticks(range(len(train_losses)))
    
    plt.tight_layout()
    plt.savefig('results/training_history.png')
    logger.info("训练历史图已保存")
    
    # 模型性能评估
    logger.info("开始模型评估...")
    y_true, y_pred, y_probs = evaluate_model(model, test_loader)
    
    # 分类报告
    report = classification_report(y_true, y_pred, target_names=['Normal', 'Depressed'])
    logger.info("分类报告:\n" + report)
    
    # 保存分类报告
    with open('results/classification_report.txt', 'w') as f:
        f.write(report)
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Depressed'], 
                yticklabels=['Normal', 'Depressed'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('results/confusion_matrix.png')
    logger.info("混淆矩阵已保存")
    
    # ROC曲线
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('results/roc_curve.png')
    logger.info("ROC曲线已保存")
    
    logger.info("训练和评估完成！")