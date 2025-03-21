import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
import os

# 配置参数
class Config:
    # 模型路径配置
    model_dir = r"C:\VScodePro\.venv\BERT\bert-base-chinese"
    tokenizer_dir = r"C:\VScodePro\.venv\BERT\bert-base-chinese"#使用本地地址的路径，不知道为什么相对不行
    
    # 训练参数
    num_labels = 10
    batch_size = 32# 每批数据量
    max_len = 256# 文本最大长度
    learning_rate = 2e-5# 学习率
    epochs = 5# 训练轮次
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#存在GPU使用GPU提高训练效率
    
    # 数据路径
    train_data_path = "./cleaned_comments.jsonl"
    output_model_path = "./bert_finetuned.bin"

# 验证模型文件结构
def validate_model_files(model_dir):
    required_files = [
        "config.json",          # 模型配置文件
        "pytorch_model.bin",    # PyTorch模型权重
        "vocab.txt",            # 词表文件
        "tokenizer_config.json" # 分词器配置
    ]
    
    missing_files = []
    for f in required_files:
        if not os.path.exists(os.path.join(model_dir, f)):
            missing_files.append(f)
    
    if missing_files:
        raise FileNotFoundError(
            f"缺少必要模型文件: {missing_files}\n"
            "请确认已正确放置以下文件：\n"
            "1. config.json\n"
            "2. pytorch_model.bin\n"
            "3. vocab.txt\n"
            "4. tokenizer_config.json"#debug，由于神秘的原因找不到文件
        )

# 自定义数据集
class CommentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts          # 文本列表
        self.labels = labels        # 评分列表（1-10）
        self.tokenizer = tokenizer  # BERT分词器
        self.max_len = max_len  # 文本最大长度

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx] - 1  # 将1-10分转换为0-9的类别索引

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True, # 添加[CLS]和[SEP]

           # 1.[CLS]（Classification Token）Classification Token（分类标记）位置：总是出现在输入序列的开头
        #用于分类任务的整体表示：在文本分类任务中，模型会将[CLS]对应位置的最终隐藏层输出（即[CLS]的向量）作为整个输入序列的语义表示，用于分类。
        #类似于“聚合标记”：通过自注意力机制，[CLS]会融合整个序列的信息。
    #EXAMPLE
        # 输入文本："这部电影很棒"
        # 分词后添加[CLS]和[SEP]：
        #['[CLS]', '这', '部', '电', '影', '很', '棒', '[SEP]']
        #模型会将 [CLS] 对应的输出向量传递给分类器，用于预测评分。

            #2. [SEP]（Separator Token）Separator Token（分隔标记）出现在输入序列的结尾

        # 句子对任务（如问答、推理）：分隔两个句子，例如：
        #[CLS]句子1[SEP]句子2[SEP]
        #标识句子边界：帮助模型区分不同句子或文本段。终止符功能：标记输入序列的结束位置。

        # 句子对任务："今天天气怎么样？[SEP]今天阳光明媚。"
        #['[CLS]', '今', '天', '天', '气', '怎', '么', '样', '？', '[SEP]', 
        #'今', '天', '阳', '光', '明', '媚', '。', '[SEP]']
        #3. 添加方式
        #在代码中，tokenizer.encode_plus方法的add_special_tokens=True参数会自动添加这些标记：

        #encoding = tokenizer.encode_plus(
            # text,
            # add_special_tokens=True,  # 自动添加[CLS]和[SEP]
                # ...
            max_length=self.max_len,
            truncation=True,#截断
            padding="max_length",#填充
            return_attention_mask=True,#注意力掩码（区分真实内容和填充
            return_tensors="pt",# 返回PyTorch张量
            )      

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# 数据加载函数
def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = CommentDataset(
        texts=df.text.to_numpy(), # 从读取文本列
        labels=df.point.to_numpy(),# 读取评分列
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4,# 使用4个子进程加载数据
        shuffle=True# 打乱训练数据顺序
    )

# 训练函数
def train_epoch(model, data_loader, optimizer, device, scheduler, n_examples):
    model.train()
    losses = []
    correct_predictions = 0
    
    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # 前向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        # 获取损失和预测结果
        loss = outputs.loss
        logits = outputs.logits

        # 统计指标
        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        # 反向传播与优化
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    # 计算本epoch平均指标
    acc = correct_predictions.double() / n_examples
    loss = np.mean(losses)
    
    return acc, loss

# 验证函数
def eval_model(model, data_loader, device, n_examples):
    model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad(): #禁用梯度计算 不执行反向传播和参数更新
        for batch in tqdm(data_loader, desc="Validation"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits

            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

    acc = correct_predictions.double() / n_examples
    loss = np.mean(losses)
    
    return acc, loss

def main():
    config = Config()
    
    # 验证模型文件
    validate_model_files(config.model_dir)
    
    # 加载分词器
    tokenizer = BertTokenizer.from_pretrained(
        config.tokenizer_dir,
        vocab_file=os.path.join(config.tokenizer_dir, "vocab.txt"),
        config_file=os.path.join(config.tokenizer_dir, "tokenizer_config.json")
    )
    
    # 加载模型
    model = BertForSequenceClassification.from_pretrained(
        config.model_dir,
        num_labels=config.num_labels,
        local_files_only=True  # 强制使用本地文件
    )
    model = model.to(config.device)
    
    # 加载数据
    df = pd.read_json(config.train_data_path, lines=True)
    df_train, df_val = train_test_split(df, test_size=0.2, random_state=42)
    
    # 创建数据加载器
    train_data_loader = create_data_loader(df_train, tokenizer, config.max_len, config.batch_size)
    val_data_loader = create_data_loader(df_val, tokenizer, config.max_len, config.batch_size)
    
    # 优化器和调度器
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    total_steps = len(train_data_loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # 训练循环
    best_accuracy = 0
    for epoch in range(config.epochs):
        print(f"Epoch {epoch + 1}/{config.epochs}")
        
        # 修正的缩进部分
        train_acc, train_loss = train_epoch(
            model,
            train_data_loader,
            optimizer,
            config.device,
            scheduler,
            len(df_train))
        
        val_acc, val_loss = eval_model(
            model,
            val_data_loader,
            config.device,
            len(df_val))
        
        print(f"Train loss: {train_loss:.4f}, accuracy: {train_acc:.4f}")
        print(f"Val loss: {val_loss:.4f}, accuracy: {val_acc:.4f}")
        
        if val_acc > best_accuracy:
            torch.save(model.state_dict(), config.output_model_path)
            best_accuracy = val_acc
            print("Saved best model")

if __name__ == "__main__":
    main()