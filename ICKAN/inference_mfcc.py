#预测
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from train_all_mfcc import AudioDataset_MFCC, load_dataset_files
from models.SimpleModels import SimpleCNN, SimpleCNN_2, SimpleLinear
from models.ConvNet import ConvNet
from models.conv_and_kan import ICKAN
from models.baseline import ConvComp

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义类别
class_names = [
    'Piano', 'Violin', 'Guitar', 'Cello', 'Saxophone', 'Flute', 'Oboe', 
    'Clarinet', 'Trumpet', 'Trombone', 'French Horn', 'Double Bass', 
    'Harp', 'Harmonica', 'Accordion', 'Organ', 'Marimba', 'Vibraphone', 
    'Celesta', 'Xylophone'
]

# 定义模型字典
model_dict = {
    "cnn_small": SimpleCNN,
    "cnn_Medium": SimpleCNN_2,
    "mlp": SimpleLinear,
    "ConvNet": ConvNet,
    "baseline": ConvComp,
    "ICKAN": ICKAN,
}

def evaluate_model(model, test_loader, model_name, criterion):
    """使用测试集评估模型"""
    model.eval()
    total_files = 0
    correct_predictions = 0
    total_loss = 0
    
    # 用于记录每个类别的性能
    class_correct = {class_name: 0 for class_name in class_names}
    class_total = {class_name: 0 for class_name in class_names}
    confusion_matrix = torch.zeros(len(class_names), len(class_names))
    
    print(f"\nStarting evaluation for {model_name}...")
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc=f"Evaluating {model_name}"):
            data, target = data.to(device), target.to(device)
            
            # 获取模型输出
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            
            # 获取预测结果
            predictions = torch.argmax(output, dim=1)
            
            # 更新混淆矩阵
            for t, p in zip(target.view(-1), predictions.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            
            # 更新统计信息
            for i, (pred, true_label) in enumerate(zip(predictions, target)):
                true_class = class_names[true_label.item()]
                pred_class = class_names[pred.item()]
                
                class_total[true_class] += 1
                if pred_class == true_class:
                    class_correct[true_class] += 1
                    correct_predictions += 1
            
            total_files += len(target)
            
            if total_files % 50 == 0:
                print(f"\nProcessed {total_files} files")
                print(f"Current accuracy: {correct_predictions/total_files:.2%}")
    
    # 计算每个类别的precision, recall, f1
    class_metrics = {}
    avg_precision = 0
    avg_recall = 0
    avg_f1 = 0
    
    for i, class_name in enumerate(class_names):
        tp = confusion_matrix[i, i]
        fp = confusion_matrix[:, i].sum() - tp
        fn = confusion_matrix[i, :].sum() - tp
        
        precision = float(tp / (tp + fp) if (tp + fp) > 0 else 0)
        recall = float(tp / (tp + fn) if (tp + fn) > 0 else 0)
        f1 = float(2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0)
        
        class_metrics[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': class_correct[class_name]/class_total[class_name] if class_total[class_name] > 0 else 0
        }
        
        avg_precision += precision
        avg_recall += recall
        avg_f1 += f1
    
    # 计算平均指标
    avg_precision /= len(class_names)
    avg_recall /= len(class_names)
    avg_f1 /= len(class_names)
    final_accuracy = correct_predictions / total_files
    avg_loss = total_loss / total_files
    
    # 保存详细结果到文件
    result_str = f"\nFinal Results for {model_name}:\n"
    result_str += f"Overall Accuracy: {final_accuracy:.2%}\n"
    result_str += f"Average Loss: {avg_loss:.4f}\n"
    result_str += f"Average Precision: {avg_precision:.2%}\n"
    result_str += f"Average Recall: {avg_recall:.2%}\n"
    result_str += f"Average F1-Score: {avg_f1:.2%}\n\n"
    
    result_str += "Per-class Metrics:\n"
    for class_name, metrics in class_metrics.items():
        result_str += f"\n{class_name}:\n"
        result_str += f"Accuracy: {metrics['accuracy']:.2%}\n"
        result_str += f"Precision: {metrics['precision']:.2%}\n"
        result_str += f"Recall: {metrics['recall']:.2%}\n"
        result_str += f"F1-Score: {metrics['f1']:.2%}\n"
    
    # 保存结果到文件
    with open(f'results/{model_name}_detailed_results.txt', 'w') as f:
        f.write(result_str)
    
    return {
        'accuracy': final_accuracy,
        'precision': avg_precision,
        'recall': avg_recall,
        'f1': avg_f1,
        'loss': avg_loss,
        'class_metrics': class_metrics
    }

def evaluate_all_models(test_loader):
    """评估所有模型"""
    results_dict = {}
    os.makedirs('results', exist_ok=True)  # 创建结果目录
    
    for model_name in model_dict:
        print(f"\nEvaluating {model_name}...")
        
        try:
            model = model_dict[model_name]()
            criterion = nn.CrossEntropyLoss()
            
            model_path = f"checkpoint/{model_name}/checkpoint_epoch_10.pth"
            print(f"Loading weights from: {model_path}")
            
            # 加载检查点
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # 打印一些检查点信息
            print(f"Checkpoint epoch: {checkpoint['epoch']}")
            print(f"Checkpoint loss: {checkpoint['loss']}")
            
            model.to(device)
            model.eval()
            
            # 评估模型并获取结果
            results = evaluate_model(model, test_loader, model_name, criterion)
            results_dict[model_name] = results
            
        except Exception as e:
            print(f"Error evaluating {model_name}: {str(e)}")
            print(f"Full error: {str(sys.exc_info())}")
            continue
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 保存所有模型的综合结果
    with open('results/all_models_summary.txt', 'w') as f:
        f.write("Summary of all models:\n\n")
        for model_name, results in results_dict.items():
            f.write(f"\n{model_name}:\n")
            f.write(f"Accuracy: {results['accuracy']:.2%}\n")
            f.write(f"Precision: {results['precision']:.2%}\n")
            f.write(f"Recall: {results['recall']:.2%}\n")
            f.write(f"F1-Score: {results['f1']:.2%}\n")
            f.write(f"Loss: {results['loss']:.4f}\n")

if __name__ == "__main__":
    current_directory = os.getcwd()

    # 加载测试集
    print("Loading test dataset...")
    _, test_paths, _, test_labels = load_dataset_files("ICKAN_Dataset")
    test_dataset = AudioDataset_MFCC(test_paths, test_labels, n_mels=64, max_length=431)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 评估所有模型
    evaluate_all_models(test_loader)

