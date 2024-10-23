from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import pandas as pd
import numpy as np
import xgboost as xgb
import datetime as dt
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from DataPretreatment import save_fig

# 隐藏警告
warnings.filterwarnings('ignore')

# 加载数据集
def load_data(file_path):
    with tqdm(total=1, desc="Loading dataset") as pbar:
        dataset = pd.read_csv(file_path, header=0, encoding='unicode_escape')
        # 打印行数
        print(f"数据集共有 {dataset.shape[0]} 行数据")
        pbar.update(1)
    return dataset


# 数据预处理
def preprocess_data(dataset):
    with tqdm(total=1, desc="Preprocessing data") as pbar:
        # 合并名字与姓氏，创建新列
        dataset['Customer Full Name'] = dataset['Customer Fname'].astype(str) + dataset['Customer Lname'].astype(str)

        # 删除不重要的列
        data = dataset.drop(['Customer Email', 'Product Status', 'Customer Password', 'Customer Street',
                             'Customer Fname', 'Customer Lname', 'Latitude', 'Longitude',
                             'Product Description', 'Product Image', 'Order Zipcode',
                             'shipping date (DateOrders)'], axis=1)

        # 填补缺失的邮政编码
        data['Customer Zipcode'] = data['Customer Zipcode'].fillna(0)

        # 时间相关特征提取
        data['order_year'] = pd.DatetimeIndex(data['order date (DateOrders)']).year
        data['order_month'] = pd.DatetimeIndex(data['order date (DateOrders)']).month
        data['order_week_day'] = pd.DatetimeIndex(data['order date (DateOrders)']).weekday
        data['order_hour'] = pd.DatetimeIndex(data['order date (DateOrders)']).hour
        data['TotalPrice'] = data['Order Item Quantity'] * data['Order Item Total']

        # 转换时间格式
        present = dt.datetime(2018, 2, 1)
        data['order date (DateOrders)'] = pd.to_datetime(data['order date (DateOrders)'])

        # 创建新的分类列
        data['fraud'] = np.where(data['Order Status'] == 'SUSPECTED_FRAUD', 1, 0)
        data['late_delivery'] = np.where(data['Delivery Status'] == 'Late delivery', 1, 0)

        # 删除重复信息的列，移除不存在的'order_month_year'
        data = data.drop(['Delivery Status', 'Late_delivery_risk', 'Order Status', 'order date (DateOrders)'], axis=1)

        # 标签编码
        le = LabelEncoder()
        categorical_columns = ['Customer Country', 'Market', 'Type', 'Product Name', 'Customer Segment',
                               'Customer State', 'Order Region', 'Order City', 'Category Name',
                               'Customer City', 'Department Name', 'Order State', 'Shipping Mode',
                               'order_week_day', 'Order Country', 'Customer Full Name']
        for col in tqdm(categorical_columns, desc="Encoding categorical columns"):
            data[col] = le.fit_transform(data[col])
        pbar.update(1)

    return data


# 划分训练集和测试集
def split_data(data, target_col):
    X = data.loc[:, data.columns != target_col]
    y = data[target_col]
    with tqdm(total=1, desc="Splitting dataset") as pbar:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        pbar.update(1)
    return X_train, X_test, y_train, y_test


# 标准化数据
def standardize_data(X_train, X_test):
    with tqdm(total=1, desc="Standardizing data") as pbar:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        pbar.update(1)
    return X_train, X_test


# 评估模型性能并绘制ROC曲线
def evaluate_model(model, X_train, X_test, y_train, y_test, target_name):
    with tqdm(total=1, desc=f"Training {model.__class__.__name__}") as pbar:
        model.fit(X_train, y_train)
        pbar.update(1)

    with tqdm(total=1, desc=f"Evaluating {model.__class__.__name__}") as pbar:
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        # 计算AUC和ROC曲线
        if y_pred_prob is not None:
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
            roc_auc = auc(fpr, tpr)

            # 绘制ROC曲线
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve ({model.__class__.__name__} - {target_name})')
            plt.legend(loc="lower right")
            save_fig(f"ROC_Curve_{model.__class__.__name__}_{target_name}.png", "Roc")
            plt.show()
        else:
            roc_auc = "N/A"

        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        print(f"Model: {model.__class__.__name__}")
        print(f"Accuracy ({target_name}): {accuracy * 100:.2f}%")
        print(f"Recall ({target_name}): {recall * 100:.2f}%")
        print(f"F1 Score ({target_name}): {f1 * 100:.2f}%")
        print(f"AUC ({target_name}): {roc_auc}")
        print(f"Confusion Matrix ({target_name}):\n{conf_matrix}\n")

        pbar.update(1)

    return accuracy, recall, f1, roc_auc

# 评估回归模型性能
def evaluate_regression_model(model, X_train, X_test, y_train, y_test, target_name):
    with tqdm(total=1, desc=f"Training {model.__class__.__name__}") as pbar:
        model.fit(X_train, y_train)
        pbar.update(1)

    with tqdm(total=1, desc=f"Evaluating {model.__class__.__name__}") as pbar:
        y_pred = model.predict(X_test)

        # 计算回归评估指标
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        print(f"Model: {model.__class__.__name__}")
        print(f"Mean Squared Error ({target_name}): {mse:.2f}")
        print(f"Root Mean Squared Error ({target_name}): {rmse:.2f}")
        print(f"R² ({target_name}): {r2:.2f}\n")

        pbar.update(1)

    return mse, rmse, r2

# 执行多个回归模型并比较
def run_order_quantity_regression(X_train, X_test, y_train, y_test):
    models = [LinearRegression(),
              RandomForestRegressor(),
              xgb.XGBRegressor()]

    results = []
    for model in tqdm(models, desc="Running order quantity prediction models"):
        mse, rmse, r2 = evaluate_regression_model(model, X_train, X_test, y_train, y_test, "Order Quantity")

        results.append({
            "Model": model.__class__.__name__,
            "MSE": mse,
            "RMSE": rmse,
            "R²": r2
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv("order_quantity_model_results.csv", index=False)

    return results_df

# 执行多个模型并比较
def run_models(X_train_f, X_test_f, y_train_f, y_test_f, X_train_l, X_test_l, y_train_l, y_test_l):
    models = [LogisticRegression(solver='lbfgs'),
              # GaussianNB(),
              # LinearSVC(),
              # KNeighborsClassifier(n_neighbors=1),
              # LinearDiscriminantAnalysis(),
              RandomForestClassifier(),
              # ExtraTreesClassifier(),
              # xgb.XGBClassifier(),
              DecisionTreeClassifier()]

    results = []
    for model in tqdm(models, desc="Running models"):
        print(f"Evaluating {model.__class__.__name__} for fraud detection...")
        acc_f, recall_f, f1_f, auc_f = evaluate_model(model, X_train_f, X_test_f, y_train_f, y_test_f, "fraud")

        print(f"Evaluating {model.__class__.__name__} for late delivery prediction...")
        acc_l, recall_l, f1_l, auc_l = evaluate_model(model, X_train_l, X_test_l, y_train_l, y_test_l, "late_delivery")

        results.append({
            "Model": model.__class__.__name__,
            "Fraud Detection Accuracy": acc_f,
            "Fraud Detection Recall": recall_f,
            "Fraud Detection F1": f1_f,
            "Fraud Detection AUC": auc_f,
            "Late Delivery Accuracy": acc_l,
            "Late Delivery Recall": recall_l,
            "Late Delivery F1": f1_l,
            "Late Delivery AUC": auc_l
        })

    # 将结果保存csv文件
    results_df = pd.DataFrame(results)
    results_df.to_csv("model_results.csv", index=False)

    return results_df


# 主程序入口
def main():
    # 加载和处理数据
    dataset = load_data("数据集/DataCoSupplyChainDataset.csv")
    processed_data = preprocess_data(dataset)

    # 划分欺诈检测数据集
    X_train_f, X_test_f, y_train_f, y_test_f = split_data(processed_data, 'fraud')
    X_train_f, X_test_f = standardize_data(X_train_f, X_test_f)

    # 划分延迟交货数据集
    X_train_l, X_test_l, y_train_l, y_test_l = split_data(processed_data, 'late_delivery')
    X_train_l, X_test_l = standardize_data(X_train_l, X_test_l)

    # 运行并评估模型
    results = run_models(X_train_f, X_test_f, y_train_f, y_test_f, X_train_l, X_test_l, y_train_l, y_test_l)
    print(results)

    # 订单量预测数据集划分和标准化
    X_train_q, X_test_q, y_train_q, y_test_q = split_data(processed_data, 'Order Item Quantity')
    X_train_q, X_test_q = standardize_data(X_train_q, X_test_q)

    # 运行并评估回归模型
    results = run_order_quantity_regression(X_train_q, X_test_q, y_train_q, y_test_q)
    print(results)


if __name__ == "__main__":
    main()
