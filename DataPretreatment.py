import numpy as np
import os
import pandas as pd
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 清空文件
def clear_file(file_name='result.txt'):
    with open(file_name, 'w', encoding='utf-8') as f:
        f.write('')

# 结果写入文件
def write_to_file(content, file_name='result.txt'):
    with open(file_name, 'a', encoding='utf-8') as f:
        f.write(content + '\n')

# 创建保存图像的文件夹
def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

# 保存图像的函数
def save_fig(fig_name, fig_directory="figures"):
    create_directory(fig_directory)
    plt.savefig(f"{fig_directory}/{fig_name}.png", bbox_inches="tight")

# 导入数据
def load_data(filepath):
    dataset = pd.read_csv(filepath, header=0, encoding='unicode_escape')
    return dataset

# 清理数据
def clean_data(dataset):
    dataset['Customer Full Name'] = dataset['Customer Fname'].astype(str) + dataset['Customer Lname'].astype(str)
    data = dataset.drop(['Customer Email', 'Product Status', 'Customer Password', 'Customer Street',
                         'Customer Fname', 'Customer Lname', 'Latitude', 'Longitude',
                         'Product Description', 'Product Image', 'Order Zipcode',
                         'shipping date (DateOrders)'], axis=1)
    data['Customer Zipcode'] = data['Customer Zipcode'].fillna(0)
    return data

# 可视化热图
def plot_heatmap(data):
    numeric_data = data.select_dtypes(include=[np.number])
    plt.figure(figsize=(24, 24))
    sns.heatmap(numeric_data.corr(), annot=True, linewidths=.5, fmt='.1g', cmap='coolwarm')
    plt.title('Correlation Heatmap')
    save_fig('heatmap')
    plt.show()

# 市场销售分析
def plot_sales_by_market_and_region(data):
    # 市场与地区
    market = data.groupby('Market')['Sales per customer'].sum().sort_values(ascending=False)
    region = data.groupby('Order Region')['Sales per customer'].sum().sort_values(ascending=False)

    # 写入文件
    market_sales_content = "所有市场的总销售额:\n"
    for index, value in market.items():
        market_sales_content += f"{index}: {value:.2f}\n"
    write_to_file(market_sales_content)
    region_sales_content = "所有地区的总销售额:\n"
    for index, value in region.items():
        region_sales_content += f"{index}: {value:.2f}\n"
    write_to_file(region_sales_content)

    # 绘制市场的总销售额图
    plt.figure(figsize=(12, 6))
    ax1 = market.plot.bar(color='skyblue')
    plt.title("所有市场的总销售额")
    plt.xticks(rotation=45)

    for i, value in enumerate(market):
        plt.text(i, value, f'{value:.2f}', ha='center', va='bottom', fontsize=10)

    save_fig('total_sales_by_market')
    plt.show()

    # 绘制地区的总销售额图
    plt.figure(figsize=(12, 8))
    ax2 = region.plot.bar(color='lightgreen')
    plt.title("所有地区的总销售额")
    plt.xticks(rotation=60)

    for i, value in enumerate(region):
        plt.text(i, value, f'{value:.2f}', ha='center', va='bottom', fontsize=10, rotation=45)

    save_fig('total_sales_by_region')
    plt.show()

# 按类别分析销售情况并保存图像
def category_sales_analysis(data):
    # 按类别进行分组
    cat = data.groupby('Category Name')

    # 所有类别的总销售额
    total_sales = cat['Sales per customer'].sum().sort_values(ascending=False)
    plt.figure(figsize=(12, 6))
    total_sales.plot.bar(title="所有类别的总销售额")
    save_fig('total_sales_by_category')
    plt.show()

    # 输出结果到文件
    write_to_file("所有类别的总销售额:\n" + total_sales.to_string())

    # 所有类别的平均销售额
    avg_sales = cat['Sales per customer'].mean().sort_values(ascending=False)
    plt.figure(figsize=(12, 6))
    avg_sales.plot.bar(title="所有类别的平均销售额")
    save_fig('avg_sales_by_category')  # 保存图像为PNG文件
    plt.show()

    # 输出结果到文件
    write_to_file("所有类别的平均销售额:\n" + avg_sales.to_string())

    # 所有类别的平均价格
    avg_price = cat['Product Price'].mean().sort_values(ascending=False)
    plt.figure(figsize=(12, 6))
    avg_price.plot.bar(title="所有类别的平均价格")
    save_fig('avg_price_by_category')  # 保存图像为PNG文件
    plt.show()

    # 输出结果到文件
    write_to_file("所有类别的平均价格:\n" + avg_price.to_string())

# 订单时间趋势分析并保存图像
def time_trend_analysis(data):
    # 将订单日期分为年、月、周、日、小时
    data['order_year'] = pd.DatetimeIndex(data['order date (DateOrders)']).year
    data['order_month'] = pd.DatetimeIndex(data['order date (DateOrders)']).month
    data['order_week_day'] = pd.DatetimeIndex(data['order date (DateOrders)']).weekday
    data['order_hour'] = pd.DatetimeIndex(data['order date (DateOrders)']).hour
    data['order_month_year'] = pd.to_datetime(data['order date (DateOrders)']).dt.to_period('M')

    # 按季度分析销售趋势
    quarter = data.groupby('order_month_year')
    quartersales = quarter['Sales'].sum().resample('Q').mean()

    # 绘制季度销售额趋势
    plt.figure(figsize=(15, 6))
    quartersales.plot(title='季度销售额趋势')
    plt.ylabel('销售额')
    plt.xlabel('季度')
    save_fig('quarter_sales_trend')  # 保存为PNG图像
    plt.show()

    # 输出季度趋势到文件
    write_to_file("季度销售额趋势:\n" + quartersales.to_string())

    # 按周几分析销售趋势
    week_sales = data.groupby('order_week_day')['Sales'].sum()
    plt.figure(figsize=(10, 6))
    week_sales.plot.bar(title='按星期几的销售额')
    plt.xticks([0, 1, 2, 3, 4, 5, 6], ['周一', '周二', '周三', '周四', '周五', '周六', '周日'], rotation=0)
    plt.ylabel('销售额')
    save_fig('weekly_sales_trend')  # 保存为PNG图像
    plt.show()

    # 输出周趋势到文件
    write_to_file("按星期几的销售额:\n" + week_sales.to_string())

    # 按小时分析销售趋势
    hour_sales = data.groupby('order_hour')['Sales'].sum()
    plt.figure(figsize=(10, 6))
    hour_sales.plot.bar(title='按小时的销售额')
    plt.ylabel('销售额')
    plt.xlabel('小时')
    save_fig('hourly_sales_trend')  # 保存为PNG图像
    plt.show()

    # 输出小时趋势到文件
    write_to_file("按小时的销售额:\n" + hour_sales.to_string())

    # 按月份分析销售趋势
    month_sales = data.groupby('order_month')['Sales'].sum()
    plt.figure(figsize=(10, 6))
    month_sales.plot.bar(title='按月份的销售额')
    plt.ylabel('销售额')
    plt.xlabel('月份')
    plt.xticks(range(1, 13), [f'{i}月' for i in range(1, 13)], rotation=0)
    save_fig('monthly_sales_trend')  # 保存为PNG图像
    plt.show()

    # 输出月趋势到文件
    write_to_file("按月份的销售额:\n" + month_sales.to_string())

    plt.figure(figsize=(10, 12))
    plt.subplot(4, 2, 1)
    quarter = data.groupby('order_year')
    quarter['Sales'].mean().plot(figsize=(12, 12), title='三年平均销售额')
    plt.subplot(4, 2, 2)
    days = data.groupby("order_week_day")
    days['Sales'].mean().plot(figsize=(12, 12), title='每周平均销售额（天）')
    plt.subplot(4, 2, 3)
    hrs = data.groupby("order_hour")
    hrs['Sales'].mean().plot(figsize=(12, 12), title='每天平均销售额（小时）')
    plt.subplot(4, 2, 4)
    month = data.groupby("order_month")
    month['Sales'].mean().plot(figsize=(12, 12), title='年平均销售额（月）')
    plt.tight_layout()
    save_fig('time_trend_analysis')
    plt.show()

# 产品价格与销售额关系分析并保存图像
def price_sales_relationship(data):
    # 绘制价格与每位客户销售额的关系图
    data.plot(x='Product Price', y='Sales per customer', linestyle='dotted', marker='o',
              markerfacecolor='blue', markersize=12)
    plt.title('产品价格与每位客户的销售额')
    plt.xlabel('产品价格')
    plt.ylabel('每个客户的销售额')
    save_fig('price_vs_sales')  # 保存图像为PNG文件
    plt.show()

    # 输出描述性分析到文件
    result = f"价格与销售额的相关性：\n产品价格与销售额之间的趋势展示在图表中。"
    write_to_file(result)

# 支付方式分析
def plot_payment_type_analysis(data):
    # 计算每种支付方式在每个地区出现的次数
    xyz1 = data[(data['Type'] == 'TRANSFER')]
    xyz2 = data[(data['Type'] == 'CASH')]
    xyz3 = data[(data['Type'] == 'PAYMENT')]
    xyz4 = data[(data['Type'] == 'DEBIT')]

    count1 = xyz1['Order Region'].value_counts()
    count2 = xyz2['Order Region'].value_counts()
    count3 = xyz3['Order Region'].value_counts()
    count4 = xyz4['Order Region'].value_counts()

    names = data['Order Region'].value_counts().keys()
    n_groups = 23
    fig, ax = plt.subplots(figsize=(20, 8))
    index = np.arange(n_groups)
    bar_width = 0.2
    opacity = 0.6

    # 绘制柱状图
    plt.bar(index, count1, bar_width, alpha=opacity, color='b', label='Transfer')
    plt.bar(index + bar_width, count2, bar_width, alpha=opacity, color='r', label='Cash')
    plt.bar(index + 2 * bar_width, count3, bar_width, alpha=opacity, color='g', label='Payment')
    plt.bar(index + 3 * bar_width, count4, bar_width, alpha=opacity, color='y', label='Debit')

    plt.xlabel('订单区域')
    plt.ylabel('付款次数')
    plt.title('所有地区使用不同类型的付款')
    plt.xticks(index + bar_width, names, rotation=90)
    plt.legend()
    plt.tight_layout()
    save_fig('payment_types_analysis')
    plt.show()

    # 将分析结果写入文件
    result_text = "各地区支付方式分析结果：\n"
    result_text += "TRANSFER:\n" + count1.to_string() + "\n\n"
    result_text += "CASH:\n" + count2.to_string() + "\n\n"
    result_text += "PAYMENT:\n" + count3.to_string() + "\n\n"
    result_text += "DEBIT:\n" + count4.to_string() + "\n\n"

    # 调用write_to_file()写入分析结果
    write_to_file(result_text)

# 损失分析
def plot_loss_analysis(data):
    # 筛选出损失的订单，即收益为负数的订单
    loss = data[(data['Benefit per order'] < 0)]

    # 绘制损失最大的产品类别的条形图
    plt.figure(figsize=(20, 8))
    loss['Category Name'].value_counts().nlargest(10).plot.bar()
    plt.title("损失最大的产品")
    plt.xticks(rotation=0)
    save_fig('top_10_loss_products')
    plt.show()

    # 绘制损失最大的地区的条形图
    plt.figure(figsize=(20, 8))
    loss['Order Region'].value_counts().nlargest(10).plot.bar()
    plt.title("损失最大的地区")
    plt.xticks(rotation=0)
    save_fig('top_10_loss_regions')
    plt.show()

    # 计算订单的总损失收入
    total_loss = loss['Benefit per order'].sum()
    print('订单损失总收入', total_loss)
    result_text = "损失分析结果：\n"
    result_text += "损失最大的产品类别:\n" + loss['Category Name'].value_counts().nlargest(10).to_string() + "\n\n"
    result_text += "损失最大的地区:\n" + loss['Order Region'].value_counts().nlargest(10).to_string() + "\n\n"
    result_text += f"订单损失总收入: {total_loss}\n"

    write_to_file(result_text)

# 欺诈检测
# 绘制欺诈地区的饼图
def plot_fraud_analysis(data):
    # 筛选出疑似欺诈并使用转账支付的订单
    high_fraud = data[(data['Order Status'] == 'SUSPECTED_FRAUD') & (data['Type'] == 'TRANSFER')]

    # 绘制饼图
    fraud = high_fraud['Order Region'].value_counts().plot.pie(
        figsize=(24, 12), startangle=180,
        explode=[0.1] + [0] * 22, autopct='%.1f%%', shadow=True)

    plt.title("欺诈最高的地区", size=15, color='y')
    plt.ylabel(" ")
    plt.axis('equal')  # 确保饼图是圆的
    save_fig('fraud_regions')
    plt.show()

    # 输出到文件
    result_text = "欺诈最高的地区:\n" + high_fraud['Order Region'].value_counts().to_string() + "\n"
    write_to_file(result_text)


# 延迟交货产品分析
def plot_late_delivery_droducts(data):
    late_delivery = data[(data['Delivery Status'] == 'Late delivery')]

    # 绘制延迟交货的前10个产品的条形图
    plt.figure(figsize=(20, 8))
    late_delivery['Category Name'].value_counts().nlargest(10).plot.bar()
    plt.title("最迟交货的前10个产品")
    plt.xticks(rotation=0)
    save_fig('top_10_late_delivery_products')
    plt.show()

    # 输出到文件
    result_text = "最迟交货的前10个产品:\n" + late_delivery['Category Name'].value_counts().nlargest(
        10).to_string() + "\n"
    write_to_file(result_text)


# 绘制欺诈产品的对比分析
def plot_fraud_products_comparison(data):
    # 所有地区的欺诈订单
    high_fraud1 = data[(data['Order Status'] == 'SUSPECTED_FRAUD')]
    # 西欧地区的欺诈订单
    high_fraud2 = data[(data['Order Status'] == 'SUSPECTED_FRAUD') & (data['Order Region'] == 'Western Europe')]

    # 绘制所有地区的欺诈产品
    plt.figure(figsize=(20, 8))
    fraud1 = high_fraud1['Category Name'].value_counts().nlargest(10).plot.bar(color='orange', label="所有地区")

    # 绘制西欧的欺诈产品
    plt.figure(figsize=(20, 8))
    fraud2 = high_fraud2['Category Name'].value_counts().nlargest(10).plot.bar(color='green', label="西欧")

    plt.legend(["所有地区", "西欧"])
    plt.title("欺诈检测率最高的十大产品", size=15)
    plt.xlabel("产品", size=13)
    plt.ylim(0, 600)
    plt.xticks(rotation=0)
    save_fig('tplot_fraud_products_comparison')
    plt.show()

    # 输出到文件
    result_text = "所有地区欺诈检测率最高的产品:\n" + high_fraud1['Category Name'].value_counts().nlargest(
        10).to_string() + "\n"
    result_text += "西欧地区欺诈检测率最高的产品:\n" + high_fraud2['Category Name'].value_counts().nlargest(
        10).to_string() + "\n"
    write_to_file(result_text)

# 绘制十大最可疑欺诈客户的条形图
def plot_top_suspected_fraud_customers(data):
        # 筛选出涉嫌欺诈的订单
        cus = data[data['Order Status'] == 'SUSPECTED_FRAUD']

        # 获取十大最可疑欺诈客户
        top_customers = cus['Customer Full Name'].value_counts().nlargest(10)
        fraud_amounts = {}
        for customer in top_customers.index:
            customer_orders = cus[cus['Customer Full Name'] == customer]
            fraud_amounts[customer] = customer_orders['Sales'].sum()

        # 获取诈骗金额最高的客户及其金额
        highest_fraud_customer = max(fraud_amounts, key=fraud_amounts.get)
        highest_fraud_amount = fraud_amounts[highest_fraud_customer]

        write_to_file("十大最可疑欺诈客户:\n")
        for customer, count in top_customers.items():
            write_to_file(f"{customer}: {count}次订单")
        write_to_file(f"诈骗金额最高的客户: {highest_fraud_customer}, 诈骗金额: {highest_fraud_amount:.2f}\n\n")

        # 绘制十大最可疑欺诈客户的条形图
        plt.figure(figsize=(20, 8))
        ax = top_customers.plot.bar(color='purple', title="十大最可疑欺诈客户")

        # 标注诈骗金额最高的客户的金额
        highest_fraud_index = top_customers.index.get_loc(highest_fraud_customer)
        highest_sales = top_customers[highest_fraud_customer]

        # 在图中标注诈骗金额最高的客户
        plt.text(highest_fraud_index, highest_sales + 0.5,
                 f': {highest_fraud_amount:.2f}', color='red', ha='center',
                 fontsize=12, weight='bold')

        # 保存图像并写入输出文件
        save_fig('top_suspected_fraud_customers')
        plt.show()

def RFM_Customer_(data):
    # R_Value（receignity）表示客户上次订购后所用时间。
    # F_Value（Frequency）表示客户订购的次数。
    # M_Value（Monetary value）告诉我们客户花了多少钱购买物品。
    # 计算每个订单的总价
    data['TotalPrice'] = data['Order Item Quantity'] * data['Order Item Total']

    # 计算最后一个订单日期
    data['order date (DateOrders)'] = pd.to_datetime(data['order date (DateOrders)'])
    present = dt.datetime(2018, 2, 1)

    # 将数据分组并聚合
    Customer_seg = data.groupby('Order Customer Id').agg({
        'order date (DateOrders)': lambda x: (present - x.max()).days,
        'Order Id': lambda x: len(x),
        'TotalPrice': lambda x: x.sum()
    })

    # 修改列名
    Customer_seg.rename(columns={'order date (DateOrders)': 'R_Value',
                                 'Order Id': 'F_Value',
                                 'TotalPrice': 'M_Value'}, inplace=True)

    # 绘制R、F、M的分布图，并保存为PNG文件
    plt.figure(figsize=(12, 10))
    plt.subplot(3, 1, 1)
    sns.histplot(Customer_seg['R_Value'], kde=True)
    plt.title("R_Value 分布")

    plt.subplot(3, 1, 2)
    sns.histplot(Customer_seg['F_Value'], kde=True)
    plt.title("F_Value 分布")

    plt.subplot(3, 1, 3)
    sns.histplot(Customer_seg['M_Value'], kde=True)
    plt.title("M_Value 分布")

    plt.tight_layout()
    save_fig('RFM_distribution')
    plt.show()

    # 将RFM数据划分为四个分位数
    quantiles = Customer_seg.quantile(q=[0.25, 0.5, 0.75]).to_dict()

    # R_Value 分数的函数
    def R_Score(a, b, c):
        if a <= c[b][0.25]:
            return 1
        elif a <= c[b][0.50]:
            return 2
        elif a <= c[b][0.75]:
            return 3
        else:
            return 4

    # F_Value 和 M_Value 的分数函数
    def FM_Score(x, y, z):
        if x <= z[y][0.25]:
            return 4
        elif x <= z[y][0.50]:
            return 3
        elif x <= z[y][0.75]:
            return 2
        else:
            return 1

    # 计算 R、F、M 的分数
    Customer_seg['R_Score'] = Customer_seg['R_Value'].apply(R_Score, args=('R_Value', quantiles))
    Customer_seg['F_Score'] = Customer_seg['F_Value'].apply(FM_Score, args=('F_Value', quantiles))
    Customer_seg['M_Score'] = Customer_seg['M_Value'].apply(FM_Score, args=('M_Value', quantiles))

    # 创建RFM分数
    Customer_seg['RFM_Score'] = Customer_seg['R_Score'].astype(str) + Customer_seg['F_Score'].astype(str) + \
                                Customer_seg['M_Score'].astype(str)

    # 计算总RFM分数
    Customer_seg['RFM_Total_Score'] = Customer_seg[['R_Score', 'F_Score', 'M_Score']].sum(axis=1)

    # 定义客户细分的函数
    def RFM_Total_Score(df):
        if df['RFM_Total_Score'] >= 11:
            return 'Champions'
        elif df['RFM_Total_Score'] == 10:
            return 'Loyal Customers'
        elif df['RFM_Total_Score'] == 9:
            return 'Recent Customers'
        elif df['RFM_Total_Score'] == 8:
            return 'Promising'
        elif df['RFM_Total_Score'] == 7:
            return 'Customers Needing Attention'
        elif df['RFM_Total_Score'] == 6:
            return 'Can\'t lose them'
        elif df['RFM_Total_Score'] == 5:
            return 'At Risk'
        else:
            return 'Lost'

    # 应用RFM级别
    Customer_seg['Customer_Segmentation'] = Customer_seg.apply(RFM_Total_Score, axis=1)

    # 保存客户细分结果到CSV文件
    Customer_seg.to_csv('Customer_Segmentation.csv')

    # 打印前5行查看结果
    Customer_seg.head()

    # 输出总客户群分类
    unique_segments = Customer_seg['RFM_Score'].unique()
    print(f"共有 {len(unique_segments)} 个不同的客户群细分：", unique_segments)
    write_to_file(f"客户细分结果(RFM级别)：\n{unique_segments}\n")

    # 每个细分市场有多少客户？
    # 计算每个RFM级别的平均值，并返回每个段的大小
    Customer_seg['Customer_Segmentation'].value_counts().plot.pie(figsize=(10, 10),
                                                                  startangle=135, explode=(0, 0, 0, 0.1, 0, 0, 0, 0),
                                                                  autopct='%.1f', shadow=True)
    plt.title("Customer Segmentation", size=15)
    plt.ylabel(" ")
    plt.axis('equal')
    save_fig('customer_segmentation')
    plt.show()

# 延迟交货订单
def plot_late_delivery_analysis(data):
    # 过滤具有延迟交货风险的订单
    xyz1 = data[(data['Late_delivery_risk'] == 1)]
    # 筛选实际发生延迟交货的订单
    xyz2 = data[(data['Delivery Status'] == 'Late delivery')]

    # 统计每个地区的延迟交货风险和实际延迟交货的订单数量
    count1 = xyz1['Order Region'].value_counts()
    count2 = xyz2['Order Region'].value_counts()
    names = data['Order Region'].value_counts().keys()

    # 确定条形图组的数量
    n_groups = len(names)
    fig, ax = plt.subplots(figsize=(20, 8))

    # 设置条形图的索引和宽度
    index = np.arange(n_groups)
    bar_width = 0.35  # 调整为合适的宽度
    opacity = 0.8

    # 绘制延迟交货风险与实际延迟交货的柱状图
    type1 = plt.bar(index, count1.reindex(names, fill_value=0), bar_width,
                    alpha=opacity, color='g', label='延迟交货的风险')

    type2 = plt.bar(index + bar_width, count2.reindex(names, fill_value=0), bar_width,
                    alpha=opacity, color='b', label='延迟交货')

    # 设置图表标题和标签
    plt.xlabel('订单区域')
    plt.ylabel('订单数量')
    plt.title('所有地区延迟交付的订单分布')
    plt.legend()
    plt.xticks(index + bar_width / 2, names, rotation=90)
    plt.tight_layout()
    save_fig('Late_delivery_analysis')
    plt.show()

    # 将分析结果写入文件
    result_text = "各地区延迟交货分析结果：\n"
    result_text += "具有延迟交货风险的订单数量:\n" + count1.to_string() + "\n\n"
    result_text += "实际延迟交货的订单数量:\n" + count2.to_string() + "\n\n"

    # 调用 write_to_file() 将分析结果写入文件
    write_to_file(result_text)

#延迟运输方式
def plot_transport_Shipping_Mode(data):
    # 使用标准类装运筛选延迟交货订单
    xyz1 = data[(data['Delivery Status'] == 'Late delivery') & (data['Shipping Mode'] == 'Standard Class')]
    # 使用头等舱装运筛选延迟交货订单
    xyz2 = data[(data['Delivery Status'] == 'Late delivery') & (data['Shipping Mode'] == 'First Class')]
    # 使用二级装运筛选延迟交货订单
    xyz3 = data[(data['Delivery Status'] == 'Late delivery') & (data['Shipping Mode'] == 'Second Class')]
    # 过滤当天发货的延迟交货订单
    xyz4 = data[(data['Delivery Status'] == 'Late delivery') & (data['Shipping Mode'] == 'Same Day')]
    # 计算总值
    count1 = xyz1['Order Region'].value_counts()
    count2 = xyz2['Order Region'].value_counts()
    count3 = xyz3['Order Region'].value_counts()
    count4 = xyz4['Order Region'].value_counts()
    # 索引名称
    names = data['Order Region'].value_counts().keys()
    n_groups = 23
    fig, ax = plt.subplots(figsize=(20, 8))
    index = np.arange(n_groups)
    bar_width = 0.2
    opacity = 0.6
    type1 = plt.bar(index, count1, bar_width, alpha=opacity, color='b', label='标准舱')
    type2 = plt.bar(index + bar_width, count2, bar_width, alpha=opacity, color='r', label='头等舱')
    type3 = plt.bar(index + bar_width + bar_width, count3, bar_width, alpha=opacity, color='g', label='二等舱')
    type4 = plt.bar(index + bar_width + bar_width + bar_width, count4, bar_width, alpha=opacity, color='y',label='当天')
    plt.xlabel('订单区域')
    plt.ylabel('装运数量')
    plt.title('各地区采用不同类型的运输方式')
    plt.legend()
    plt.xticks(index + bar_width, names, rotation=90)
    plt.tight_layout()
    save_fig('Transport_Shipping_Mode')
    plt.show()

    result_text = f"不同运输方式的延迟交货分析：\n标准舱：\n{count1.to_string()}\n\n头等舱：\n{count2.to_string()}\n\n二等舱：\n{count3.to_string()}\n\n当天发货：\n{count4.to_string()}\n"
    write_to_file(result_text)

# 主函数：运行所有分析
def main():
    clear_file('result.txt')

    # 数据加载和清理
    filepath = "数据集/DataCoSupplyChainDataset.csv"
    dataset = load_data(filepath)
    data = clean_data(dataset)

    # 热图可视化
    plot_heatmap(data)
    #
    # # 销售分析
    # plot_sales_by_market_and_region(data)
    #
    # # 支付方式分析
    # plot_payment_type_analysis(data)
    #
    # # 损失分析
    # plot_loss_analysis(data)
    #
    # # 欺诈检测
    # plot_fraud_analysis(data)
    #
    # # 延迟交货分析
    # plot_late_delivery_droducts(data)
    #
    # # 欺诈产品分析
    # plot_fraud_products_comparison(data)
    #
    # # 欺诈客户分析
    # plot_top_suspected_fraud_customers(data)
    #
    # # 延迟交货订单
    # plot_late_delivery_analysis(data)
    #
    # # 运输方式
    # plot_transport_Shipping_Mode(data)
    #
    # # 分析各类别的销售情况
    # category_sales_analysis(data)
    #
    # # 分析产品价格与销售额的关系
    # price_sales_relationship(data)
    #
    # # 分析订单时间趋势
    # time_trend_analysis(data)
    #
    # # RFM客户订单分析
    # RFM_Customer_(data)

if __name__ == "__main__":
    main()
