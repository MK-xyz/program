import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 學生資訊
STUDENT_NAME = "游孟凱"
STUDENT_ID = "113101032"

print(f"學生：{STUDENT_NAME}")
print(f"學號：{STUDENT_ID}")
print("=" * 50)

# ==================== 第一部分：環境熟悉 (20分) ====================

print("第一部分：環境熟悉")

# 題目 1.1：檢查 Kaggle 環境 (5分)
print("1.1 檢查可用的套件版本")
import sys
print(f"Python 版本: {sys.version}")
print(f"NumPy 版本: {np.__version__}")
print(f"Pandas 版本: {pd.__version__}")

# 題目 1.2：載入內建資料集 (10分)
print("\n1.2 使用 Kaggle 內建資料集")
# 使用 seaborn 的內建資料集作為範例
titanic=pd.read_csv("C:/Users/xyz70/Desktop/program/titanic/train.csv")

print("Titanic 資料集基本資訊：")
print(f"資料形狀: {titanic.shape}")
print(f"欄位名稱: {list(titanic.columns)}")

# 請完成：顯示前 5 筆資料
print(f"前 5 筆資料：{titanic[:5]}")
# 題目 1.3：基本資料探索 (5分)
print("\n1.3 基本資料統計")
# 請完成：計算存活率
survival_rate = titanic["Survived"].mean()
print(f"整體存活率: {survival_rate:.2%}")

# ==================== 第二部分：Python 基礎 (25分) ====================

print("\n" + "=" * 50)
print("第二部分：Python 基礎")

# 題目 2.1：列表操作 (8分)
print("2.1 列表和迴圈操作")
ages = [22, 38, 26, 35, 35, 27, 54, 2, 27, 14]

# 請完成以下任務：
# a) 計算平均年齡
average_age = sum(ages)/len(ages)

# b) 找出所有大於 30 的年齡
ages_over_30 = []
for i in ages:
    if i>30:
       ages_over_30.append(i)

# c) 計算年齡的標準差（不使用 numpy）
age_std = 0
for i in ages:
   age_std += (i-average_age)**2 
age_std = (age_std/len(ages))**0.5

print("年齡統計結果：")
print(f"平均年齡: {average_age}")
print(f"大於30的年齡: {ages_over_30}")
print(f"標準差: {age_std}")

# 題目 2.2：函數定義 (8分)
print("\n2.2 函數定義與使用")

def analyze_passenger_class(data):
    """
    分析乘客艙等的統計資訊
    請完成這個函數
    """
    # 請完成函數內容
    # 回傳每個艙等的人數和存活率
    analyze={"Pclass":[1,2,3],
             "number":[0,0,0],
             "Survival_rate":[0,0,0]}
    for i in range(len(data)):
        analyze["number"][data["Pclass"][i]-1]+=1
        analyze["Survival_rate"][data["Pclass"][i]-1]+=data["Survived"][i]
    analyze = pd.DataFrame(analyze)
    analyze["Survival_rate"]=analyze["Survival_rate"]/analyze["number"]
    return analyze

# 測試你的函數
class_analysis = analyze_passenger_class(titanic)
print("艙等分析結果：")
print(class_analysis)

# 題目 2.3：字典操作 (9分)
print("\n2.3 字典和資料整理")

# 建立一個字典來儲存不同港口的登船人數
embark_counts = {}

# 請完成：統計每個登船港口的人數
# 提示：使用 titanic['Embarked'] 欄位
for port in titanic['Embarked'].dropna():
    if port in embark_counts:
        embark_counts[port]+=1
    else:
        embark_counts[port]=1

print("登船港口統計：")
for port, count in embark_counts.items():
    print(f"{port}: {count} 人")

# ==================== 第三部分：資料科學套件 (30分) ====================

print("\n" + "=" * 50)
print("第三部分：資料科學套件")

# 題目 3.1：NumPy 操作 (10分)
print("3.1 NumPy 陣列操作")

# 建立一個模擬的票價陣列
np.random.seed(42)
fake_fares = np.random.normal(50, 20, 100)

# 請完成以下計算：
# a) 計算平均票價
mean_fare = fake_fares.mean()

# b) 找出票價的 25%, 50%, 75% 分位數
quartiles = np.quantile(fake_fares, [0.25,0.5,0.75])

# c) 計算有多少票價超過平均值 + 1個標準差
outliers_count = 0
for i in fake_fares:
    if i>fake_fares.mean()+fake_fares.std():outliers_count+=1

print("票價統計：")
print(f"平均票價: ${mean_fare:.2f}")
print(f"分位數: {quartiles}")
print(f"異常值數量: {outliers_count}")

# 題目 3.2：Pandas 進階操作 (15分)
print("\n3.2 Pandas 資料處理")

# 請完成以下任務：
# a) 建立一個新欄位 'age_group'，將年齡分為：兒童(<18)、成人(18-60)、老人(>60)
bins = [0, 18, 60, 100]
labels = ["兒童", "成人", "老人"]
titanic["age_group"] = pd.cut(titanic["Age"], bins=bins, labels=labels, right=False)


# b) 計算每個年齡組的存活率
survival_by_age = titanic.groupby("age_group")["Survived"].mean()

# c) 找出票價最高的 10 位乘客資訊
top_10_expensive = titanic.sort_values(by="Fare", ascending=False)[:10]

print("年齡組存活率：")
print(survival_by_age)

print("\n票價最高的 10 位乘客：")
print(top_10_expensive[['Name', 'Age', 'Fare', 'Survived']])

# 題目 3.3：資料視覺化 (5分)
print("\n3.3 基本視覺化")

# 請建立以下圖表：
# a) 年齡分布直方圖
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.hist(titanic["Age"].dropna())
plt.title('Age Distribution')
plt.xlabel('Age')             
plt.ylabel('Count')
# b) 艙等存活率長條圖
plt.subplot(1, 3, 2)
survived_by_class=titanic.groupby("age_group")["Survived"].mean()
plt.bar(survived_by_class.index,survived_by_class.values)
plt.title('Survival Rate by Class')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')

# c) 票價 vs 年齡散佈圖
plt.subplot(1, 3, 3)
plt.scatter(titanic["Age"],titanic["Fare"])
plt.title('Fare vs Age')
plt.xlabel('Age')
plt.ylabel('Fare')

plt.tight_layout()
plt.show()
# ==================== 第四部分：機器學習入門 (25分) ====================

print("\n" + "=" * 50)
print("第四部分：機器學習入門")

# 題目 4.1：資料預處理 (10分)
print("4.1 資料預處理")

# 準備建模資料
# 請完成以下預處理步驟：

#a) 選擇特徵欄位（排除文字和缺失值過多的欄位）
selected_features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
titanic_selected = titanic[selected_features + ["Survived"]].copy()

# b) 處理缺失值
titanic_clean = titanic_selected.dropna(subset=selected_features)

# c) 將類別變數轉換為數值（例如：性別）
# 提示：可以使用 pd.get_dummies() 或手動編碼
titanic_trans = pd.get_dummies(titanic_clean, columns=["Sex", "Embarked"], drop_first=True)

print("預處理後的資料形狀：")
print(f"特徵數量: {titanic_trans.shape[1]}")
print(f"樣本數量: {titanic_trans.shape[0]}")

# 題目 4.2：模型訓練 (10分)
print("\n4.2 模型訓練與評估")

# 請完成以下步驟：
# a) 分離特徵和目標變數
X = titanic_trans.drop("Survived", axis=1)
y = titanic_trans["Survived"]

# b) 分割訓練集和測試集 (80:20)
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2)

# c) 訓練邏輯回歸模型
model = LogisticRegression()
model.fit(X_train,y_train)

# d) 進行預測
predictions = model.predict(X_test)

# e) 計算準確率
accuracy = accuracy_score(y_test,predictions)

print("模型評估結果：")
print(f"準確率: {accuracy:.3f}")
print("\n詳細分類報告：")
print(classification_report(y_test, predictions))

# 題目 4.3：結果視覺化 (5分)
print("\n4.3 結果視覺化")

# 請建立混淆矩陣熱力圖
from sklearn.metrics import confusion_matrix

confusion_mat = confusion_matrix(y_test, predictions) 
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ==================== 自我評估與反思 ====================

print("\n" + "=" * 50)
print("自我評估與反思")

self_assessment = """
請回答以下問題：

1. 完成這個評估大約花了多少時間？

2. 哪個部分對你來說最困難？為什麼？

3. 你對哪個部分最有信心？

4. 在機器學習方面，你希望在課程中學到什麼？

5. 你之前有使用過 Kaggle 嗎？對這個平台的感想如何？

你的回答：
1. 2-3小時
2.機器學習，有學過但沒有從頭寫過
3.無
4.各種模型的詳細原理與編寫
5.沒有
"""

print(self_assessment)

print("評估完成！")
print("請將此 notebook 設為 Public 並分享連結，或下載後提交。")
print("記得檢查所有程式碼都能正常執行！")
