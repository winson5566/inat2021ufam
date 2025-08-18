import json
import os

# 读取原始 train_mini.json
with open("inat2021/train_mini.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 提取 categories
categories = data.get("categories", [])

# 保存到新文件 inat2021/categories.json
os.makedirs("inat2021", exist_ok=True)
with open("inat2021/categories.json", "w", encoding="utf-8") as f:
    json.dump(categories, f, ensure_ascii=False, indent=2)

print(f"✅ 已保存 {len(categories)} 个类别到 inat2021/categories.json")
