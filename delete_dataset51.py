import fiftyone as fo

# 获取所有存在的 dataset 名称
datasets = fo.list_datasets()

if not datasets:
    print("No datasets found.")
else:
    print("Available datasets:")
    for i, name in enumerate(datasets):
        print(f"[{i}] {name}")

    # 提示用户选择要删除的 dataset，可以输入多个索引，用逗号分隔
    to_delete = input("Enter the numbers of the datasets to delete (comma-separated): ")
    try:
        indices = [int(x.strip()) for x in to_delete.split(",")]
        for idx in indices:
            if 0 <= idx < len(datasets):
                name = datasets[idx]
                fo.delete_dataset(name)
                print(f"Deleted dataset: {name}")
            else:
                print(f"Index {idx} is out of range.")
    except ValueError:
        print("Invalid input. Please enter comma-separated numbers.")

