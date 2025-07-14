一些毫无作用的序言：此readme为关于everything.py脚本以及其他两个辅助脚本的具体使用说明，这个脚本将赋予你在FiftyOne和CVAT之间传输数据时的超爽体验（或超糟体验）。  
作者是时年20岁，美国圣路易斯华盛顿大学的Rising Junior黄越同学，南京人，天秤座，曼联死忠球迷。  
他在他来北京实习的这段时间费劲了几乎他的浑身解数完成了这个可能对99%人来说都微不足道，毫不起眼的任务。  
如果您正在使用他写的这个脚本，他想对你表达由衷的感激！   
如果您正在阅读此readme并感觉得到了有用的帮助，这将是他最大的荣幸！  

If you have any question or comments regarding the script or readme, never hesitate to reach out to him at h.yue1@wustl.edu.


正文从这里开始：
# Part I: 使用脚本前：环境变量配置
在一切开始之前，确保您的服务器账号里已经安装了fiftyone和ultralytics。  
然后，请登录您的服务器账号，只要您需要完成将数据从fiftyone上传到CVAT的任务，就必须麻烦您在任意目录下配置好您的CVAT账号信息：

```bash
export FIFTYONE_CVAT_USERNAME=<YOUR_CVAT_USERNAME>
export FIFTYONE_CVAT_PASSWORD=<YOUR_CVAT_PASSWORD>
export FIFTYONE_CVAT_URL= <VALID_URL>

#使用示例:
export FIFTYONE_CVAT_USERNAME="YueeeeH"
export FIFTYONE_CVAT_PASSWORD="Boeing737800"
export FIFTYONE_CVAT_URL="http://192.168.200.4:8080"
```


# Part II: 使用脚本前的须知

请您cd到存有此脚本的目录，进入conda的base环境，如下所示：  
```(base) huangyue@booster-BVG:~/0625$ ```  
注：在此示例中，“0625”文件夹中含有everything.py脚本 和 接下来将要导入fiftyone的数据文件夹  
请您牢记，无论什么情况下运行脚本必须遵从的命令行格式   
python everything.py <数据集名称> <模型文件名>   --后为可选的参数 



# Part III: 使用脚本（一）：从外部导入新数据到fiftyone
## 情况（一）：现有一存有纯图片的文件夹，不含任何label
### 示例情景： 
```
(base) huangyue@booster-BVG:~/0625/207-realsense/2025-06-25-23-05-50$ ls
color_1750863951.069933.jpg  color_1750864964.248727.jpg  depth_1750864636.772997.png  pose_1750864350.114441.yaml
color_1750863980.182525.jpg  color_1750864965.049534.jpg  depth_1750864637.773034.png  pose_1750864350.681252.yaml
```
### 脚本使用方法：
``` python everything.py <直接包含纯图片的文件夹的路径> <模型文件名> (optional: --prelabel --prelabel-class <label名称>) ```  
- 使用示例：（1）  
    ```(base) huangyue@booster-BVG:~/0625$ python everything.py 207-realsense/2025-06-25-23-05-50/ 0508.pt```  
    注：上一行中，我的所有图片在0625/207-realsense/2025-06-25-23-05-50/路径下，故如此输入 

- 使用示例：（2）  
    ```(base) huangyue@booster-BVG:~/0625$ python everything.py 207-realsense/2025-06-25-23-05-50/ 0508.pt best1115.pt```  
    注：上一行中，输入多个模型文件，会使用多个模型对数据集进行推理  
    注：预期接下来会出现大量的滚动，会在fiftyone里生成两个dataset，一个是不含任何label field的无前缀版本，一个是含模型推理的有cl_前缀的版本

- 使用示例：（3）  
    ```(base) huangyue@booster-BVG:~/0625$ python everything.py 207-realsense/2025-06-25-23-05-50/ 0508.pt best1115.pt --prelabel --prelabel-class Ball```  
    注：如果我再后面添加这两个optional的参数，那我可以选择只标注其中一个label，比如这里我只让模型标注Ball  
    注：可选的label名称: ['Ball', 'Goalpost', 'Person', 'LCross', 'TCross', 'XCross', 'PenaltyPoint', 'Opponent', 'BRMarker']  


## 情况（二）：现有一存有：图片文件夹、推理出来的标签的文件夹(和yaml文件)的 文件夹
### 示例情景：
```
(base) huangyue@booster-BVG:~/0625/fiftyone_export/prelable/cl_207-realsense/2025-06-25-23-05-50/_20250709_140946$ ls
dataset.yaml  images  labels
```
注：我的当前目录里存有dataset.yaml文件 images图片文件夹 以及 labels标签文件夹  
注：如果不存在yaml文件，脚本会自动生成

### 脚本使用方法：
```python everything.py <任意内容> <模型文件名> --importdataset <符合该情况定义的文件夹的路径> (optional: --rename <自定义数据集名称> --prelabel --prelabel-class <label名称>)```  
- 使用示例：（1）  
```(base) huangyue@booster-BVG:~/0625$ python everything.py abhkdf 0508.pt --importdataset 0707_37_prelabel/0707_37_prelabel/ ```  
    注：上一行中abhkdf的位置可以自己随便输入名称，最终新生成的数据集名称都会是imported_0707_37_prelabel  
    注：新生成的两个数据集里都会自动生成ground_truth字段，因为这是默认从现有标签导入时会存入的字段  
    注：带cl_前缀的数据集还会额外生成一个字段，其中存放着0508.pt模型推理的结果

- 使用示例：（2）  
```(base) huangyue@booster-BVG:~/0625$ python everything.py abhkdf 0508.pt --importdataset 0707_37_prelabel/0707_37_prelabel/ --rename testdataset --prelabel --prelabel-class LCross Ball```  
    注：此时我新导入的数据集会被命名为imported_testdataset和cl_imported_testdataset
    注：cl_前缀的数据集里会有模型推理出来的所有LCross和Ball的结果  



# Part IV: 使用脚本（二）：将fiftyone现有数据集上传到cvat

## 情况（一）： 想要把fiftyone一个现有数据集里的所有原图都上传，但不上传任何的标注狂或推理结果

### 脚本使用方法：
```python everything.py <一个fiftyone的现有数据集名称，千万不要输入cl_前缀> <任意挑选一个模型文件名> --upload```  
- 使用示例：（1）  
```(base) huangyue@booster-BVG:~/0625$ python everything.py imported_impsample02 0508.pt --upload```

- 使用示例：（2）  
```(base) huangyue@booster-BVG:~/0625$ python everything.py imported_impsample02 best1115.pt --upload```

- 使用示例：（3）  
```(base) huangyue@booster-BVG:~/0625$ python everything.py imported_impsample02 best1115.pt 0508.pt --upload``` 

    注：以上三种使用示例的效果是完全等价的，因为模型并不会在此过程中起到作用，但还是一定要输入至少一个模型文件名  
    注：现有数据集名称千万不要输入cl_前缀，因为在实际操作过程中，脚本会自动使用数据集的cl_版本去运行！！！
- 使用示例：（4）  
```(base) huangyue@booster-BVG:~/0625$ python everything.py imported_testdataset 0508.pt --upload --project testproject01```  
    注：如果我加上--project参数，那我可以指定将数据指定上传到某个cvat的project，默认project是```detection2507```



## 情况（二）：想要把fiftyone一个现有数据集里的一个特定字段的标注框（以及图片）上传

### 脚本使用方法：
```python everything.py <一个fiftyone的现有数据集名称，千万不要输入cl_前缀> <任意挑选一个模型文件名> --prelabelupload```  
- 使用示例：  
```(base) huangyue@booster-BVG:~/0625$ python everything.py imported_impsample02 best1115.pt --prelabelupload``` 
```
    Available detection label fields:
    [0] ground_truth
    [1] 05080pt_conf_002
    [2] best11150pt_conf_002
    Select the number for the field to upload (TARGET): 1
``` 
注：你需要在交互中输入你想要上传的字段的编号，如这里我想上传"05080pt_conf_002"字段的所有标注框，于是我输入他的对应编号1即可

## 情况（三）：想要把fiftyone一个现有数据集里的一个模型标注字段的标注框通过confidence筛选（optional: 以及IOU NMS筛选）后上传

### 脚本使用方法：  
```python everything.py <一个fiftyone数据集，不要输入cl_前缀> <任意挑选一个模型文件名> --filteredupload (optional: --filterconf <一个0-1之间的小数>) (optional: --filterIOU <一个0-1之间的小数>) ```  
- 使用示例：（1）  
```(base) huangyue@booster-BVG:~/0625$ python everything.py imported_impsample02 best1115.pt --filteredupload```
```
Available detection label fields:
[0] ground_truth
[1] 05080pt_conf_002
[2] best11150pt_conf_002
[3] newgroundtruth
Select the number for the field to upload (TARGET): 1
[4] None (skip ground truth filtering)
Select the number for the field to use as ground truth (GT) (or 4 for None): 4

Final selection:
TARGET_FIELD_4_FILTERED_UPLOADING: 05080pt_conf_002
GT_FIELD_4_FILTERED_UPLOADING: None
```
注：当我只输入--filteredupload后不加任何参数时，会默认confidence和IOU阈值为0.5  
注：我需要在前后两次交互中分别选择代表着被筛选的模型标注框字段，和作为我完全信赖的"人工标注"框的字段  
注：第二次交互，选择"人工标注"框的字段的过程中可以选择None（上为4），从而跳过第二层筛选，只进行对模型标注框的confidence筛选。  

- 使用示例：（2）  
``` (base) huangyue@booster-BVG:~/0625$ python everything.py imported_impsample02 best1115.pt --filteredupload --filterconf 0.65 ```  
```
Available detection label fields:
[0] ground_truth
[1] 05080pt_conf_002
[2] best11150pt_conf_002
[3] newgroundtruth
[4] merged_field_for_upload20250709_145437
Select the number for the field to upload (TARGET): 1
[5] None (skip ground truth filtering)
Select the number for the field to use as ground truth (GT) (or 5 for None): 3

Final selection:
TARGET_FIELD_4_FILTERED_UPLOADING: 05080pt_conf_002
GT_FIELD_4_FILTERED_UPLOADING: newgroundtruth
```
注：这次我选择了加入filterconf参数，使confidence阈值更改为了0.65，而IOU维持在默认的0.5阈值  
注：这次我选择了编号为3的newgroundtruth字段作为"人工标注"的字段进行筛选  

- 使用示例：（3）  
``` (base) huangyue@booster-BVG:~/0625$ python everything.py imported_impsample02 best1115.pt --filteredupload --filterconf 0.57 --filterIOU 0.42```
```
Available detection label fields:
[0] ground_truth
[1] 05080pt_conf_002
[2] best11150pt_conf_002
[3] newgroundtruth
[4] merged_field_for_upload20250709_145437
[5] merged_field_for_upload20250709_150226
[6] merged_field_for_upload20250709_150442
Select the number for the field to upload (TARGET): 1
[7] None (skip ground truth filtering)
Select the number for the field to use as ground truth (GT) (or 7 for None): 7

Final selection:
TARGET_FIELD_4_FILTERED_UPLOADING: 05080pt_conf_002
GT_FIELD_4_FILTERED_UPLOADING: None
```
注：这次我同时设置了confidence的IOU两个阈值的参数，但因为我给人工标注框的字段选的是None，所以IOU参数并不会造成任何影响，只会根据0.57对于conf字段进行筛选



# Part V: 使用脚本（三）：将CVAT标注好的数据集导回到fiftyone

### 脚本使用方法：
```python everything.py <一个此前已经上传到CVAT的fiftyone数据集，不含cl_前缀> <任意模型名称> --download (optional: --dest <传回的标注框进入的字段>)```
- 使用示例：（1）  
```(base) huangyue@booster-BVG:~/0625$ python everything.py imported_impsample02 best1115.pt --download --dest brandnewgroundtruth ```   
注：这里所有下载的标注框的来源是该数据集最后一次上传到CVAT的task，我为了方便大家不用管理annotation key在与顾文老师的交流后做出这个勉为其难的选择。希望不会带来太多麻烦  
注：这里因为使用了--dest 参数，后跟的字段会成为所有传回来的标注框的去处。无论这个字段名此前是否存在这条命令都能正确执行

- 使用示例：（2）  
```(base) huangyue@booster-BVG:~/0625$ python everything.py imported_impsample02 0508.pt --download```  
注：如果我没有加入--dest 参数，那么所有传回来的标注框都会默认进入"ground_truth"字段  
注：如果选择的传回来的字段此前已经存在，那么最终这个字段会含有原有标注框和新增标注框的并集，这是一个非常重要的问题！！！！！！



# Part VI: 使用脚本（四）：将现有fiftyone的数据集导出

### 脚本使用方法：  
```python everything.py <一个现有fiftyone数据集，不含cl_前缀> <任意模型名称> --exportdataset```  
- 使用示例：  
```(base) huangyue@booster-BVG:~/0625$ python everything.py 207-realsense/2025-06-25-23-05-50/ 0508.pt --exportdataset```
```
Available detection label fields:
[0] 05080pt_conf_002
[1] best11150pt_conf_002
[2] ground_truth
Select the number for the field to export (or 3 for None): 1
 100% |█████████████████████████████████████████████| 1154/1154 [1.2s elapsed, 0s remaining, 914.3 samples/s]
Export directory: fiftyone_export/prelable/cl_207-realsense/2025-06-25-23-05-50/_20250709_154307  
```  
注：交互会让你选择想要导出的标注框字段，根据编号选择即可      
注：导出的文件路径为Export directory，它含有时间戳，可根据时间戳快速找寻想要的文件夹



# Part VII: 使用脚本（五）：对现有fiftyone数据集的模型标注字段进行评价

### 脚本使用方法：  
```python everything.py <一个现有fiftyone数据集，不含cl_前缀> <任意模型名称> --eval```  
- 使用示例：
```(base) huangyue@booster-BVG:~/0625$ python everything.py 207-realsense/2025-06-25-23-05-50/ 0508.pt --eval```  
```
Available detection label fields:
[0] 05080pt_conf_002
[1] best11150pt_conf_002
[2] ground_truth
Select the number of the field serving as the target for evaluating: 0
Select the number of the field serving as the ground truth for evaluating: 2
```  
注：两次交互会分别让你选择 被评价的模型标注框的字段 和 作为评判标准的"人工"标注字段  
注：上图中我选择了对0508.pt模型的推理结果进行评价，参照标准为"ground_truth"字段  
注：evaluation key含有时间戳，如果一个数据集已经多次被评价，可以根据时间戳找寻想要的评价结果  



# Part VIII: 使用脚本（六）：导出fiftyone数据集里的bad_case（含评价结果）

### 脚本使用方法：  
```python everything.py <一个现有fiftyone数据集，不含cl_前缀> <任意模型名称> --eval --export-bad-case (optional: --fplabel <想要的fp标签>)```  
- 使用示例：（1）  
```(base) huangyue@booster-BVG:~/0625$ python everything.py 207-realsense/2025-06-25-23-05-50/ 0508.pt --eval --export-bad-case```
    ```
    Available detection label fields:
    [0] 05080pt_conf_002
    [1] best11150pt_conf_002
    [2] ground_truth
    Select the number of the field serving as the target for evaluating: 0
    Select the number of the field serving as the ground truth for evaluating: 2

    len of fn_view: 0
    len of fp_view: 718
    len of bad_case_view: 718
    exporting the field ground_truth to fiftyone_export/bad_case/cl_207-realsense/2025-06-25-23-05-50/_20250709_155906 
    ```
    注：--export-bad-case只有在有--eval的前提下才会触发，所以先走完eval流程后，会根据评价结果打印出fn fp bad_case_view的字段长度  
    注：导出的最终内容是你在交互中选择的作为"人工标注"内容的字段，导出的文件夹含时间戳  
    注：默认的fp标签是"Ball"，如需更改，请看使用示例（2）

- 使用示例：（2）  
``` (base) huangyue@booster-BVG:~/0625$ python everything.py 207-realsense/2025-06-25-23-05-50/ 0508.pt --eval --export-bad-case --fplabel LCross```
    ```
    Available detection label fields:
    [0] 05080pt_conf_002
    [1] best11150pt_conf_002
    [2] ground_truth
    Select the number of the field serving as the target for evaluating: 0
    Select the number of the field serving as the ground truth for evaluating: 2

    len of fn_view: 0
    len of fp_view: 627
    len of bad_case_view: 627
    exporting the field ground_truth to fiftyone_export/bad_case/cl_207-realsense/2025-06-25-23-05-50/_20250709_160724
    ```
    注：当我带上了fplabel后，我可以自己设置参数，找出特定标签的fp数量  
    注：本人偷了个小懒，没有在这里设置标签名称输错的情况下报错，如果出现fn fp都为0，要小心了！  
    注：所有标签名称在此，注意大小写：class_names = ['Ball', 'Goalpost', 'Person', 'LCross', 'TCross', 'XCross', 'PenaltyPoint', 'Opponent', 'BRMarker']



# Part IX: 使用delete_dataset51.py脚本，该脚本用于彻底删除一个或多个现有的fiftyone数据集

### 脚本使用方法：  
```python delete_dataset51.py```   
- 使用示例：（1）  
```(base) huangyue@booster-BVG:~/0625$ python delete_dataset51.py```
    ```
    Available datasets:
    [0] 207-realsense/2025-06-25-23-05-50/
    [1] cl_207-realsense/2025-06-25-23-05-50/
    [2] cl_imported_impsample02
    [3] imported_impsample02
    [4] test
    Enter the numbers of the datasets to delete (comma-separated): 0
    Deleted dataset: 207-realsense/2025-06-25-23-05-50/
    ```
    注：在交互中输入现有dataset的序号，即可完成删除

- 使用示例：（2）  
```(base) huangyue@booster-BVG:~/0625$ python delete_dataset51.py``` 
    ```
    Available datasets:
    [0] cl_207-realsense/2025-06-25-23-05-50/
    [1] cl_imported_impsample02
    [2] imported_impsample02
    [3] test
    Enter the numbers of the datasets to delete (comma-separated): 0,1,2
    Deleted dataset: cl_207-realsense/2025-06-25-23-05-50/
    Deleted dataset: cl_imported_impsample
    ```  
    注：如需同时删除多个脚本，用逗号隔开序号即可



# Part X: 使用session_start.py脚本，该脚本用于启动fiftyone app

###脚本使用方法：  
```python session_start.py --port <端口(0-65535)>```
- 使用示例：  
```(base) huangyue@booster-BVG:~/0625$ python session_start.py --port 55551```   
    ```
    Connected to FiftyOne on port 55551 at 0.0.0.0.
    If you are not connecting to a remote session, you may need to start a new session and specify a port
    You have launched a remote App on port 55551. To connect to this App from another
    machine, issue the following command to configure port forwarding:

    ssh -N -L 5151:127.0.0.1:55551 [<username>@]<hostname>

    where `[<username>@]<hostname>` refers to your current machine. The App can
    then be viewed in your browser at http://localhost:5151.

    Alternatively, if you have FiftyOne installed on your local machine, just run:

    fiftyone app connect --destination [<username>@]<hostname> --port 55551

    See https://docs.voxel51.com/user_guide/app.html#remote-sessions
    for more information about remote sessions.
    ```

    注：注意输入合法的端口即可成功启动fiftyone在浏览器打开，如果后续使用时发现fiftyone在浏览器无法成功刷新或者打开，重新跑一边此脚本即可


## 2025/07/14 更新内容
#### 1. 修复了文件夹中没有yaml文件无法成功import的问题  
#### 2. 修复了filteredupload中headers未定义的问题  
#### 3. 修复了可能非法的annokey变量的问题  
#### 4. 增加了新功能：上传到cvat的project可用参数--project控制，三种上传方式均可使用，默认project为```detection2507```  
- 使用示例：  
```(base) huangyue@booster-BVG:~/0625$ python everything.py imported_wo_yamlfile116 best1115.pt --upload --project testproject0714```  
注：按照以上命令能将task上传到testproject0714这个cvat project里，如果不存在此project则会自动新建

#### 5. 增加了新功能：import现有label的文件夹到fiftyone时可用可选参数--rename控制自定义改名，否则就默认为路径的最后一节为数据集名称。无论是否用rename参数，数据集名称均会加上imported_前缀  
- 使用示例：  
```(base) huangyue@booster-BVG:~/0625$ python everything.py asdhjk 0508.pt --importdataset 0707_37_prelabel/0707_37_prelabel/ --rename testset --prelabel --prelabel-class Ball ```  
注：按照以上命令能将新导入的数据集命名为imported_testset并预标注所有的Ball，原数据集名称位置可任意输入占位。