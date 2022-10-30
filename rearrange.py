import shutil, os
import pandas as pd

dir_path = 'raw_data/'
fds = ['test/','valid/','train/']
CLASSES = [' unripe', ' freshunripe', ' freshripe',' ripe', ' overripe',' rotten']

### Creating subfolders
# for fd in fds:
#     if not os.path.exists(fd):
#         os.makedirs(fd)
#     for c in CLASSES:
#         if not os.path.exists(fd+c):
#             os.makedirs(fd+c)

# print("subfolders exist")

# Moving the image files to their respective categories

ctr = 0

for fd in fds:
    df = pd.read_csv(dir_path+fd+'_classes.csv')
    print(df[:5])
    for c in CLASSES:
        image_list = list(df.loc[df[c]==1,'filename'])
        print(image_list[:5])
        for img in image_list: # Image Id
            old_path = dir_path+fd+img # Path to Images 
            new_path = fd+c
            if os.path.exists(dir_path+fd+img) == True:
                print(img+' exists')
                try:
                    shutil.copy(old_path, new_path)
                    print(img+' copied')
                except:
                    print(img + "does not exist")
                    ctr+=1

print("A total number of "+str(ctr)+" images missing")