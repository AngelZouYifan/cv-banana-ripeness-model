import os, random, shutil

srcs = ['train','valid','test']
dsts = ['avg_ds','dev_ds','leftover_ds']

class_names = os.listdir(srcs[0])
class_names.remove('.DS_Store')

# Create directories

# for dst in dsts[1:]:

dst = 'dev_ds2'

for src in srcs:
    for c in class_names:
        if not os.path.exists(dst+c):
            os.makedirs(os.path.join(dst+'/'+src, c))

num_move = {'train': {'avg_ds':1590,'dev_ds':100, 'dev_ds2':100},
            'valid':{'avg_ds':161,'dev_ds':50,'dev_ds2':20},
            'test':{'avg_ds':77,'dev_ds':50,'dev_ds2':20}}    # (avg, dev)


for src in srcs:
    for c in class_names:
        files = os.listdir(os.path.join(src,c))
        cnt = len(files)
        rdm_idxs = random.sample(range(cnt),num_move[src][dst])
        for i in range(cnt):
            if i in rdm_idxs:
                shutil.copy(os.path.join(src,c,files[i]),os.path.join(dst,src,c))
                # else:
                #     shutil.copy(os.path.join(src,c,files[i]),os.path.join(dsts[2],src,c))
        print("Finishing moving files in "+src+c+"to "+dst+src+c)
            # print(str(len(os.listdir(os.path.join(dst,src,c))))+" in"+dst+src+c)
            # print(str(len(os.listdir(os.path.join(dsts[2],src,c))))+" in"+dsts[2]+src+c)
